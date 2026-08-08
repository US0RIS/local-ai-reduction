from __future__ import annotations
import torch
from torch import nn

Q4_MIN=-8
Q4_MAX=7

def _q4_scale(x: torch.Tensor) -> torch.Tensor:
    pos=torch.clamp(x.amax(dim=1),min=0.0)/float(Q4_MAX)
    neg=torch.clamp(-x.amin(dim=1),min=0.0)/float(-Q4_MIN)
    return torch.clamp(torch.maximum(pos,neg),min=1e-8)

def q4_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Canonical LARC Q4_ROW: signed -8..7, two nibbles/byte, FP16 row scale."""
    w=weight.detach().float().contiguous()
    if w.ndim!=2:
        raise ValueError('Q4_ROW expects a rank-2 matrix')
    out,cols=w.shape
    scale=_q4_scale(w).to(torch.float16)
    q=torch.round(w/scale.float()[:,None]).clamp(Q4_MIN,Q4_MAX).to(torch.int8)
    code=(q.to(torch.int16)-Q4_MIN).to(torch.uint8)
    if cols&1:
        code=torch.cat([code,torch.full((out,1),-Q4_MIN,dtype=torch.uint8,device=code.device)],dim=1)
    packed=(code[:,0::2] | (code[:,1::2]<<4)).contiguous()
    return packed,scale.contiguous(),cols

def _unpack_rows(packed: torch.Tensor,cols:int) -> torch.Tensor:
    out=packed.shape[0]
    codes=torch.empty((out,packed.shape[1]*2),dtype=torch.uint8,device=packed.device)
    codes[:,0::2]=packed&0x0F
    codes[:,1::2]=packed>>4
    return (codes[:,:cols].to(torch.int16)+Q4_MIN).to(torch.int8).contiguous()

def dequantize_q4_rows(packed:torch.Tensor,scale:torch.Tensor,cols:int)->torch.Tensor:
    return _unpack_rows(packed,cols).float()*scale.float()[:,None]

def q4_mm(x:torch.Tensor,packed:torch.Tensor,scale:torch.Tensor,cols:int,tile_rows:int=256)->torch.Tensor:
    """Compute x @ W.T from canonical packed Q4 rows without dense float W.

    Activations are dynamically quantized to INT8. Weight rows are unpacked only
    one output tile at a time to INT8, consumed by INT8xINT8->INT32 matmul, then
    discarded. This is a bounded-tile reference implementation, not a claim
    about exact allocator peak scratch.
    """
    shape=x.shape
    x2=x.reshape(-1,shape[-1]).float().contiguous()
    if x2.shape[1]!=cols:
        raise ValueError((x2.shape,cols))
    xs=torch.clamp(x2.abs().amax(dim=1)/127.0,min=1e-8)
    xq=torch.round(x2/xs[:,None]).clamp(-127,127).to(torch.int8).contiguous()
    ys=[]
    for s in range(0,packed.shape[0],tile_rows):
        p=packed[s:s+tile_rows]
        qw=_unpack_rows(p,cols)
        acc=torch._int_mm(xq,qw.t().contiguous())
        ys.append(acc.float()*xs[:,None]*scale[s:s+len(p)].float()[None,:])
    y=torch.cat(ys,dim=1)
    return y.reshape(*shape[:-1],packed.shape[0])

class Q4Basis(nn.Module):
    def __init__(self,rows:torch.Tensor):
        super().__init__();p,s,c=q4_rows(rows)
        self.register_buffer('packed',p);self.register_buffer('scale',s);self.cols=c
    @property
    def rank(self):return self.packed.shape[0]
    @property
    def storage_bytes(self):return self.packed.numel()+self.scale.numel()*self.scale.element_size()
    def dequantized(self):return dequantize_q4_rows(self.packed,self.scale,self.cols)
    def forward(self,x):return q4_mm(x,self.packed,self.scale,self.cols)

class Q4ProjectedLinear(nn.Module):
    def __init__(self,basis:Q4Basis,projected:torch.Tensor,bias:torch.Tensor|None=None):
        super().__init__();self.basis=basis;self.in_features=basis.cols;self.out_features=projected.shape[0];p,s,c=q4_rows(projected)
        self.register_buffer('packedA',p);self.register_buffer('scaleA',s);self.colsA=c
        if bias is None:self.bias=None
        else:self.register_buffer('bias',bias.detach().float().contiguous())
    @property
    def storage_bytes_excluding_shared_basis(self):
        n=self.packedA.numel()+self.scaleA.numel()*self.scaleA.element_size()
        if self.bias is not None:n+=self.bias.numel()*self.bias.element_size()
        return n
    def forward(self,x):
        z=self.basis(x);y=q4_mm(z,self.packedA,self.scaleA,self.colsA)
        return y if self.bias is None else y+self.bias

class Q4VocabFactors(nn.Module):
    def __init__(self,coeff:torch.Tensor,right_basis:torch.Tensor):
        super().__init__()
        pc,sc,cc=q4_rows(coeff);pv,sv,cv=q4_rows(right_basis);pvt,svt,cvt=q4_rows(right_basis.t().contiguous())
        self.register_buffer('pC',pc);self.register_buffer('sC',sc);self.cC=cc
        self.register_buffer('pV',pv);self.register_buffer('sV',sv);self.cV=cv
        self.register_buffer('pVT',pvt);self.register_buffer('sVT',svt);self.cVT=cvt
    @property
    def storage_bytes(self):return sum(t.numel()*t.element_size() for t in [self.pC,self.sC,self.pV,self.sV,self.pVT,self.sVT])
    def embed(self,ids):
        flat=ids.reshape(-1);p=self.pC[flat];q=_unpack_rows(p,self.cC).float();c=q*self.sC[flat].float()[:,None]
        e=q4_mm(c,self.pV,self.sV,self.cV)
        return e.reshape(*ids.shape,self.pV.shape[0])
    def logits(self,h):
        z=q4_mm(h,self.pVT,self.sVT,self.cVT)
        return q4_mm(z,self.pC,self.sC,self.cC,tile_rows=512)

class LARCEmbedding(nn.Module):
    def __init__(self,f):super().__init__();self.f=f;self.num_embeddings=f.pC.shape[0];self.embedding_dim=f.pV.shape[0]
    def forward(self,ids):return self.f.embed(ids)

class LARCHead(nn.Module):
    def __init__(self,f):super().__init__();self.f=f;self.in_features=f.pV.shape[0];self.out_features=f.pC.shape[0]
    def forward(self,h):return self.f.logits(h)

def fit_basis(x:torch.Tensor,rank:int)->torch.Tensor:
    x=x.detach().float().reshape(-1,x.shape[-1]);rank=max(1,min(rank,x.shape[0],x.shape[1]));q=min(rank+8,min(x.shape))
    _,_,v=torch.pca_lowrank(x,q=q,center=False,niter=2)
    return v[:,:rank].t().contiguous()
