from __future__ import annotations
from dataclasses import dataclass
import math
import torch


def pack_q2_rows(x: torch.Tensor):
    """Asymmetric 2-bit row quantization with fp16 min/scale per row."""
    x=x.detach().float().contiguous(); rows,cols=x.shape
    mn=x.amin(1); mx=x.amax(1); scale=torch.clamp((mx-mn)/3.0,min=1e-8)
    q=torch.round((x-mn[:,None])/scale[:,None]).clamp(0,3).to(torch.uint8)
    pad=(-cols)%4
    if pad: q=torch.cat([q,torch.zeros((rows,pad),dtype=torch.uint8)],1)
    p=(q[:,0::4] | (q[:,1::4]<<2) | (q[:,2::4]<<4) | (q[:,3::4]<<6)).contiguous()
    return p,mn.to(torch.float16).contiguous(),scale.to(torch.float16).contiguous(),cols


def unpack_q2_rows(p,mn,scale,cols):
    rows=p.shape[0]; q=torch.empty((rows,p.shape[1]*4),dtype=torch.uint8,device=p.device)
    q[:,0::4]=p&3; q[:,1::4]=(p>>2)&3; q[:,2::4]=(p>>4)&3; q[:,3::4]=(p>>6)&3
    return q[:,:cols].float()*scale.float()[:,None]+mn.float()[:,None]


def fit_basis(x:torch.Tensor,rank:int):
    x=x.detach().float().reshape(-1,x.shape[-1]); rank=min(rank,*x.shape)
    _,_,v=torch.pca_lowrank(x,q=min(rank+4,min(x.shape)),center=False,niter=3)
    return v[:,:rank].t().contiguous()


@dataclass
class EncodedLatent:
    p:torch.Tensor; mn:torch.Tensor; scale:torch.Tensor; cols:int
    @property
    def storage_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.p,self.mn,self.scale))


class LatentQ2KV:
    """Reference latent KV representation.

    K/V vectors are projected into learned rank-r bases and only the latent
    coefficients are cached at 2 bits. Attention can score against latent K
    after projecting Q into the K basis, and reconstruct only the weighted V
    aggregate, so full historical K/V vectors need not be resident.
    """
    def __init__(self,k_basis:torch.Tensor,v_basis:torch.Tensor):
        self.k_basis=k_basis.float().contiguous()
        self.v_basis=v_basis.float().contiguous()
    def encode(self,k:torch.Tensor,v:torch.Tensor):
        ks=k.reshape(-1,k.shape[-1]).float()@self.k_basis.t()
        vs=v.reshape(-1,v.shape[-1]).float()@self.v_basis.t()
        pk,mk,sk,ck=pack_q2_rows(ks); pv,mv,sv,cv=pack_q2_rows(vs)
        return EncodedLatent(pk,mk,sk,ck),EncodedLatent(pv,mv,sv,cv),k.shape[:-1]
    def decode(self,ek:EncodedLatent,ev:EncodedLatent,prefix_shape):
        k=(unpack_q2_rows(ek.p,ek.mn,ek.scale,ek.cols)@self.k_basis).reshape(*prefix_shape,-1)
        v=(unpack_q2_rows(ev.p,ev.mn,ev.scale,ev.cols)@self.v_basis).reshape(*prefix_shape,-1)
        return k,v
    def attention(self,q:torch.Tensor,ek:EncodedLatent,ev:EncodedLatent,prefix_shape):
        kl=unpack_q2_rows(ek.p,ek.mn,ek.scale,ek.cols)
        vl=unpack_q2_rows(ev.p,ev.mn,ev.scale,ev.cols)
        ql=q.float()@self.k_basis.t()
        a=torch.softmax((ql@kl.t())/math.sqrt(q.shape[-1]),dim=-1)
        return (a@vl)@self.v_basis


def cache_bytes(*,layers:int,seq:int,kv_heads:int,head_dim:int,rank:int,bits:int=2,scale_bytes_per_vector:int=4,basis_bits:int=4):
    """Byte accounting for K+V latent cache and per-layer/head K/V bases."""
    vectors=layers*seq*kv_heads*2
    latent_payload=math.ceil(vectors*rank*bits/8)
    quant_meta=vectors*scale_bytes_per_vector
    basis_values=layers*kv_heads*2*rank*head_dim
    basis_payload=math.ceil(basis_values*basis_bits/8)
    return latent_payload+quant_meta+basis_payload


def fp16_cache_bytes(*,layers:int,seq:int,kv_heads:int,head_dim:int):
    return layers*seq*kv_heads*head_dim*2*2
