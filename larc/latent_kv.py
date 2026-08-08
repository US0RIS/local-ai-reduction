from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from .q4_runtime import q4_rows,dequantize_q4_rows

FP8_E4M3=getattr(torch,'float8_e4m3fn',None)
FP8_E4M3_MIN_SUBNORMAL=2.0**-9


def _pack_codes_q2(q:torch.Tensor)->torch.Tensor:
    q=q.to(torch.uint8).contiguous();cols=q.shape[-1];pad=(-cols)%4
    if pad:q=torch.cat([q,torch.zeros((*q.shape[:-1],pad),dtype=torch.uint8,device=q.device)],dim=-1)
    return (q[...,0::4]|(q[...,1::4]<<2)|(q[...,2::4]<<4)|(q[...,3::4]<<6)).contiguous()

def _unpack_codes_q2(p:torch.Tensor,cols:int)->torch.Tensor:
    shape=(*p.shape[:-1],p.shape[-1]*4);q=torch.empty(shape,dtype=torch.uint8,device=p.device)
    q[...,0::4]=p&3;q[...,1::4]=(p>>2)&3;q[...,2::4]=(p>>4)&3;q[...,3::4]=(p>>6)&3
    return q[...,:cols]

def _fp8_bytes(x:torch.Tensor)->torch.Tensor:
    if FP8_E4M3 is None:raise RuntimeError('torch.float8_e4m3fn is required for FP8 metadata codec')
    return x.to(FP8_E4M3).view(torch.uint8).contiguous()

def _fp8_float(x:torch.Tensor)->torch.Tensor:
    if FP8_E4M3 is None:raise RuntimeError('torch.float8_e4m3fn is required for FP8 metadata codec')
    return x.view(FP8_E4M3).float()

def pack_q2_rows(x:torch.Tensor):
    x=x.detach().float().contiguous();rows,cols=x.shape;mn=x.amin(1);mx=x.amax(1);scale=torch.clamp((mx-mn)/3.0,min=1e-8);q=torch.round((x-mn[:,None])/scale[:,None]).clamp(0,3)
    return _pack_codes_q2(q),mn.to(torch.float16).contiguous(),scale.to(torch.float16).contiguous(),cols

def unpack_q2_rows(p,mn,scale,cols):
    q=_unpack_codes_q2(p,cols).float();return q*scale.float()[:,None]+mn.float()[:,None]

def pack_q2_rows_fp8meta(x:torch.Tensor):
    x=x.detach().float().contiguous();rows,cols=x.shape;mn=x.amin(1);mx=x.amax(1);scale=torch.clamp((mx-mn)/3.0,min=FP8_E4M3_MIN_SUBNORMAL);q=torch.round((x-mn[:,None])/scale[:,None]).clamp(0,3)
    return _pack_codes_q2(q),_fp8_bytes(mn),_fp8_bytes(scale),cols

def unpack_q2_rows_fp8meta(p,mn_bits,scale_bits,cols):
    q=_unpack_codes_q2(p,cols).float();mn=_fp8_float(mn_bits);scale=torch.clamp(_fp8_float(scale_bits),min=FP8_E4M3_MIN_SUBNORMAL);return q*scale[:,None]+mn[:,None]

def pack_q2_key_channels(x:torch.Tensor,group_tokens:int=64):
    x=x.detach().float().contiguous();tokens,rank=x.shape;groups=math.ceil(tokens/group_tokens);pad=groups*group_tokens-tokens
    xpad=torch.cat([x,torch.zeros((pad,rank),dtype=x.dtype,device=x.device)],0) if pad else x
    g=xpad.reshape(groups,group_tokens,rank);mn=g.amin(1);mx=g.amax(1);scale=torch.clamp((mx-mn)/3.0,min=1e-8);q=torch.round((g-mn[:,None,:])/scale[:,None,:]).clamp(0,3);p=_pack_codes_q2(q)
    return p,mn.to(torch.float16).contiguous(),scale.to(torch.float16).contiguous(),rank,tokens,group_tokens

def unpack_q2_key_channels(p,mn,scale,rank,tokens,group_tokens):
    q=_unpack_codes_q2(p,rank).float();x=q*scale.float()[:,None,:]+mn.float()[:,None,:];return x.reshape(-1,rank)[:tokens]

def fit_basis(x:torch.Tensor,rank:int):
    """Deterministic uncentered principal row basis via eig(X^T X)."""
    x=x.detach().float().reshape(-1,x.shape[-1]);rank=min(rank,*x.shape);cov=x.t()@x;_,v=torch.linalg.eigh(cov);return v[:,-rank:].t().flip(0).contiguous()

@dataclass
class QuantizedHeadBasisQ4:
    packed:torch.Tensor
    scale:torch.Tensor
    cols:int
    heads:int
    rank:int
    dequantized:torch.Tensor
    gram_inv_fp16:torch.Tensor|None=None
    @property
    def storage_bytes(self):
        n=self.packed.numel()*self.packed.element_size()+self.scale.numel()*self.scale.element_size()
        if self.gram_inv_fp16 is not None:n+=self.gram_inv_fp16.numel()*self.gram_inv_fp16.element_size()
        return n
    @property
    def gram_inv(self):return None if self.gram_inv_fp16 is None else self.gram_inv_fp16.float()

def quantize_head_basis_q4(basis:torch.Tensor,*,store_gram_inverse:bool=True)->QuantizedHeadBasisQ4:
    if basis.ndim!=3:raise ValueError('basis must be [heads, rank, head_dim]')
    h,r,d=basis.shape;p,s,c=q4_rows(basis.reshape(h*r,d));bh=dequantize_q4_rows(p,s,c).reshape(h,r,d);metric=None
    if store_gram_inverse:
        grams=bh@bh.transpose(-1,-2);eye=torch.eye(r,dtype=torch.float32,device=bh.device).expand(h,r,r);diag=torch.diagonal(grams,dim1=-2,dim2=-1).mean(-1).clamp_min(1e-8);metric=torch.linalg.inv(grams+eye*(diag[:,None,None]*1e-5)).to(torch.float16).contiguous()
    return QuantizedHeadBasisQ4(p,s,c,h,r,bh,metric)

@dataclass
class EncodedLatent:
    p:torch.Tensor;mn:torch.Tensor;scale:torch.Tensor;cols:int
    @property
    def storage_bytes(self):return sum(t.numel()*t.element_size() for t in (self.p,self.mn,self.scale))
@dataclass
class EncodedKeyChannels:
    p:torch.Tensor;mn:torch.Tensor;scale:torch.Tensor;rank:int;tokens:int;group_tokens:int
    @property
    def storage_bytes(self):return sum(t.numel()*t.element_size() for t in (self.p,self.mn,self.scale))

class LatentQ2KV:
    def __init__(self,k_basis:torch.Tensor,v_basis:torch.Tensor,*,fp8_metadata:bool=False):
        self.kq=quantize_head_basis_q4(k_basis) if k_basis.ndim==3 else None;self.vq=quantize_head_basis_q4(v_basis) if v_basis.ndim==3 else None
        self.k_basis=(self.kq.dequantized if self.kq else k_basis.float().contiguous());self.v_basis=(self.vq.dequantized if self.vq else v_basis.float().contiguous());self.fp8_metadata=fp8_metadata
    def encode(self,k:torch.Tensor,v:torch.Tensor):
        ks=k.reshape(-1,k.shape[-1]).float()@self.k_basis.t();vs=v.reshape(-1,v.shape[-1]).float()@self.v_basis.t();fn=pack_q2_rows_fp8meta if self.fp8_metadata else pack_q2_rows;pk,mk,sk,ck=fn(ks);pv,mv,sv,cv=fn(vs);return EncodedLatent(pk,mk,sk,ck),EncodedLatent(pv,mv,sv,cv),k.shape[:-1]
    def decode(self,ek:EncodedLatent,ev:EncodedLatent,prefix_shape):
        fn=unpack_q2_rows_fp8meta if self.fp8_metadata else unpack_q2_rows;kl=fn(ek.p,ek.mn,ek.scale,ek.cols);vl=fn(ev.p,ev.mn,ev.scale,ev.cols)
        if self.kq is not None:kl=kl@self.kq.gram_inv
        if self.vq is not None:vl=vl@self.vq.gram_inv
        return (kl@self.k_basis).reshape(*prefix_shape,-1),(vl@self.v_basis).reshape(*prefix_shape,-1)

class KIVILatentQ2KV:
    def __init__(self,k_basis:torch.Tensor,v_basis:torch.Tensor,group_tokens:int=64):
        if k_basis.ndim!=3 or v_basis.ndim!=3:raise ValueError('KIVI latent bases must be [heads,rank,head_dim]')
        self.kq=quantize_head_basis_q4(k_basis);self.vq=quantize_head_basis_q4(v_basis);self.k_basis=self.kq.dequantized;self.v_basis=self.vq.dequantized;self.group_tokens=group_tokens
    @property
    def basis_storage_bytes(self):return self.kq.storage_bytes+self.vq.storage_bytes
    def encode_head(self,k:torch.Tensor,v:torch.Tensor,head:int):
        kb=self.k_basis[head];vb=self.v_basis[head];ks=k.reshape(-1,k.shape[-1]).float()@kb.t();vs=v.reshape(-1,v.shape[-1]).float()@vb.t();pk,mk,sk,r,n,g=pack_q2_key_channels(ks,self.group_tokens);pv,mv,sv,cv=pack_q2_rows(vs);return EncodedKeyChannels(pk,mk,sk,r,n,g),EncodedLatent(pv,mv,sv,cv),k.shape[:-1]
    def attention_head(self,q:torch.Tensor,ek:EncodedKeyChannels,ev:EncodedLatent,head:int):
        kl=unpack_q2_key_channels(ek.p,ek.mn,ek.scale,ek.rank,ek.tokens,ek.group_tokens);vl=unpack_q2_rows(ev.p,ev.mn,ev.scale,ev.cols);kb=self.k_basis[head];vb=self.v_basis[head];ql=(q.float()@kb.t())@self.kq.gram_inv[head];a=torch.softmax((kl@ql)/math.sqrt(q.shape[-1]),dim=-1);vlat=(a@vl)@self.vq.gram_inv[head];return vlat@vb

def _q4_basis_bytes(*,layers:int,kv_heads:int,rank:int,head_dim:int,scale_bytes:int=2)->int:
    rows=layers*kv_heads*2*rank;return rows*math.ceil(head_dim/2)+rows*scale_bytes

def cache_bytes(*,layers:int,seq:int,kv_heads:int,head_dim:int,rank:int,bits:int=2,scale_bytes_per_vector:int=4,basis_bits:int=4,basis_scale_bytes:int=2,both_metrics_fp16:bool=True):
    vectors=layers*seq*kv_heads*2;latent_payload=math.ceil(vectors*rank*bits/8);quant_meta=vectors*scale_bytes_per_vector
    if basis_bits!=4:raise ValueError('v0.3 byte accounting currently defines Q4 basis storage only')
    basis_payload=_q4_basis_bytes(layers=layers,kv_heads=kv_heads,rank=rank,head_dim=head_dim,scale_bytes=basis_scale_bytes);metric=layers*kv_heads*rank*rank*2*(2 if both_metrics_fp16 else 1);return latent_payload+quant_meta+basis_payload+metric

def kivi_latent_cache_bytes(*,layers:int,seq:int,kv_heads:int,head_dim:int,rank:int,bits:int=2,group_tokens:int=64,basis_bits:int=4,basis_scale_bytes:int=2,tail_fp16:bool=False,key_metric_fp16:bool=True,value_metric_fp16:bool=True):
    vectors=layers*seq*kv_heads;payload=2*math.ceil(vectors*rank*bits/8);key_groups=layers*kv_heads*math.ceil(seq/group_tokens);key_meta=key_groups*rank*4;value_meta=vectors*4
    if basis_bits!=4:raise ValueError('v0.3 byte accounting currently defines Q4 basis storage only')
    basis_payload=_q4_basis_bytes(layers=layers,kv_heads=kv_heads,rank=rank,head_dim=head_dim,scale_bytes=basis_scale_bytes);metric=(layers*kv_heads*rank*rank*2 if key_metric_fp16 else 0)+(layers*kv_heads*rank*rank*2 if value_metric_fp16 else 0);tail=layers*kv_heads*2*group_tokens*rank if tail_fp16 else 0;return payload+key_meta+value_meta+basis_payload+metric+tail

def fp16_cache_bytes(*,layers:int,seq:int,kv_heads:int,head_dim:int):return layers*seq*kv_heads*head_dim*2*2
