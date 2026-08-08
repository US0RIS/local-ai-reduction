from __future__ import annotations
from dataclasses import dataclass
import math
import torch


def _pack_codes_q2(q: torch.Tensor) -> torch.Tensor:
    q=q.to(torch.uint8).contiguous(); cols=q.shape[-1]; pad=(-cols)%4
    if pad:
        q=torch.cat([q,torch.zeros((*q.shape[:-1],pad),dtype=torch.uint8,device=q.device)],dim=-1)
    return (q[...,0::4] | (q[...,1::4]<<2) | (q[...,2::4]<<4) | (q[...,3::4]<<6)).contiguous()


def _unpack_codes_q2(p: torch.Tensor,cols:int) -> torch.Tensor:
    shape=(*p.shape[:-1],p.shape[-1]*4); q=torch.empty(shape,dtype=torch.uint8,device=p.device)
    q[...,0::4]=p&3; q[...,1::4]=(p>>2)&3; q[...,2::4]=(p>>4)&3; q[...,3::4]=(p>>6)&3
    return q[...,:cols]


def pack_q2_rows(x: torch.Tensor):
    """Asymmetric 2-bit per-token/row quantization with fp16 min+scale."""
    x=x.detach().float().contiguous(); rows,cols=x.shape
    mn=x.amin(1); mx=x.amax(1); scale=torch.clamp((mx-mn)/3.0,min=1e-8)
    q=torch.round((x-mn[:,None])/scale[:,None]).clamp(0,3)
    return _pack_codes_q2(q),mn.to(torch.float16).contiguous(),scale.to(torch.float16).contiguous(),cols


def unpack_q2_rows(p,mn,scale,cols):
    q=_unpack_codes_q2(p,cols).float()
    return q*scale.float()[:,None]+mn.float()[:,None]


def pack_q2_key_channels(x:torch.Tensor,group_tokens:int=64):
    """KIVI-style key quantization: group tokens, quantize each latent channel.

    x is [tokens, latent_rank]. Metadata is one FP16 min/scale pair per
    (token-group, latent-channel), rather than one pair per token.
    """
    x=x.detach().float().contiguous(); tokens,rank=x.shape; groups=math.ceil(tokens/group_tokens)
    pad=groups*group_tokens-tokens
    if pad: xpad=torch.cat([x,torch.zeros((pad,rank),dtype=x.dtype,device=x.device)],0)
    else: xpad=x
    g=xpad.reshape(groups,group_tokens,rank)
    mn=g.amin(1); mx=g.amax(1); scale=torch.clamp((mx-mn)/3.0,min=1e-8)
    q=torch.round((g-mn[:,None,:])/scale[:,None,:]).clamp(0,3)
    # Pack along latent rank. Shape [groups, group_tokens, ceil(rank/4)].
    p=_pack_codes_q2(q)
    return p,mn.to(torch.float16).contiguous(),scale.to(torch.float16).contiguous(),rank,tokens,group_tokens


def unpack_q2_key_channels(p,mn,scale,rank,tokens,group_tokens):
    q=_unpack_codes_q2(p,rank).float()
    x=q*scale.float()[:,None,:]+mn.float()[:,None,:]
    return x.reshape(-1,rank)[:tokens]


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


@dataclass
class EncodedKeyChannels:
    p:torch.Tensor; mn:torch.Tensor; scale:torch.Tensor; rank:int; tokens:int; group_tokens:int
    @property
    def storage_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.p,self.mn,self.scale))


class LatentQ2KV:
    """First LARC latent KV prototype: K/V both quantized per-token."""
    def __init__(self,k_basis:torch.Tensor,v_basis:torch.Tensor):
        self.k_basis=k_basis.float().contiguous(); self.v_basis=v_basis.float().contiguous()
    def encode(self,k:torch.Tensor,v:torch.Tensor):
        ks=k.reshape(-1,k.shape[-1]).float()@self.k_basis.t(); vs=v.reshape(-1,v.shape[-1]).float()@self.v_basis.t()
        pk,mk,sk,ck=pack_q2_rows(ks); pv,mv,sv,cv=pack_q2_rows(vs)
        return EncodedLatent(pk,mk,sk,ck),EncodedLatent(pv,mv,sv,cv),k.shape[:-1]
    def decode(self,ek:EncodedLatent,ev:EncodedLatent,prefix_shape):
        k=(unpack_q2_rows(ek.p,ek.mn,ek.scale,ek.cols)@self.k_basis).reshape(*prefix_shape,-1)
        v=(unpack_q2_rows(ev.p,ev.mn,ev.scale,ev.cols)@self.v_basis).reshape(*prefix_shape,-1); return k,v
    def attention(self,q:torch.Tensor,ek:EncodedLatent,ev:EncodedLatent,prefix_shape):
        kl=unpack_q2_rows(ek.p,ek.mn,ek.scale,ek.cols); vl=unpack_q2_rows(ev.p,ev.mn,ev.scale,ev.cols); ql=q.float()@self.k_basis.t()
        a=torch.softmax((ql@kl.t())/math.sqrt(q.shape[-1]),dim=-1); return (a@vl)@self.v_basis


class KIVILatentQ2KV:
    """LARC v0.2 latent Q2 KV with KIVI-style asymmetric quantization.

    Keys: project to rank-r latent coordinates, then quantize per latent channel
    across token groups. Values: project to a separate rank-r basis and quantize
    per token. Historical full-dimensional K/V are never required resident.
    """
    def __init__(self,k_basis:torch.Tensor,v_basis:torch.Tensor,group_tokens:int=64):
        self.k_basis=k_basis.float().contiguous(); self.v_basis=v_basis.float().contiguous(); self.group_tokens=group_tokens
    def encode(self,k:torch.Tensor,v:torch.Tensor):
        ks=k.reshape(-1,k.shape[-1]).float()@self.k_basis.t(); vs=v.reshape(-1,v.shape[-1]).float()@self.v_basis.t()
        pk,mk,sk,r,n,g=pack_q2_key_channels(ks,self.group_tokens); pv,mv,sv,cv=pack_q2_rows(vs)
        return EncodedKeyChannels(pk,mk,sk,r,n,g),EncodedLatent(pv,mv,sv,cv),k.shape[:-1]
    def decode(self,ek:EncodedKeyChannels,ev:EncodedLatent,prefix_shape):
        kl=unpack_q2_key_channels(ek.p,ek.mn,ek.scale,ek.rank,ek.tokens,ek.group_tokens); vl=unpack_q2_rows(ev.p,ev.mn,ev.scale,ev.cols)
        return (kl@self.k_basis).reshape(*prefix_shape,-1),(vl@self.v_basis).reshape(*prefix_shape,-1)
    def attention(self,q:torch.Tensor,ek:EncodedKeyChannels,ev:EncodedLatent,prefix_shape):
        kl=unpack_q2_key_channels(ek.p,ek.mn,ek.scale,ek.rank,ek.tokens,ek.group_tokens); vl=unpack_q2_rows(ev.p,ev.mn,ev.scale,ev.cols); ql=q.float()@self.k_basis.t()
        a=torch.softmax((ql@kl.t())/math.sqrt(q.shape[-1]),dim=-1); return (a@vl)@self.v_basis


def cache_bytes(*,layers:int,seq:int,kv_heads:int,head_dim:int,rank:int,bits:int=2,scale_bytes_per_vector:int=4,basis_bits:int=4):
    """v0.1 row/row latent-cache accounting."""
    vectors=layers*seq*kv_heads*2; latent_payload=math.ceil(vectors*rank*bits/8); quant_meta=vectors*scale_bytes_per_vector
    basis_values=layers*kv_heads*2*rank*head_dim; basis_payload=math.ceil(basis_values*basis_bits/8)
    return latent_payload+quant_meta+basis_payload


def kivi_latent_cache_bytes(*,layers:int,seq:int,kv_heads:int,head_dim:int,rank:int,bits:int=2,group_tokens:int=64,basis_bits:int=4,tail_fp16:bool=False):
    """Stored bytes for v0.2 K-channel/V-token latent Q2 cache.

    Optional tail_fp16 accounts for a KIVI-like unquantized current token group.
    The default assumes incremental repacking of the partial group.
    """
    vectors=layers*seq*kv_heads
    payload=2*math.ceil(vectors*rank*bits/8)
    key_groups=layers*kv_heads*math.ceil(seq/group_tokens); key_meta=key_groups*rank*4
    value_meta=vectors*4
    basis_values=layers*kv_heads*2*rank*head_dim; basis_payload=math.ceil(basis_values*basis_bits/8)
    tail=0
    if tail_fp16: tail=layers*kv_heads*2*group_tokens*rank
    return payload+key_meta+value_meta+basis_payload+tail


def fp16_cache_bytes(*,layers:int,seq:int,kv_heads:int,head_dim:int):
    return layers*seq*kv_heads*head_dim*2*2
