from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from .latent_kv import _pack_codes_q2,_unpack_codes_q2

@dataclass
class PackedQ2TokenGroup:
    packed: torch.Tensor
    minimum: torch.Tensor
    scale: torch.Tensor
    tokens: int
    rank: int
    @property
    def storage_bytes(self) -> int:
        return sum(t.numel()*t.element_size() for t in (self.packed,self.minimum,self.scale))

def pack_q2_token_group_scalar(x: torch.Tensor) -> PackedQ2TokenGroup:
    """Pack one [tokens, rank] latent group with one FP16 min/scale pair.

    The scale is shared across both the token and latent dimensions. This is the
    Run-5 metadata-amortization primitive. Production causal execution can keep
    the current incomplete group as an FP16 residual tail, then seal it into
    this representation when the group reaches group_tokens.
    """
    if x.ndim != 2:
        raise ValueError('expected [tokens, rank]')
    x=x.detach().float().contiguous();tokens,rank=x.shape
    mn=x.amin();mx=x.amax();sc=torch.clamp((mx-mn)/3.0,min=1e-8)
    q=torch.round((x-mn)/sc).clamp(0,3)
    p=_pack_codes_q2(q)
    return PackedQ2TokenGroup(p,mn.to(torch.float16).reshape(()),sc.to(torch.float16).reshape(()),tokens,rank)

def unpack_q2_token_group_scalar(g: PackedQ2TokenGroup) -> torch.Tensor:
    q=_unpack_codes_q2(g.packed,g.rank).float()
    return (q*g.scale.float()+g.minimum.float())[:g.tokens]

def grouped_latent_cache_bytes(*,layers:int,seq:int,kv_heads:int,head_dim:int,rank:int,group_tokens:int=3,bits:int=2,basis_bits:int=4,basis_scale_bytes:int=2,metric_fp16:bool=True,residual_tail_fp16:bool=True) -> int:
    """Worst-case resident bytes for grouped-scalar latent K/V.

    Includes:
      * Q2 coefficients for every token,
      * one FP16 min + FP16 scale for each completed/current K and V group,
      * Q4 K/V bases including FP16 row scales,
      * FP16 inverse-Gram matrices for both K and V,
      * optionally a worst-case FP16 incomplete-group residual tail.
    """
    vectors=layers*seq*kv_heads*2
    payload=math.ceil(vectors*rank*bits/8)
    groups=layers*kv_heads*2*math.ceil(seq/group_tokens)
    metadata=groups*4
    basis_values=layers*kv_heads*2*rank*head_dim
    basis_payload=math.ceil(basis_values*basis_bits/8)
    basis_scales=layers*kv_heads*2*rank*basis_scale_bytes
    metric=layers*kv_heads*2*rank*rank*2 if metric_fp16 else 0
    tail=0
    if residual_tail_fp16 and group_tokens>1:
        tail=layers*kv_heads*2*(group_tokens-1)*rank*2
    return payload+metadata+basis_payload+basis_scales+metric+tail

def reference_workspace_bytes(*,context:int,hidden:int,heads:int,rank:int,intermediate:int) -> int:
    """Run-5 reference scratch model.

    Generalizes the earlier context-64 formula instead of keeping scratch
    constant: one hidden input, one hidden residual, q/k/v staging, FFN staging,
    attention scores, and one dequantized latent-history buffer.
    """
    return (hidden+hidden+3*hidden+intermediate+heads*context+rank*context)*4
