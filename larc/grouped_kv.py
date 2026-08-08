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

def grouped_latent_cache_bytes(*,layers:int,seq:int,kv_heads:int,head_dim:int,rank:int,group_tokens:int=3,bits:int=2,basis_bits:int=4,basis_scale_bytes:int=2,metric_fp16:bool=True,residual_tail_fp16:bool=True,basis_sets:int|None=None) -> int:
    """Worst-case resident bytes for grouped-scalar latent K/V.

    `layers` counts logical KV histories. `basis_sets` counts physically distinct
    K/V basis+metric objects. Recurrent/shared models may have many logical
    histories but only one physical basis set.
    """
    if basis_sets is None:
        basis_sets=layers
    vectors=layers*seq*kv_heads*2
    payload=math.ceil(vectors*rank*bits/8)
    groups=layers*kv_heads*2*math.ceil(seq/group_tokens)
    metadata=groups*4
    basis_values=basis_sets*kv_heads*2*rank*head_dim
    basis_payload=math.ceil(basis_values*basis_bits/8)
    basis_scales=basis_sets*kv_heads*2*rank*basis_scale_bytes
    metric=basis_sets*kv_heads*2*rank*rank*2 if metric_fp16 else 0
    tail=0
    if residual_tail_fp16 and group_tokens>1:
        tail=layers*kv_heads*2*(group_tokens-1)*rank*2
    return payload+metadata+basis_payload+basis_scales+metric+tail

def reference_workspace_bytes(*,context:int,hidden:int,heads:int,rank:int,intermediate:int) -> int:
    return (hidden+hidden+3*hidden+intermediate+heads*context+rank*context)*4
