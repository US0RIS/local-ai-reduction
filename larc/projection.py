"""Activation-subspace projection bundles for LARC."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .q4 import quantize_q4,dequantize_q4
from .q8 import quantize_q8,dequantize_q8

@dataclass
class ProjectionBundle:
    input_dim: int
    rank: int
    precision: str
    basis: Any
    operators: list[Any]
    @property
    def storage_bytes(self) -> int: return self.basis.storage_bytes + sum(x.storage_bytes for x in self.operators)

def _q(x,p):
    if p=="q4": return quantize_q4(x)
    if p=="q8": return quantize_q8(x)
    raise ValueError(p)
def _dq(x,p):
    if p=="q4": return dequantize_q4(x)
    if p=="q8": return dequantize_q8(x)
    raise ValueError(p)

def fit_projection_bundle(weights:list[np.ndarray],calibration_x:np.ndarray,rank:int,precision:str="q8") -> ProjectionBundle:
    if not weights: raise ValueError("at least one operator is required")
    n=weights[0].shape[1]
    if any(w.ndim!=2 or w.shape[1]!=n for w in weights): raise ValueError("all operators must share input dimension")
    x=np.asarray(calibration_x,dtype=np.float32)
    if x.ndim!=2 or x.shape[0]!=n: raise ValueError("calibration_x must have shape [input_dim, samples]")
    rank=min(rank,n,x.shape[1]); u,_,_=np.linalg.svd(x,full_matrices=False); basis=u[:,:rank].astype(np.float32)
    projected=[np.asarray(w,dtype=np.float32)@basis for w in weights]
    return ProjectionBundle(n,rank,precision,_q(basis,precision),[_q(a,precision) for a in projected])

def decode_weights(bundle:ProjectionBundle)->list[np.ndarray]:
    u=_dq(bundle.basis,bundle.precision); return [_dq(a,bundle.precision)@u.T for a in bundle.operators]

def run_bundle(bundle:ProjectionBundle,x:np.ndarray)->list[np.ndarray]:
    u=_dq(bundle.basis,bundle.precision); z=u.T@np.asarray(x,dtype=np.float32); return [_dq(a,bundle.precision)@z for a in bundle.operators]
