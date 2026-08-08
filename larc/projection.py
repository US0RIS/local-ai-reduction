"""Activation-subspace projection bundles for LARC."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .q4 import quantize_q4,dequantize_q4
from .q8 import quantize_q8,dequantize_q8

@dataclass
class ProjectionBundle:
    input_dim:int
    rank:int
    precision:str
    basis:Any
    operators:list[Any]
    fit_method:str='quantized_activation_lstsq'
    @property
    def storage_bytes(self)->int:
        return self.basis.storage_bytes+sum(x.storage_bytes for x in self.operators)

def _q(x,p):
    if p=='q4':return quantize_q4(x)
    if p=='q8':return quantize_q8(x)
    raise ValueError(p)

def _dq(x,p):
    if p=='q4':return dequantize_q4(x)
    if p=='q8':return dequantize_q8(x)
    raise ValueError(p)

def _fit_A_activation_lstsq(weight:np.ndarray,u_hat:np.ndarray,x:np.ndarray,ridge:float=1e-6)->np.ndarray:
    """Solve min_A ||W X - A (U_hat^T X)||_F^2.

    This fits against the *stored/quantized* basis rather than computing W@U
    before quantization. Ridge is scaled to the average latent covariance so it
    is dimensionless across calibration sets.
    """
    z=u_hat.T@x
    y=np.asarray(weight,dtype=np.float32)@x
    gram=z@z.T
    lam=float(ridge)*max(float(np.trace(gram))/max(gram.shape[0],1),1e-12)
    reg=gram+np.eye(gram.shape[0],dtype=np.float32)*lam
    return ((y@z.T)@np.linalg.pinv(reg,rcond=1e-7)).astype(np.float32)

def fit_projection_bundle(weights:list[np.ndarray],calibration_x:np.ndarray,rank:int,precision:str='q8')->ProjectionBundle:
    if not weights:raise ValueError('at least one operator is required')
    n=weights[0].shape[1]
    if any(w.ndim!=2 or w.shape[1]!=n for w in weights):raise ValueError('all operators must share input dimension')
    x=np.asarray(calibration_x,dtype=np.float32)
    if x.ndim!=2 or x.shape[0]!=n:raise ValueError('calibration_x must have shape [input_dim, samples]')
    rank=min(rank,n,x.shape[1])
    u,_,_=np.linalg.svd(x,full_matrices=False)
    basis_float=u[:,:rank].astype(np.float32)
    basis_q=_q(basis_float,precision)
    u_hat=_dq(basis_q,precision)
    projected=[_fit_A_activation_lstsq(w,u_hat,x) for w in weights]
    return ProjectionBundle(n,rank,precision,basis_q,[_q(a,precision) for a in projected])

def decode_weights(bundle:ProjectionBundle)->list[np.ndarray]:
    u=_dq(bundle.basis,bundle.precision)
    return [_dq(a,bundle.precision)@u.T for a in bundle.operators]

def run_bundle(bundle:ProjectionBundle,x:np.ndarray)->list[np.ndarray]:
    u=_dq(bundle.basis,bundle.precision)
    z=u.T@np.asarray(x,dtype=np.float32)
    return [_dq(a,bundle.precision)@z for a in bundle.operators]
