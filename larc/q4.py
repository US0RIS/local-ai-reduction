from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Q4Matrix:
    shape: tuple[int, int]
    scales: np.ndarray
    packed: np.ndarray
    @property
    def storage_bytes(self) -> int: return int(self.scales.nbytes + self.packed.nbytes)

def quantize_q4(matrix: np.ndarray) -> Q4Matrix:
    x=np.asarray(matrix,dtype=np.float32); rows,cols=x.shape
    scales=np.maximum(np.max(np.abs(x),axis=1)/7.0,1e-12).astype(np.float32)
    q=np.rint(x/scales[:,None]).clip(-7,7).astype(np.int8); codes=(q+8).astype(np.uint8)
    if cols%2: codes=np.pad(codes,((0,0),(0,1)),constant_values=8)
    packed=(codes[:,0::2] | (codes[:,1::2]<<4)).reshape(-1)
    return Q4Matrix((rows,cols),scales.astype(np.float16),packed)

def dequantize_q4(qm: Q4Matrix) -> np.ndarray:
    rows,cols=qm.shape; pair=(cols+1)//2; p=qm.packed.reshape(rows,pair)
    codes=np.empty((rows,pair*2),dtype=np.uint8); codes[:,0::2]=p&15; codes[:,1::2]=p>>4
    return (codes[:,:cols].astype(np.int16)-8).astype(np.float32)*qm.scales.astype(np.float32)[:,None]
