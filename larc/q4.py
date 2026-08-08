from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Q4_MIN=-8
Q4_MAX=7

@dataclass
class Q4Matrix:
    shape: tuple[int, int]
    scales: np.ndarray
    packed: np.ndarray
    @property
    def storage_bytes(self) -> int:
        return int(self.scales.nbytes + self.packed.nbytes)

def _row_scales(x: np.ndarray) -> np.ndarray:
    """Range-aware signed INT4 scale for codes -8..7.

    Scale is chosen so both the most-positive and most-negative element fit
    without clipping. This uses all 16 nibble codes while preserving zero at
    code 8.
    """
    pos=np.maximum(np.max(x,axis=1),0.0)/Q4_MAX
    neg=np.maximum(-np.min(x,axis=1),0.0)/(-Q4_MIN)
    return np.maximum(np.maximum(pos,neg),1e-12).astype(np.float32)

def quantize_q4(matrix: np.ndarray) -> Q4Matrix:
    x=np.asarray(matrix,dtype=np.float32)
    if x.ndim!=2:
        raise ValueError('Q4_ROW expects a rank-2 matrix')
    rows,cols=x.shape
    scales=_row_scales(x)
    q=np.rint(x/scales[:,None]).clip(Q4_MIN,Q4_MAX).astype(np.int8)
    codes=(q.astype(np.int16)-Q4_MIN).astype(np.uint8)  # -8 -> 0, 0 -> 8, 7 -> 15
    if cols%2:
        codes=np.pad(codes,((0,0),(0,1)),constant_values=-Q4_MIN)
    packed=(codes[:,0::2] | (codes[:,1::2]<<4)).reshape(-1)
    return Q4Matrix((rows,cols),scales.astype(np.float16),packed)

def dequantize_q4(qm: Q4Matrix) -> np.ndarray:
    rows,cols=qm.shape
    pair=(cols+1)//2
    p=qm.packed.reshape(rows,pair)
    codes=np.empty((rows,pair*2),dtype=np.uint8)
    codes[:,0::2]=p&15
    codes[:,1::2]=p>>4
    q=(codes[:,:cols].astype(np.int16)+Q4_MIN).astype(np.float32)
    return q*qm.scales.astype(np.float32)[:,None]
