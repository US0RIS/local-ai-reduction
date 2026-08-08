from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Q8Matrix:
    shape: tuple[int, int]
    scales: np.ndarray
    values: np.ndarray
    @property
    def storage_bytes(self) -> int: return int(self.scales.nbytes + self.values.nbytes)

def quantize_q8(matrix: np.ndarray) -> Q8Matrix:
    x=np.asarray(matrix,dtype=np.float32); scales=np.maximum(np.max(np.abs(x),axis=1)/127.0,1e-12).astype(np.float32)
    q=np.rint(x/scales[:,None]).clip(-127,127).astype(np.int8)
    return Q8Matrix(x.shape,scales.astype(np.float16),q)

def dequantize_q8(qm: Q8Matrix) -> np.ndarray:
    return qm.values.astype(np.float32)*qm.scales.astype(np.float32)[:,None]
