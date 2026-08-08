"""Triton packed-INT4 kernels for LARC CUDA execution.

The kernel consumes packed nibbles directly; it never materializes a dense W.
This module is optional and imports without Triton/CUDA installed.
"""
from __future__ import annotations
import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

if triton is not None:
    @triton.jit
    def _q4_gemv_kernel(packed, scales, x, y, M: tl.constexpr, K: tl.constexpr,
                        STRIDE_BYTES: tl.constexpr, BLOCK_K: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_K)
        mask = offs < K
        byte_offs = row * STRIDE_BYTES + (offs >> 1)
        b = tl.load(packed + byte_offs, mask=mask, other=0).to(tl.int32)
        shift = (offs & 1) * 4
        code = (b >> shift) & 0xF
        q = code - 8
        xv = tl.load(x + offs, mask=mask, other=0.0).to(tl.float32)
        acc = tl.sum(q.to(tl.float32) * xv, axis=0)
        s = tl.load(scales + row).to(tl.float32)
        tl.store(y + row, acc * s)


def q4_gemv_cuda(packed: torch.Tensor, scales: torch.Tensor, cols: int,
                  x: torch.Tensor) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not packed.is_cuda or not scales.is_cuda or not x.is_cuda:
        raise ValueError("packed, scales, and x must be CUDA tensors")
    if x.ndim != 1 or x.numel() != cols:
        raise ValueError("v0.2 GPU reference kernel currently supports one vector")
    rows = packed.shape[0]; stride = packed.shape[1]; block = triton.next_power_of_2(cols)
    y = torch.empty((rows,), device=x.device, dtype=torch.float32)
    _q4_gemv_kernel[(rows,)](packed, scales, x, y, M=rows, K=cols,
                             STRIDE_BYTES=stride, BLOCK_K=block)
    return y


def q4_projected_gemv_cuda(b_packed, b_scales, b_cols,
                            a_packed, a_scales, a_cols, x):
    """Compute A(Bx) with packed B/A and only a rank-sized CUDA scratch vector."""
    z = q4_gemv_cuda(b_packed, b_scales, b_cols, x)
    if z.numel() != a_cols:
        raise ValueError("projected dimensions do not match")
    return q4_gemv_cuda(a_packed, a_scales, a_cols, z)
