#!/usr/bin/env python3
"""Exact-tail wrapper for the canonical Run-5 full-stack protocol.

The base protocol retains the current incomplete latent token group long enough
to re-quantize it when new tokens arrive. The memory model charges that tail as
FP16, so this wrapper forces the entire current group through FP16 before its
min/max, code assignment, and FP16 metadata are derived. This closes the last
quality-vs-byte pairing mismatch without duplicating the training protocol.
"""
import torch
import tools.run5_fullstack_protocol as base

def groupdq_fp16tail(a: torch.Tensor) -> torch.Tensor:
    a=a.half().float()
    mn=a.amin(dim=(-2,-1),keepdim=True)
    mx=a.amax(dim=(-2,-1),keepdim=True)
    sc=((mx-mn)/3).clamp_min(1e-8)
    mnh=mn.half().float();sch=sc.half().float()
    q=torch.round((a-mn)/sc).clamp(0,3)
    return q*sch+mnh

base.groupdq=groupdq_fp16tail

if __name__=='__main__':
    base.main()
