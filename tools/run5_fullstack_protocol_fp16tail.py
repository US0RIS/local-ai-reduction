#!/usr/bin/env python3
"""Exact-tail wrapper for the canonical Run-5 reference protocol."""
import torch
import tools.run5_fullstack_protocol as base

def groupdq_fp16tail(a: torch.Tensor) -> torch.Tensor:
    a=a.half().float()
    mn=a.amin(dim=(-2,-1),keepdim=True);mx=a.amax(dim=(-2,-1),keepdim=True)
    sc=((mx-mn)/3).clamp_min(1e-8);mnh=mn.half().float();sch=sc.half().float()
    q=torch.round((a-mn)/sc).clamp(0,3)
    return q*sch+mnh
base.groupdq=groupdq_fp16tail
if __name__=='__main__':base.main()
