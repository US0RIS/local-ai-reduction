#!/usr/bin/env python3
"""Precommitted gate for cross-depth full-matrix basis decomposition."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('benchmarks/RUN11_GATE.json'));a=ap.parse_args()
    d=json.load(open(a.result));rows=d['candidates']
    strong=[r for r in rows if r['projection_element_reduction_x']>=7 and r['whole_parameter_reduction_x']>=3 and r['ppl_ratio_vs_fp32']<=1.10]
    useful=[r for r in rows if r['projection_element_reduction_x']>=5 and r['whole_parameter_reduction_x']>=2.5 and r['ppl_ratio_vs_fp32']<=1.25]
    borderline=[r for r in rows if r['projection_element_reduction_x']>=3 and r['whole_parameter_reduction_x']>=2 and r['ppl_ratio_vs_fp32']<=1.50]
    if strong:status='pass_strong_depth_basis';best=max(strong,key=lambda r:r['whole_parameter_reduction_x']);nxt='quantize the learned full-matrix depth bases with a strong second-order W2/W4 optimizer and separately compress the tied embedding/head'
    elif useful:status='promising_depth_basis';best=max(useful,key=lambda r:r['whole_parameter_reduction_x']);nxt='fit activation-aware depth coefficients/bases and quantize bases with optimized rounding before expanding runtime work'
    elif borderline:status='borderline_depth_basis';best=max(borderline,key=lambda r:r['whole_parameter_reduction_x']);nxt='add a small layer-specific residual/adapter channel at fixed byte budget; retain only if quality crosses the useful gate'
    else:status='fail_depth_basis';best=min(rows,key=lambda r:r['ppl_ratio_vs_fp32']);nxt='treat depth sharing as a training/distillation problem rather than post-training tensor decomposition'
    out={'run':11,'status':status,'best_under_gate':best,'precommitted_thresholds':{'strong':'projection elements >=7x, whole parameters >=3x, PPL <=1.10x FP32','useful':'projection elements >=5x, whole parameters >=2.5x, PPL <=1.25x FP32','borderline':'projection elements >=3x, whole parameters >=2x, PPL <=1.50x FP32'},'note':'Thresholds committed before first SmolLM2 depth-basis result. This is structural FP32/BF16 parameter sharing, not a packed-memory claim.','next_step':nxt}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
