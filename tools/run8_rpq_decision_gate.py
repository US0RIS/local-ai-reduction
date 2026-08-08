#!/usr/bin/env python3
"""Precommitted gate for the Run-8B residual product-quantization refinement."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('benchmarks/RUN8_RPQ_GATE.json'));a=ap.parse_args()
    d=json.load(open(a.result));rows=d['stage_results']
    strong=[r for r in rows if r['whole_model_weight_reduction_x']>=2.0 and r['ppl_ratio_vs_fp32']<=1.50]
    useful=[r for r in rows if r['whole_model_weight_reduction_x']>=1.70 and r['ppl_ratio_vs_fp32']<=2.00]
    borderline=[r for r in rows if r['whole_model_weight_reduction_x']>=1.50 and r['ppl_ratio_vs_fp32']<=2.50]
    if strong:status='pass_strong_rpq';best=max(strong,key=lambda r:r['whole_model_weight_reduction_x']);nxt='optimize fixed indices/codebooks at the passing rate, then expand evaluation and implement direct packed RPQ GEMV'
    elif useful:status='promising_rpq';best=max(useful,key=lambda r:r['whole_model_weight_reduction_x']);nxt='apply fixed-index codebook tuning and outlier escape coding at the same byte budget'
    elif borderline:status='borderline_rpq';best=max(borderline,key=lambda r:r['whole_model_weight_reduction_x']);nxt='test second-order block whitening and selective outlier residuals before retaining RPQ'
    else:status='fail_naive_rpq';best=min(rows,key=lambda r:r['ppl_ratio_vs_fp32']);nxt='move to second-order/outlier-aware VQ or structured sparse residuals; do not spend more work on naive Euclidean product codebooks'
    out={'run':'8B','status':status,'best_under_gate':best,'precommitted_thresholds':{'strong':'>=2.00x whole-model reduction and <=1.50x PPL vs FP32','useful':'>=1.70x and <=2.00x','borderline':'>=1.50x and <=2.50x'},'note':'Thresholds committed before the first RPQ result. Modeled bytes are not RSS/VRAM.','next_step':nxt}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
