#!/usr/bin/env python3
"""Precommitted Run-8 gate for real-model additive vector quantization."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('benchmarks/RUN8_GATE.json'));a=ap.parse_args()
    d=json.load(open(a.result)); rows=d['stage_results']
    strong=[r for r in rows if r['whole_model_weight_reduction_x']>=2.0 and r['ppl_ratio_vs_fp32']<=1.50]
    useful=[r for r in rows if r['whole_model_weight_reduction_x']>=1.70 and r['ppl_ratio_vs_fp32']<=2.00]
    borderline=[r for r in rows if r['whole_model_weight_reduction_x']>=1.50 and r['ppl_ratio_vs_fp32']<=2.50]
    if strong:
        best=max(strong,key=lambda r:r['whole_model_weight_reduction_x']);status='pass_strong_additive_vq';next_step='freeze exact VQ representation, tune codebooks/indices with activation-aware loss, extend evaluation, and implement direct packed lookup-GEMV kernel'
    elif useful:
        best=max(useful,key=lambda r:r['whole_model_weight_reduction_x']);status='promising_additive_vq';next_step='optimize codebooks and discrete assignments under fixed byte budgets before expanding to embeddings and native runtime'
    elif borderline:
        best=max(borderline,key=lambda r:r['whole_model_weight_reduction_x']);status='borderline_additive_vq';next_step='apply fixed-index codebook fine-tuning and outlier/residual escape coding; retain additive VQ only if the same byte budget improves materially'
    else:
        best=min(rows,key=lambda r:r['ppl_ratio_vs_fp32']);status='fail_naive_additive_vq';next_step='do not claim sub-bit/low-bit VQ transfer; test second-order/outlier-aware VQ or structured sparsity with a competitive Q4 baseline'
    out={
      'run':8,'status':status,'best_under_gate':best,
      'precommitted_thresholds':{
        'strong':'whole-model modeled weight reduction >=2.00x and PPL <=1.50x FP32',
        'useful':'whole-model modeled weight reduction >=1.70x and PPL <=2.00x FP32',
        'borderline':'whole-model modeled weight reduction >=1.50x and PPL <=2.50x FP32'
      },
      'note':'Thresholds were committed before the first Run-8 SmolLM2 result. Row-Q4 is reported but is too weak on this short slice to define success alone. Modeled bytes are not RSS/VRAM.',
      'next_step':next_step
    }
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
