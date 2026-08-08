#!/usr/bin/env python3
"""Precommitted gate for tied embedding/LM-head factorization."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('benchmarks/RUN12_GATE.json'));a=ap.parse_args()
    d=json.load(open(a.result));rows=d['ranks']
    strong=[r for r in rows if r['embedding_element_reduction_x']>=5.5 and r['ppl_ratio_vs_fp32']<=1.10]
    useful=[r for r in rows if r['embedding_element_reduction_x']>=4.0 and r['ppl_ratio_vs_fp32']<=1.25]
    borderline=[r for r in rows if r['embedding_element_reduction_x']>=2.8 and r['ppl_ratio_vs_fp32']<=1.50]
    if strong:status='pass_strong_embedding_factorization';best=max(strong,key=lambda r:r['embedding_element_reduction_x']);nxt='combine factorized tied embedding with the best surviving transformer-weight structure, then optimize factor quantization'
    elif useful:status='promising_embedding_factorization';best=max(useful,key=lambda r:r['embedding_element_reduction_x']);nxt='distill/fine-tune the tied factors at this rank and quantize them with a strong optimizer'
    elif borderline:status='borderline_embedding_factorization';best=max(borderline,key=lambda r:r['embedding_element_reduction_x']);nxt='test learned factorization/distillation; if quality does not improve, a smaller vocabulary/tokenizer is required for >10x on this model'
    else:status='fail_posthoc_embedding_factorization';best=min(rows,key=lambda r:r['ppl_ratio_vs_fp32']);nxt='treat the vocabulary/head floor as an architecture/tokenizer-training problem rather than post-hoc SVD'
    out={'run':12,'status':status,'best_under_gate':best,'precommitted_thresholds':{'strong':'embedding >=5.5x and PPL <=1.10x FP32','useful':'embedding >=4x and PPL <=1.25x','borderline':'embedding >=2.8x and PPL <=1.50x'},'note':'Thresholds committed before first result. Structural FP32 factors only, not packed bytes.','next_step':nxt}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
