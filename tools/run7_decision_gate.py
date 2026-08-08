#!/usr/bin/env python3
"""Precommitted Run-7 decision gate.

These thresholds are intentionally fixed before the first Run-7 SmolLM2 result.
The gate distinguishes a structural shared-basis failure from a factor-quantizer
failure and from a useful but still insufficient compression mechanism.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def classify_candidate(name, d):
    return {
        'name': name,
        'selected_group_count': d['selected_group_count'],
        'ppl_ratio_vs_fp32': d['ppl_ratio_vs_fp32'],
        'ppl_ratio_vs_row_q4': d['ppl_ratio_vs_row_q4'],
        'whole_model_weight_reduction_x': d['whole_model_weight_reduction_x'],
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--result',type=Path,required=True); ap.add_argument('--out',type=Path,default=Path('benchmarks/RUN7_GATE.json')); a=ap.parse_args()
    d=json.load(open(a.result)); c={k:classify_candidate(k,v) for k,v in d['end_to_end'].items()}
    candidates=list(c.values())
    strong=[x for x in candidates if x['selected_group_count']>0 and x['ppl_ratio_vs_row_q4']<=1.10 and x['whole_model_weight_reduction_x']>=1.50]
    useful=[x for x in candidates if x['selected_group_count']>0 and x['ppl_ratio_vs_row_q4']<=1.15 and x['whole_model_weight_reduction_x']>=1.25]
    structural=[x for x in candidates if x['selected_group_count']>0 and x['ppl_ratio_vs_fp32']<=1.10]
    quant_bad=[x for x in candidates if x['selected_group_count']>0 and x['ppl_ratio_vs_fp32']<=1.10 and x['ppl_ratio_vs_row_q4']>1.25]
    if strong:
        best=max(strong,key=lambda x:x['whole_model_weight_reduction_x']); status='pass_expand_shared_basis'; next_step='expand adaptive shared-basis grouping/rank search, integrate latent KV, then compare against Q4_K_M with measured RSS'
    elif useful:
        best=max(useful,key=lambda x:x['whole_model_weight_reduction_x']); status='promising_but_insufficient'; next_step='improve group/rank allocation and factor quantization before expanding; preserve operator segmentation'
    elif structural and quant_bad:
        best=min(quant_bad,key=lambda x:x['ppl_ratio_vs_fp32']); status='structural_pass_quantization_fail'; next_step='retain shared-basis architecture but replace/tune factor quantization; do not reject basis sharing from Q4 factor damage'
    elif structural:
        best=min(structural,key=lambda x:x['ppl_ratio_vs_row_q4']); status='structural_only'; next_step='shared-basis structure survives but current byte-quality tradeoff is insufficient; optimize grouping/ranks/quantizer'
    else:
        populated=[x for x in candidates if x['selected_group_count']>0]
        best=min(populated,key=lambda x:x['ppl_ratio_vs_fp32']) if populated else None; status='fail_current_shared_basis_recipe'; next_step='do not force cross-layer basis sharing broadly; move to per-operator structured pruning/sparsity or learned residual dictionaries'
    out={
        'run':7,
        'status':status,
        'candidates':c,
        'best_under_gate':best,
        'precommitted_thresholds':{
            'strong':'Q4-factor PPL <=1.10x row-Q4 and whole-model modeled weight reduction >=1.50x',
            'useful':'Q4-factor PPL <=1.15x row-Q4 and whole-model modeled weight reduction >=1.25x',
            'structural_survival':'FP32-factor PPL <=1.10x FP32 with at least one selected group',
            'quantization_failure':'structural survival plus Q4-factor PPL >1.25x row-Q4',
        },
        'next_step':next_step,
        'gate_note':'Thresholds committed before the first Run-7 SmolLM2 result. Modeled bytes are not RSS/VRAM.'
    }
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
