#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path


def load(p: Path): return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    r=a.raw
    fp=load(r/'ppl_fp.json'); rtn=load(r/'ppl_rtn.json'); tuned=load(r/'ppl_tuned.json')
    mr=load(r/'meta_rtn.json'); mt=load(r/'meta_tuned.json')
    out={
      'run':10,
      'evidence_level':'external pretrained W2 post-training quantization reference',
      'model':mr['model'],
      'source_model_commit':mr['source_model_commit'],
      'scheme':'W2A16G64',
      'evaluation':{
        'method':fp['method'],
        'context':fp['context'],
        'predicted_tokens':fp['predicted_tokens'],
        'fp_reference':fp,
        'pure_rtn':rtn,
        'tuned_autoround':tuned,
      },
      'quality_ratios':{
        'rtn_ppl_ratio_vs_fp':rtn['perplexity']/fp['perplexity'],
        'tuned_ppl_ratio_vs_fp':tuned['perplexity']/fp['perplexity'],
        'tuned_ppl_ratio_vs_rtn':tuned['perplexity']/rtn['perplexity'],
        'rtn_delta_nll_vs_fp':rtn['mean_nll']-fp['mean_nll'],
        'tuned_delta_nll_vs_fp':tuned['mean_nll']-fp['mean_nll'],
      },
      'serialization':{
        'pure_rtn_directory_bytes':mr['serialized_directory_bytes'],
        'tuned_directory_bytes':mt['serialized_directory_bytes'],
      },
      'quantization':{'pure_rtn':mr,'tuned_autoround':mt},
      'claim_boundary':'AutoRound W2A16G64 quality/serialization reference under one HF evaluator. Absolute PPL is not directly substituted for Run-9 llama.cpp PPL; compare within-runtime PPL ratios. This is not a LARC result or measured VRAM claim.'
    }
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'quality_ratios':out['quality_ratios'],'serialization':out['serialization']},indent=2))
if __name__=='__main__': main()
