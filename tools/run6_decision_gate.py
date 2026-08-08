#!/usr/bin/env python3
"""Classify Run-6 real-model evidence without moving the goalposts post hoc."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path


def classify_projection(d):
    s32=d.get('projection_summary',{}).get('32',{})
    s64=d.get('projection_summary',{}).get('64',{})
    if not s32 or not s64:return {'status':'insufficient'}
    # Rank-32 is the aggressive target; rank-64 is the fallback. These gates use
    # held-out *operator-output* NMSE, not calibration spectral energy.
    if s32['median_output_nmse'] <= .03 and s32['fraction_below_0.05'] >= .70:
        status='pass_rank32'
    elif s64['median_output_nmse'] <= .03 and s64['fraction_below_0.05'] >= .70:
        status='pass_rank64_only'
    else:
        status='fail_low_rank_projection'
    return {'status':status,'rank32':s32,'rank64':s64}


def classify_raw_sharing(d):
    vals=[x['perplexity_ratio'] for x in d.get('single_layer_alias',[])]
    groups=[x['perplexity_ratio'] for x in d.get('contiguous_group_alias',[])]
    if not vals:return {'status':'insufficient'}
    med=statistics.median(vals)
    if med <= 1.10:status='surprisingly_interchangeable'
    elif med <= 1.50:status='recoverable_candidate'
    else:status='raw_aliasing_severe'
    return {'status':status,'single_layer_median_perplexity_ratio':med,'group_perplexity_ratios':groups}


def classify_partial(d):
    if not d:return {'status':'not_run'}
    q=d['quality'];w=d['weight_accounting']
    ppl=q['post_perplexity_ratio_vs_row_q4'];gr=w['group_weight_reduction_x']
    # A 4->1 real group should retain most of its theoretical weight reduction and
    # stay inside ~10% PPL before scaling the method to more groups.
    if ppl <= 1.10 and gr >= 3.5:status='pass_expand_real_conversion'
    elif ppl <= 1.25 and gr >= 3.0:status='borderline_tune_recovery'
    else:status='fail_current_sharing_recipe'
    return {'status':status,'post_perplexity_ratio_vs_row_q4':ppl,'group_weight_reduction_x':gr,'whole_model_weight_reduction_x':w['whole_model_weight_reduction_x']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--falsification',type=Path,required=True);ap.add_argument('--partial',type=Path);ap.add_argument('--out',type=Path,default=Path('benchmarks/RUN6_GATE.json'));a=ap.parse_args()
    f=json.load(open(a.falsification));p=json.load(open(a.partial)) if a.partial and a.partial.exists() else None
    projection=classify_projection(f);sharing=classify_raw_sharing(f);partial=classify_partial(p)
    if projection['status']=='fail_low_rank_projection':
        next_step='stop treating low-rank activation projection as a universal core; segment by operator/layer and investigate structured/sparse alternatives before full conversion'
    elif partial['status']=='pass_expand_real_conversion':
        next_step='expand recovered sharing to multiple real layer groups, then integrate packed group64 weights plus packed Q2/E4M3 KV and measure RSS'
    elif partial['status']=='borderline_tune_recovery':
        next_step='tune recovery/adapters on the same fixed real-model group before expanding sharing scope'
    else:
        next_step='keep projection results but reduce sharing aggressiveness or add depth-specific correction capacity; do not extrapolate the toy 16->1 result'
    out={'run':6,'projection_gate':projection,'raw_sharing_gate':sharing,'partial_conversion_gate':partial,'next_step':next_step,'gate_note':'Thresholds were committed before the real checkpoint result was observed.'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
