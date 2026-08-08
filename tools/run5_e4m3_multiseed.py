#!/usr/bin/env python3
"""Run-5 five-seed bridge: function-prefit/group64-QAT weights + E4M3 latent Q2.

Uses the same latent coefficient/metadata mathematics as the native packed
Run-4 attention path, while retaining the Run-5 conversion method.
"""
import json,math,statistics
from pathlib import Path
import torch
import tools.run5_fullstack_protocol as base

# Disjoint deterministic latent-basis calibration stream.
base.cal=torch.tensor([base.stoi[c] for c in base.corpus(200,555)])

def fit_basis_det(x):
    x=x.float();cov=x.t()@x;_,v=torch.linalg.eigh(cov)
    return v[:,-base.RANK:].t().flip(0).contiguous()

def e4m3_vector_dq(a):
    # Per-vector metadata, independent along token dimension.
    mn=a.amin(dim=-1,keepdim=True);mx=a.amax(dim=-1,keepdim=True)
    sc=((mx-mn)/3).clamp_min(2.0**-9)
    q=torch.round((a-mn)/sc).clamp(0,3)
    mnd=mn.to(torch.float8_e4m3fn).float()
    scd=sc.to(torch.float8_e4m3fn).float().clamp_min(2.0**-9)
    return q*scd+mnd

base.fit_basis=fit_basis_det
base.groupdq=e4m3_vector_dq

def main():
    rows=[]
    for seed in [3,7,11,19,23]:
        t=base.train_teacher(seed)
        baseline=base.quantized_copy(t,row=True)
        m=base.convert_prefit_qat(t,seed)
        bn,n=base.evaluate(baseline)
        fn,_=base.evaluate_larc(m)
        fp,_=base.evaluate(t)
        rows.append({'seed':seed,'baseline_row_q4_nll':bn,'larc_e4m3_full_nll':fn,'delta_nats_per_char':fn-bn,'perplexity_ratio':math.exp(fn-bn),'fp32_teacher_nll':fp,'perplexity_ratio_vs_fp32':math.exp(fn-fp)})
        print('seed',seed,rows[-1],flush=True)
    d=[r['delta_nats_per_char'] for r in rows];p=[r['perplexity_ratio'] for r in rows];pf=[r['perplexity_ratio_vs_fp32'] for r in rows]
    out={'evidence_level':'L2C multi-seed quality using native-packed-equivalent KV codec','quality_context':64,'training_seeds':[3,7,11,19,23],'evaluation_chars_per_seed':100032,'baseline':'project canonical row-Q4 teacher + normal/full KV','larc_weights':'teacher-layer function prefit + hard-projected group-64 QAT','larc_kv':'rank16 latent Q2 with E4M3-FN min/scale per vector, deterministic Q4 K/V bases, both ridge-stabilized inverse-Gram metrics','seeds':rows,'statistics':{'mean_delta_nats_per_char_vs_row_q4':statistics.mean(d),'sample_std_delta_nats_per_char_vs_row_q4':statistics.stdev(d),'mean_perplexity_ratio_vs_row_q4':statistics.mean(p),'sample_std_perplexity_ratio_vs_row_q4':statistics.stdev(p),'mean_perplexity_ratio_vs_fp32_teacher':statistics.mean(pf)},'execution_boundary':'KV mathematics matches native packed Run-4 attention path; group-64 weights are still reference-dequantized during quality evaluation.','memory_artifact':'benchmarks/run5_packed_context_sweep.json'}
    Path('benchmarks/run5_e4m3_multiseed.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
