#!/usr/bin/env python3
"""Run 10: external tuned W2A16G64 quality reference on SmolLM2-135M.

This is not a LARC codec. It establishes what a maintained extreme-low-bit
optimizer can do on the same real pretrained model before we invent another
representation. AutoRound is run with genuine tuning (not RTN/model-free),
W2A16 group64, algorithm extensions enabled, and a reduced CPU-friendly
calibration budget.
"""
from __future__ import annotations
import argparse,json,math,os,time
from pathlib import Path
import torch

from run6_real_model_falsification import TEXT,nll


def dir_bytes(p:Path)->int:
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='HuggingFaceTB/SmolLM2-135M')
    ap.add_argument('--iters',type=int,default=100)
    ap.add_argument('--nsamples',type=int,default=64)
    ap.add_argument('--seqlen',type=int,default=512)
    ap.add_argument('--eval-tokens',type=int,default=512)
    ap.add_argument('--out-dir',type=Path,default=Path('/tmp/smollm2-w2g64-autoround'))
    ap.add_argument('--out',type=Path,default=Path('benchmarks/run10_autoround_w2.json'))
    a=ap.parse_args()

    import auto_round
    from auto_round import AutoRound
    from transformers import AutoModelForCausalLM,AutoTokenizer

    torch.manual_seed(10);torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    tok=AutoTokenizer.from_pretrained(a.model)
    base=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.float32).eval().cpu()
    ids=tok(TEXT*8,return_tensors='pt',add_special_tokens=False).input_ids
    if ids.shape[1] < a.eval_tokens+1: raise RuntimeError('insufficient evaluation text')
    ev=ids[:,-(a.eval_tokens+1):]
    fp32_nll=nll(base,ev)
    del base

    t0=time.time()
    ar=AutoRound(
        model=a.model,
        scheme='W2A16G64',
        iters=a.iters,
        nsamples=a.nsamples,
        seqlen=a.seqlen,
        batch_size=1,
        device_map='cpu',
        enable_alg_ext=True,
        enable_deterministic_algorithms=True,
    )
    qmodel,layer_config=ar.quantize()
    tune_seconds=time.time()-t0
    qmodel.eval()
    w2_nll=nll(qmodel,ev)

    a.out_dir.mkdir(parents=True,exist_ok=True)
    save_error=None
    try:
        ar.save_quantized(format='auto_round',output_dir=str(a.out_dir))
    except Exception as e:
        save_error=repr(e)
    packed_bytes=dir_bytes(a.out_dir) if any(a.out_dir.iterdir()) else None

    def config_summary(cfg):
        vals=[]
        for k,v in cfg.items():
            if isinstance(v,dict):
                vals.append((k,{x:v.get(x) for x in ('bits','group_size','sym','data_type') if x in v}))
        return vals[:20]

    out={
      'run':10,
      'evidence_level':'external-maintained tuned W2 real-model reference',
      'model':a.model,
      'model_commit':getattr(qmodel.config,'_commit_hash',None),
      'autoround_version':getattr(auto_round,'__version__',None),
      'protocol':{
        'scheme':'W2A16G64','bits':2,'group_size':64,'activation_bits':16,'iters':a.iters,'nsamples':a.nsamples,'seqlen':a.seqlen,'batch_size':1,
        'device_map':'cpu','enable_alg_ext':True,'deterministic_algorithms':True,'calibration_dataset':'AutoRound default NeelNanda/pile-10k','evaluation_tokens':a.eval_tokens,
        'note':'Reduced-cost tuned AutoRound reference, not the documented 1000-iteration/512-sample AutoRoundBest recipe.'
      },
      'quality':{
        'fp32_nll':fp32_nll,'w2_nll':w2_nll,'delta_nats_per_token':w2_nll-fp32_nll,'ppl_ratio_vs_fp32':math.exp(w2_nll-fp32_nll)
      },
      'tuning_wall_seconds':tune_seconds,
      'saved_auto_round_directory_bytes':packed_bytes,
      'save_error':save_error,
      'layer_config_count':len(layer_config),
      'layer_config_sample':config_summary(layer_config),
      'claim_boundary':'AutoRound quality reference on the project custom held-out text. Not LARC, not standard WikiText perplexity, not measured runtime RSS, and not the full AutoRoundBest W2 recipe.'
    }
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
