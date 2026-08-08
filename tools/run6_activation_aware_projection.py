#!/usr/bin/env python3
"""Run 6 activation-aware reduced-rank operator test on a real pretrained model.

This is deliberately stronger than projecting inputs onto their PCA subspace.
For each real linear operator W and calibration activation matrix X:
  Y = X W^T
  1. choose a rank-r output basis from SVD(Y),
  2. ridge-fit a rank-r latent map X -> Z where Z are output-basis coords,
  3. reconstruct held-out outputs as (X_eval B) A.

The resulting A(Bx) is a direct activation-aware reduced-rank regression baseline.
It is still training-free and is evaluated only on held-out activations.
"""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
import torch
from run6_real_model_falsification import TEXT,collect_inputs,find_projection,get_layers,parse_csv_ints,DEFAULT_LAYERS,DEFAULT_RANKS


def fit_eval(xc,xe,w,ranks,ridge_rel=1e-4):
    w=w.detach().float().cpu();yc=xc@w.t();ye=xe@w.t();ypow=float((ye*ye).sum().clamp_min(1e-30))
    # Output basis on calibration outputs.
    _,s,vh=torch.linalg.svd(yc,full_matrices=False);tot=float((s*s).sum().clamp_min(1e-30))
    # Dual ridge inverse is N_cal x N_cal, avoiding a d_in x d_in inverse.
    gram=xc@xc.t();lam=float(torch.diagonal(gram).mean().clamp_min(1e-12))*ridge_rel
    solve=torch.linalg.inv(gram+torch.eye(gram.shape[0])*lam)
    rows=[]
    for req in ranks:
        r=min(req,vh.shape[0]);A=vh[:r]                 # [r,out]
        z=yc@A.t()                                      # [N,r]
        B=xc.t()@(solve@z)                              # [in,r]
        pred=(xe@B)@A
        nmse=float(((pred-ye)**2).sum()/ypow)
        energy=float((s[:r]*s[:r]).sum()/tot)
        rows.append({'requested_rank':req,'effective_rank':r,'input_dim':xc.shape[1],'output_dim':w.shape[0],'rank_fraction_input':r/xc.shape[1],'calibration_output_energy_fraction':energy,'heldout_operator_output_nmse':nmse})
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='HuggingFaceTB/SmolLM2-135M');ap.add_argument('--out',type=Path,default=Path('benchmarks/run6_activation_aware_projection.json'));ap.add_argument('--layers',default=','.join(map(str,DEFAULT_LAYERS)));ap.add_argument('--ranks',default=','.join(map(str,DEFAULT_RANKS)));ap.add_argument('--cal-tokens',type=int,default=256);ap.add_argument('--eval-tokens',type=int,default=256);a=ap.parse_args()
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.manual_seed(0);tok=AutoTokenizer.from_pretrained(a.model);m=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float32).eval().cpu();layers=get_layers(m)
    selected=tuple(i for i in parse_csv_ints(a.layers) if i<len(layers));ranks=parse_csv_ints(a.ranks);ids=tok(TEXT,return_tensors='pt',add_special_tokens=False).input_ids;need=a.cal_tokens+a.eval_tokens
    if ids.shape[1]<need:raise RuntimeError(f'need {need} tokens; got {ids.shape[1]}')
    cal=ids[:,:a.cal_tokens];ev=ids[:,a.cal_tokens:need];ci=collect_inputs(m,cal,selected);ei=collect_inputs(m,ev,selected)
    sites={}
    for key,xc in ci.items():
        _,li,site=key.split('.');mod=find_projection(layers[int(li)],site);sites[key]=fit_eval(xc,ei[key],mod.weight,ranks)
    summary={}
    for rank in (32,64):
        vals=[]
        for rows in sites.values():
            q=next((x for x in rows if x['requested_rank']==rank),None)
            if q:vals.append(q['heldout_operator_output_nmse'])
        summary[str(rank)]={'sites':len(vals),'median_output_nmse':statistics.median(vals),'mean_output_nmse':statistics.mean(vals),'fraction_below_0.01':sum(v<.01 for v in vals)/len(vals),'fraction_below_0.03':sum(v<.03 for v in vals)/len(vals),'fraction_below_0.05':sum(v<.05 for v in vals)/len(vals)}
    out={'run':6,'evidence_level':'L3-precheck activation-aware reduced-rank real operator test','model':a.model,'model_commit':getattr(m.config,'_commit_hash',None),'protocol':{'selected_layers':selected,'ranks':ranks,'calibration_tokens':a.cal_tokens,'evaluation_tokens':a.eval_tokens,'fit':'output-SVD basis + dual ridge reduced-rank regression','ridge_relative_to_mean_dual_gram_diagonal':1e-4},'projection_summary':summary,'projection_sites':sites,'claim_boundary':'Held-out real-activation operator test only; no model replacement or end-to-end quality claim.'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
