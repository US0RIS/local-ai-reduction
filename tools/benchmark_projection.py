#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from larc.projection import fit_projection_bundle,run_bundle
from larc.q4 import quantize_q4

def covariance(n,core_rank,energy,seed):
    rng=np.random.default_rng(seed); q,_=np.linalg.qr(rng.standard_normal((n,n),dtype=np.float32)); eig=np.empty(n,dtype=np.float32); eig[:core_rank]=energy/core_rank; eig[core_rank:]=(1-energy)/max(n-core_rank,1); return q.astype(np.float32),eig

def sample(q,eig,count,seed):
    rng=np.random.default_rng(seed); z=rng.standard_normal((q.shape[0],count),dtype=np.float32); return (q@(np.sqrt(eig)[:,None]*z)).astype(np.float32)
def nmse(ys,yh): return sum(float(np.sum((a-b)**2)) for a,b in zip(ys,yh))/max(sum(float(np.sum(a**2)) for a in ys),1e-30)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=384); ap.add_argument('--operators',type=int,default=5); ap.add_argument('--samples',type=int,default=1024); ap.add_argument('--out',type=Path); args=ap.parse_args(); rng=np.random.default_rng(123); ws=[rng.standard_normal((args.n,args.n),dtype=np.float32)/np.sqrt(args.n) for _ in range(args.operators)]; baseline=sum(quantize_q4(w).storage_bytes for w in ws); cases=[]
    for precision in ['q4','q8']:
      for frac in [.025,.05,.10]:
        rank=max(1,round(args.n*frac))
        for energy in [.90,.95,.98]:
          q,e=covariance(args.n,rank,energy,7); cal=sample(q,e,args.samples,11); test=sample(q,e,args.samples//2,13); b=fit_projection_bundle(ws,cal,rank,precision); ys=[w@test for w in ws]; yh=run_bundle(b,test); cases.append({'precision':precision,'n':args.n,'operators':args.operators,'retained_rank':rank,'rank_fraction':rank/args.n,'activation_energy_in_core':energy,'bundle_bytes':b.storage_bytes,'row_q4_baseline_bytes':baseline,'compression_multiple_vs_row_q4':baseline/b.storage_bytes,'heldout_output_nmse':nmse(ys,yh)})
    text=json.dumps(cases,indent=2); print(text); args.out and args.out.write_text(text+'\n')
if __name__=='__main__': main()
