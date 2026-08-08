#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from larc.hrvq import HRVQConfig,decode,encode,storage_bytes,train_codebooks
from larc.metrics import linear_output_nmse,reconstruction_metrics

def make_matrix(kind,n,seed):
    rng=np.random.default_rng(seed)
    if kind=="gaussian": return rng.standard_normal((n,n),dtype=np.float32)/np.sqrt(n)
    if kind=="heavy_tail":
        x=rng.standard_t(df=4.0,size=(n,n)).astype(np.float32); return x/np.sqrt(float(np.mean(x*x))*n)
    if kind=="low_rank_plus_noise":
        rank=max(8,n//32); u=rng.standard_normal((n,rank),dtype=np.float32); v=rng.standard_normal((rank,n),dtype=np.float32); low=(u@v)/np.sqrt(rank*n); noise=rng.standard_normal((n,n),dtype=np.float32)/np.sqrt(n); w=.75*low+.25*noise; return w/np.sqrt(float(np.mean(w*w))*n)
    raise ValueError(kind)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=1024); ap.add_argument('--out',type=Path); args=ap.parse_args(); results=[]
    for kind in ['gaussian','heavy_tail','low_rank_plus_noise']:
      w=make_matrix(kind,args.n,7)
      for stages in [1,2,3]:
        cfg=HRVQConfig(stages=stages,max_train_vectors=32768,random_state=17); model=train_codebooks([w],cfg); enc=encode(w,model); dec=decode(enc,model); m=reconstruction_metrics(w,dec); total=storage_bytes(enc,model,True); bpw=total*8/w.size
        results.append({'distribution':kind,'stages':stages,'nominal_bpw_without_codebook':cfg.nominal_bits_per_weight,'effective_bpw_including_codebook':bpw,'q4_compression_multiple':4.5/bpw,'weight_nmse':m['nmse'],'weight_cosine':m['cosine'],'weight_snr_db':m['snr_db'],'linear_output_nmse':linear_output_nmse(w,dec,32,23),'payload_bytes':enc.payload_bytes,'codebook_bytes':model.codebook_bytes})
    text=json.dumps(results,indent=2); print(text); args.out and args.out.write_text(text+'\n')
if __name__=='__main__': main()
