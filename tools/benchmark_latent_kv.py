#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
import torch
from larc.latent_kv import LatentQ2KV,fit_basis,cache_bytes,fp16_cache_bytes

def nmse(a,b): return float(torch.mean((a-b)**2)/torch.mean(a**2).clamp_min(1e-30))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path); args=ap.parse_args(); torch.manual_seed(4)
 d=64; ncal=4096; ntest=2048
 latent=torch.randn(ncal,16); mixk=torch.randn(16,d)/4; mixv=torch.randn(16,d)/4
 kc=latent@mixk+0.10*torch.randn(ncal,d); vc=latent@mixv+0.10*torch.randn(ncal,d)
 lt=torch.randn(ntest,16); k=lt@mixk+0.10*torch.randn(ntest,d); v=lt@mixv+0.10*torch.randn(ntest,d); q=torch.randn(d)
 out=[]
 for r in [8,12,16,24,32,48]:
  kb=fit_basis(kc,r); vb=fit_basis(vc,r); codec=LatentQ2KV(kb,vb); ek,ev,sh=codec.encode(k,v); kh,vh=codec.decode(ek,ev,sh)
  ref=torch.softmax((q@k.T)/math.sqrt(d),-1)@v; got=codec.attention(q,ek,ev,sh)
  mem2k=cache_bytes(layers=30,seq=2048,kv_heads=3,head_dim=64,rank=r); base2k=fp16_cache_bytes(layers=30,seq=2048,kv_heads=3,head_dim=64)
  mem8k=cache_bytes(layers=30,seq=8192,kv_heads=3,head_dim=64,rank=r); base8k=fp16_cache_bytes(layers=30,seq=8192,kv_heads=3,head_dim=64)
  out.append({'rank':r,'rank_fraction':r/d,'k_nmse':nmse(k,kh),'v_nmse':nmse(v,vh),'attention_output_nmse':nmse(ref,got),'encoded_test_bytes':ek.storage_bytes+ev.storage_bytes,'kv_2048_bytes':mem2k,'kv_2048_reduction':base2k/mem2k,'kv_8192_bytes':mem8k,'kv_8192_reduction':base8k/mem8k})
 text=json.dumps(out,indent=2); print(text)
 if args.out:
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(text+'\n')
if __name__=='__main__': main()
