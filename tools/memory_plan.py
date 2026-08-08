#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from larc.latent_kv import cache_bytes,kivi_latent_cache_bytes,fp16_cache_bytes

CONFIG=dict(layers=30,hidden=576,intermediate=1536,qout=576,kvout=192,kv_heads=3,heads=9,head_dim=64,vocab=49152)
PROFILES={
 '10x': dict(hidden=32,o=32,down=48,vocab=128),
 '15x': dict(hidden=20,o=20,down=32,vocab=80),
 '20x': dict(hidden=12,o=12,down=20,vocab=48),
 '30x': dict(hidden=8,o=8,down=12,vocab=32),
}
GGUF_Q4=105_000_000

def q4(rows,cols): return rows*((cols+1)//2)+2*rows

def weight_bytes(p):
 c=CONFIG; L=c['layers']; h=c['hidden']; inter=c['intermediate']; vocab=c['vocab']; kv=c['kvout']
 total=q4(vocab,p['vocab'])+q4(h,p['vocab'])+q4(p['vocab'],h)
 per=q4(p['hidden'],h)+q4(h,p['hidden'])+2*q4(kv,p['hidden'])+2*q4(inter,p['hidden'])
 per+=q4(p['o'],h)+q4(h,p['o'])
 per+=q4(p['down'],inter)+q4(h,p['down'])
 total+=L*per
 total+=(L*2*h+h)*4
 return total

def workspace_bound(p,batch_tokens=1,tile_rows=256):
 c=CONFIG; max_in=max(c['hidden'],c['intermediate'],p['vocab']); max_rank=max(p['hidden'],p['o'],p['down'],p['vocab'])
 # Input row, bounded packed-Q4 decode/compute tile, output/logit staging and rank scratch.
 return batch_tokens*max_in + tile_rows*max(max_rank,c['hidden']) + 4*max(c['intermediate'],c['vocab']) + 4*max_rank

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path); ap.add_argument('--kv-codec',choices=['row','kivi'],default='kivi'); args=ap.parse_args(); rows=[]
 for name,p in PROFILES.items():
  wb=weight_bytes(p)
  for kr in [12,16,24,32]:
   for seq in [2048,8192]:
    kw=dict(layers=30,seq=seq,kv_heads=3,head_dim=64,rank=kr)
    basekv=fp16_cache_bytes(layers=30,seq=seq,kv_heads=3,head_dim=64)
    kv=kivi_latent_cache_bytes(**kw) if args.kv_codec=='kivi' else cache_bytes(**kw)
    ws=workspace_bound(p); base=GGUF_Q4+basekv+ws; larc=wb+kv+ws
    rows.append({'profile':name,'kv_codec':args.kv_codec,'kv_rank':kr,'context':seq,'larc_weight_bytes':wb,'weight_reduction_vs_q4':GGUF_Q4/wb,'baseline_fp16_kv_bytes':basekv,'larc_latent_q2_kv_bytes':kv,'kv_reduction':basekv/kv,'workspace_bound_bytes':ws,'baseline_modeled_peak_bytes':base,'larc_modeled_peak_bytes':larc,'modeled_total_memory_reduction':base/larc})
 text=json.dumps(rows,indent=2); print(text)
 if args.out:
  args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(text+'\n')
if __name__=='__main__': main()
