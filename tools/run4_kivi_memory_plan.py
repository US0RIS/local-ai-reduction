#!/usr/bin/env python3
import json
from pathlib import Path
from memory_plan import CONFIG,PROFILES,GGUF_Q4,weight_bytes,workspace_bound
from larc.latent_kv import fp16_cache_bytes,kivi_latent_cache_bytes

def main():
 rows=[]
 for name,p in PROFILES.items():
  wb=weight_bytes(p);ws=workspace_bound(p)
  for seq in (2048,8192):
   basekv=fp16_cache_bytes(layers=CONFIG['layers'],seq=seq,kv_heads=CONFIG['kv_heads'],head_dim=CONFIG['head_dim'])
   kv=kivi_latent_cache_bytes(layers=CONFIG['layers'],seq=seq,kv_heads=CONFIG['kv_heads'],head_dim=CONFIG['head_dim'],rank=16)
   base=GGUF_Q4+basekv+ws;larc=wb+kv+ws
   rows.append({'profile':name,'context':seq,'larc_weight_bytes':wb,'baseline_fp16_kv_bytes':basekv,'larc_latent_q2_kv_bytes':kv,'kv_reduction_x':basekv/kv,'workspace_bound_bytes':ws,'baseline_modeled_peak_bytes':base,'larc_modeled_peak_bytes':larc,'modeled_total_memory_reduction_x':base/larc})
 Path('benchmarks/run4_kivi_memory_plan_rank16.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
