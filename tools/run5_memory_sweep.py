#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from larc.grouped_kv import grouped_latent_cache_bytes,reference_workspace_bytes
from larc.latent_kv import fp16_cache_bytes

LAYERS=16
HIDDEN=128
HEADS=4
HEAD_DIM=32
INTERMEDIATE=256
RANK=16
GROUP=3
BASELINE_ROW_Q4_WEIGHT_BYTES=1_129_482
LARC_GROUP64_Q4_WEIGHT_BYTES=79_828
CONTEXTS=[64,128,256,512,1024,2048,4096,8192]

def row(context:int):
    base_kv=fp16_cache_bytes(layers=LAYERS,seq=context,kv_heads=HEADS,head_dim=HEAD_DIM)
    larc_kv=grouped_latent_cache_bytes(layers=LAYERS,seq=context,kv_heads=HEADS,head_dim=HEAD_DIM,rank=RANK,group_tokens=GROUP,residual_tail_fp16=True)
    scratch=reference_workspace_bytes(context=context,hidden=HIDDEN,heads=HEADS,rank=RANK,intermediate=INTERMEDIATE)
    base=BASELINE_ROW_Q4_WEIGHT_BYTES+base_kv+scratch
    larc=LARC_GROUP64_Q4_WEIGHT_BYTES+larc_kv+scratch
    return {
      'context':context,
      'baseline_row_q4_weight_bytes':BASELINE_ROW_Q4_WEIGHT_BYTES,
      'larc_group64_q4_weight_bytes':LARC_GROUP64_Q4_WEIGHT_BYTES,
      'baseline_fp16_kv_bytes':base_kv,
      'larc_group3_q2_kv_bytes':larc_kv,
      'scratch_bytes_each':scratch,
      'baseline_modeled_total_bytes':base,
      'larc_modeled_total_bytes':larc,
      'modeled_total_reduction_x':base/larc,
      'kv_reduction_x':base_kv/larc_kv,
    }

def main():
    rows=[row(c) for c in CONTEXTS]
    text=json.dumps(rows,indent=2)
    print(text)
    Path('benchmarks/run5_memory_context.json').write_text(text+'\n')
if __name__=='__main__':main()
