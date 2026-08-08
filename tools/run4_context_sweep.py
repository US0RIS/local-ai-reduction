#!/usr/bin/env python3
import json
from pathlib import Path

BASELINE_W=1129482
LARC_W=77322
SCRATCH=8704
LAYERS=16
HEADS=4
D=128
RANK=16

def row(seq):
    baseline_kv=LAYERS*seq*D*4  # FP16 K+V
    latent_payload=LAYERS*seq*HEADS*16  # K+V: 8 B q2 payload + 8 B row metadata
    basis_payload=HEADS*2*RANK*(D//HEADS)//2
    basis_scales=HEADS*2*RANK*2
    gram_metrics=HEADS*2*RANK*RANK*2
    larc_kv=latent_payload+basis_payload+basis_scales+gram_metrics
    baseline=BASELINE_W+baseline_kv+SCRATCH
    larc=LARC_W+larc_kv+SCRATCH
    return {"context":seq,"baseline_fp16_kv_bytes":baseline_kv,"larc_q2_kv_plus_q4_bases_and_two_metrics_bytes":larc_kv,"kv_reduction_x":baseline_kv/larc_kv,"baseline_total_bytes":baseline,"larc_total_bytes":larc,"modeled_total_reduction_x":baseline/larc}

def main():
    rows=[row(x) for x in (64,128,256,512,1024,2048,4096,8192)]
    Path('benchmarks/run4_context_sweep.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
