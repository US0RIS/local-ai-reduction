#!/usr/bin/env python3
import json,math
from pathlib import Path
D,H,HD,L,R,FF=128,4,32,16,16,256
TEACHER_W,SHARED_GROUP64_W,BASIS=1129482,79828,6400
CONTEXTS=(64,256,512,1024,2048,4096,8192)
def row(T):
    baseline_kv=L*T*H*HD*4
    larc_kv=L*T*H*2*(math.ceil(R*2/8)+2)  # Q2 coeffs + E4M3 min/scale bytes
    baseline_scratch=(5*D+FF+H*T)*4
    larc_scratch=baseline_scratch+4*R*4
    baseline=TEACHER_W+baseline_kv+baseline_scratch
    larc=SHARED_GROUP64_W+larc_kv+BASIS+larc_scratch
    return {'context':T,'baseline_fp16_kv_bytes':baseline_kv,'larc_q2_e4m3_kv_bytes':larc_kv,'baseline_scratch_bytes':baseline_scratch,'larc_direct_packed_scratch_bytes':larc_scratch,'baseline_total_bytes':baseline,'larc_total_bytes':larc,'modeled_total_reduction_x':baseline/larc,'quality_validated_context64_multiseed':T==64}
def main():
    rows=[row(T) for T in CONTEXTS];text=json.dumps(rows,indent=2);Path('benchmarks/run5_packed_context_sweep.json').write_text(text+'\n');print(text)
if __name__=='__main__':main()
