#!/usr/bin/env python3
import json,math
from pathlib import Path
D,H,HD,L,R,FF=128,4,32,16,16,256
TEACHER_W,SHARED_W,BASIS=1129482,77322,6400
CONTEXTS=(64,256,512,1024,2048,4096,8192)
def row(T):
    baseline_kv=L*T*H*HD*4
    larc_kv=L*T*H*2*(math.ceil(R*2/8)+2)  # q2 coeffs + E4M3 min/scale
    baseline_scratch=(5*D+FF+H*T)*4
    larc_scratch=baseline_scratch+4*R*4
    baseline=TEACHER_W+baseline_kv+baseline_scratch
    larc=SHARED_W+larc_kv+BASIS+larc_scratch
    return {"context":T,"baseline_fp16_kv_bytes":baseline_kv,"larc_q2_fp8meta_kv_bytes":larc_kv,"baseline_scratch_bytes":baseline_scratch,"larc_packed_scratch_bytes":larc_scratch,"baseline_total_bytes":baseline,"larc_total_bytes":larc,"modeled_total_reduction_x":baseline/larc,"quality_validated_equivalent_codec_at_this_context":T==64}
def main():
    rows=[row(T) for T in CONTEXTS]
    # Historical artifact is intentionally compact one-object-per-line. Emit that
    # exact canonical form so provenance CI checks semantics rather than whitespace.
    text='[\n'+',\n'.join('  '+json.dumps(r,separators=(',',':')) for r in rows)+'\n]\n'
    Path('benchmarks/run4_packed_attention_context_sweep.json').write_text(text)
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()