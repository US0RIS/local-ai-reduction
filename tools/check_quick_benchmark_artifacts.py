#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from larc.latent_kv import fp16_cache_bytes,kivi_latent_cache_bytes
from tools.memory_plan import CONFIG,PROFILES,GGUF_Q4,weight_bytes,workspace_bound


def controlled_context_sweep(metadata_bytes_per_vector:int=4):
    D,H,HD,L,R=128,4,32,16,16
    teacher_w,shared_w,basis=1129482,77322,6400
    out=[]
    fp8=metadata_bytes_per_vector==2
    for T in [64,256,512,1024,2048,4096,8192]:
        basekv=L*T*H*HD*4
        q2=L*T*H*2*(math.ceil(R*2/8)+metadata_bytes_per_vector)
        scratch=(D+D+3*D+H*T+256+T*R)*4
        base=teacher_w+basekv+scratch;larc=shared_w+q2+basis+scratch
        row={'context':T,'baseline_total_bytes':base,'larc_total_bytes':larc,'modeled_total_reduction_x':base/larc,'baseline_fp16_kv_bytes':basekv,'kv_reduction_x':basekv/q2,'basis_bytes':basis,'scratch_bytes':scratch,'quality_validated_at_this_context':T==64}
        row['larc_q2_fp8meta_kv_bytes' if fp8 else 'larc_row_q2_kv_bytes']=q2
        out.append(row)
    return out


def smollm2_rank16():
    out=[]
    for name,p in PROFILES.items():
        wb=weight_bytes(p)
        for seq in [2048,8192]:
            basekv=fp16_cache_bytes(layers=CONFIG['layers'],seq=seq,kv_heads=CONFIG['kv_heads'],head_dim=CONFIG['head_dim'])
            kv=kivi_latent_cache_bytes(layers=CONFIG['layers'],seq=seq,kv_heads=CONFIG['kv_heads'],head_dim=CONFIG['head_dim'],rank=16)
            ws=workspace_bound(p);base=GGUF_Q4+basekv+ws;larc=wb+kv+ws
            out.append({'profile':name,'context':seq,'kv_rank':16,'larc_weight_bytes':wb,'weight_reduction_vs_105MB_q4km':GGUF_Q4/wb,'baseline_fp16_kv_bytes':basekv,'larc_kivi_q2_kv_bytes':kv,'kv_reduction_x':basekv/kv,'workspace_bound_bytes':ws,'baseline_modeled_total_bytes':base,'larc_modeled_total_bytes':larc,'modeled_total_memory_reduction':base/larc,'quality_validated':False})
    return out


def close(a,b,path='root'):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a:close(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),(path,len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)):close(x,y,f'{path}[{i}]')
    elif isinstance(a,float) or isinstance(b,float):
        assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-9),(path,a,b)
    else:assert a==b,(path,a,b)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rewrite',action='store_true');args=ap.parse_args()
    items=[
      ('benchmarks/run4_context_sweep.json',controlled_context_sweep(4)),
      ('benchmarks/run4_fp8meta_context_sweep.json',controlled_context_sweep(2)),
      ('benchmarks/run4_smollm2_structural_rank16.json',smollm2_rank16())]
    for rel,obj in items:
        p=ROOT/rel
        if args.rewrite:p.write_text(json.dumps(obj,indent=2)+'\n')
        else:close(json.loads(p.read_text()),obj,rel)
        print('OK',rel)
if __name__=='__main__':main()
