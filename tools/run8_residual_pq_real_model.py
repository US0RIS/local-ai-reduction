#!/usr/bin/env python3
"""Run 8B: per-layer activation-aware residual product quantization.

Run 8A showed that a single 16-way choice over a 32-D direction leaves ~1/3 of
activation-weighted vector energy unexplained even after eight greedy residual
stages. 8B keeps the same 32-weight magnitude block but splits its normalized
direction into four independent 8-D subspaces. Each residual-PQ stage therefore
stores four 4-bit indices (16 bits/vector = 0.50 bpw), while codebooks are now
per-layer/per-operator rather than shared across depth.

With the same FP16 magnitude (0.50 bpw), stages 1/2/3 have nominal vector payloads
1.0/1.5/2.0 bpw before small codebook and RMS metadata.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math
from pathlib import Path
import torch

from run6_real_model_falsification import TEXT,find_projection,get_layers,nll
from run8_additive_vq_real_model import (
    SITES,row_q4_bytes,modeled_rowq4_bytes,project_model_row_q4_,collect_input_rms,
    vectors_and_scale,nearest,fit_kmeans,target_rowq4_bytes,zero_targets_
)


def fit_matrix_books(base,rms,subspaces,k,stages,max_sample,iters):
    layers=get_layers(base);books={};diag={}
    for li,layer in enumerate(layers):
        for si,site in enumerate(SITES):
            w=find_projection(layer,site).weight
            norm,_,_=vectors_and_scale(w,rms[(site,li)],32)
            x=norm.reshape(norm.shape[0],subspaces,32//subspaces)
            if x.shape[0]>max_sample:
                g=torch.Generator().manual_seed(880000+li*101+si)
                x=x[torch.randperm(x.shape[0],generator=g)[:max_sample]]
            residual=x.clone();stages_out=[];dout=[]
            original_energy=float(x.square().sum().clamp_min(1e-30))
            for st in range(stages):
                sb=[]
                for s in range(subspaces):
                    g=torch.Generator().manual_seed(890000+li*10000+si*100+st*10+s)
                    c=fit_kmeans(residual[:,s,:],k,iters,g)
                    idx=nearest(residual[:,s,:],c)
                    residual[:,s,:]-=c[idx];sb.append(c)
                stages_out.append(sb)
                dout.append({'stage':st+1,'sample_residual_energy_fraction':float(residual.square().sum()/original_energy),'sample_residual_mse':float(residual.square().mean())})
            books[(site,li)]=stages_out
            diag[f'{site}:{li}']={'vectors_used':x.shape[0],'stages':dout}
    return books,diag


def rpq_bytes(base,stage,subspaces,k):
    nvec=0;rms_bytes=0;nmat=0
    for li,layer in enumerate(get_layers(base)):
        for site in SITES:
            w=find_projection(layer,site).weight;rows,cols=w.shape
            nvec+=rows*(cols//32);rms_bytes+=cols*2;nmat+=1
    scale_bytes=nvec*2
    # 4 subspaces x 4 bits = 2 bytes/vector per residual-PQ stage.
    index_bytes=nvec*stage*math.ceil(subspaces*4/8)
    codebook_bytes=nmat*stage*subspaces*k*(32//subspaces)*2
    total=scale_bytes+index_bytes+codebook_bytes+rms_bytes
    return {'vectors':nvec,'matrices':nmat,'scale_bytes':scale_bytes,'index_bytes':index_bytes,'codebook_bytes':codebook_bytes,'activation_rms_bytes':rms_bytes,'total_vq_bytes':total,'nominal_target_bpw':(scale_bytes+index_bytes)*8/(nvec*32)}


def apply_stage_(base,candidate,rms,books,stage,subspaces,trace):
    bl=get_layers(base);cl=get_layers(candidate);sd=32//subspaces
    with torch.no_grad():
        for li,(lb,lc) in enumerate(zip(bl,cl)):
            for site in SITES:
                ow=find_projection(lb,site).weight.detach().float().cpu();cw=find_projection(lc,site).weight.detach().float().cpu()
                norm,scale,rb=vectors_and_scale(ow,rms[(site,li)],32)
                rows,cols=ow.shape;nb=cols//32
                curz=cw.reshape(rows,nb,32)*rb.unsqueeze(0)
                curnorm=(curz/scale.reshape(rows,nb).unsqueeze(-1)).reshape(-1,subspaces,sd)
                target=norm.reshape(-1,subspaces,sd);residual=target-curnorm
                new=curnorm.clone()
                for s in range(subspaces):
                    code=books[(site,li)][stage-1][s]
                    idx=nearest(residual[:,s,:],code)
                    new[:,s,:]+=code[idx]
                    trace.update(site.encode());trace.update(li.to_bytes(2,'little'));trace.update(stage.to_bytes(1,'little'));trace.update(s.to_bytes(1,'little'));trace.update(idx.to(torch.uint8).numpy().tobytes())
                newz=new.reshape(rows,nb,32)*scale.reshape(rows,nb).unsqueeze(-1)
                neww=(newz/rb.unsqueeze(0)).reshape(rows,cols)
                find_projection(lc,site).weight.copy_(neww)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='HuggingFaceTB/SmolLM2-135M');ap.add_argument('--cal-tokens',type=int,default=128);ap.add_argument('--eval-tokens',type=int,default=512);ap.add_argument('--subspaces',type=int,default=4);ap.add_argument('--codebook-size',type=int,default=16);ap.add_argument('--max-stages',type=int,default=3);ap.add_argument('--sample-vectors',type=int,default=4096);ap.add_argument('--kmeans-iters',type=int,default=6);ap.add_argument('--out',type=Path,default=Path('benchmarks/run8_residual_pq_real_model.json'));a=ap.parse_args()
    assert 32%a.subspaces==0
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.manual_seed(88);torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    tok=AutoTokenizer.from_pretrained(a.model);base=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.float32).eval().cpu()
    ids=tok(TEXT*8,return_tensors='pt',add_special_tokens=False).input_ids;need=a.cal_tokens+a.eval_tokens+1
    if ids.shape[1]<need:raise RuntimeError(f'need {need}, got {ids.shape[1]}')
    cal=ids[:,:a.cal_tokens];ev=ids[:,a.cal_tokens:need]
    fp32_nll=nll(base,ev);rowq4=copy.deepcopy(base).eval();project_model_row_q4_(rowq4);rowq4_nll=nll(rowq4,ev)
    baseline_bytes=modeled_rowq4_bytes(rowq4);target_q4=target_rowq4_bytes(base);rms=collect_input_rms(base,cal)
    books,fitdiag=fit_matrix_books(base,rms,a.subspaces,a.codebook_size,a.max_stages,a.sample_vectors,a.kmeans_iters)
    candidate=copy.deepcopy(rowq4).eval();zero_targets_(candidate);trace=hashlib.sha256();results=[]
    for stage in range(1,a.max_stages+1):
        apply_stage_(base,candidate,rms,books,stage,a.subspaces,trace)
        q=nll(candidate,ev);vb=rpq_bytes(base,stage,a.subspaces,a.codebook_size);total=baseline_bytes-target_q4+vb['total_vq_bytes']
        results.append({'stages':stage,'target_nominal_bpw_excluding_codebook_rms':vb['nominal_target_bpw'],'vq_payload':vb,'nll':q,'delta_nats_vs_fp32':q-fp32_nll,'ppl_ratio_vs_fp32':math.exp(q-fp32_nll),'delta_nats_vs_row_q4':q-rowq4_nll,'ppl_ratio_vs_row_q4':math.exp(q-rowq4_nll),'modeled_candidate_weight_bytes':total,'whole_model_weight_reduction_x':baseline_bytes/total,'target_matrix_reduction_x':target_q4/vb['total_vq_bytes'],'assignment_trace_sha256_through_stage':trace.copy().hexdigest()})
    out={'run':'8B','evidence_level':'L3-precheck real pretrained residual product-VQ experiment','model':a.model,'model_commit':getattr(base.config,'_commit_hash',None),'protocol':{'calibration_tokens':a.cal_tokens,'evaluation_tokens':a.eval_tokens,'weight_vector_dim':32,'subspaces':a.subspaces,'subspace_dim':32//a.subspaces,'codebook_size':a.codebook_size,'max_stages':a.max_stages,'sample_vectors_per_matrix':a.sample_vectors,'kmeans_iters':a.kmeans_iters,'codebook_scope':'per layer and operator','activation_weighting':'per-layer input-feature RMS','magnitude':'FP16 per 32-weight vector','indices':'4 bits per 8-D subspace per residual stage','non_target_representation':'simple row-Q4 / FP16 1-D','execution':'exact dense reconstruction for quality only'},'baselines':{'fp32_nll':fp32_nll,'row_q4_nll':rowq4_nll,'row_q4_ppl_ratio_vs_fp32':math.exp(rowq4_nll-fp32_nll),'modeled_row_q4_weight_bytes':baseline_bytes,'target_projection_row_q4_bytes':target_q4},'codebook_fit_diagnostics':fitdiag,'stage_results':results,'claim_boundary':'Real-model exact-reconstruction RPQ quality and modeled weight bytes only; no packed kernel, RSS/VRAM, competitive Q4_K_M baseline, or standard benchmark claim.'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'baselines':out['baselines'],'stage_results':results},indent=2))
if __name__=='__main__':main()
