#!/usr/bin/env python3
"""Run 8: activation-aware additive vector quantization on SmolLM2.

The experiment intentionally abandons low-rank cross-layer sharing after Run 6/7.
Every logical matrix remains distinct.  Contiguous D=32 weight vectors are encoded
as a per-vector FP16 magnitude plus M 4-bit indices into residual codebooks:

    diag(rms_x) w ~= scale * sum_m codebook[group, site, m][index_m]

Codebooks are shared across ten adjacent layers of the same operator family;
indices and magnitudes are layer/matrix specific.  The activation RMS makes the
Euclidean codebook loss a diagonal-Hessian approximation to output error.

For D=32 and K=16, the variable-rate payload is:
  scale: 16/32 = 0.50 bits/weight
  each residual stage: 4/32 = 0.125 bits/weight
Thus M={2,4,6,8} gives {0.75,1.00,1.25,1.50} bpw before small codebook/RMS metadata.

Quality execution reconstructs exactly from FP16 scales, FP16 codebooks and the
chosen indices.  Dense reconstruction is evaluation-only; no native lookup-GEMV
or measured RSS claim is made here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from run6_real_model_falsification import TEXT, find_projection, get_layers, nll

SITES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
SITE_PARENT={"q_proj":"self_attn","k_proj":"self_attn","v_proj":"self_attn","o_proj":"self_attn","gate_proj":"mlp","up_proj":"mlp","down_proj":"mlp"}


def row_q4_tensor(w:torch.Tensor)->torch.Tensor:
    x=w.detach().float()
    if x.ndim!=2:return x.half().float()
    pos=x.amax(1).clamp_min(0)/7.0;neg=(-x.amin(1)).clamp_min(0)/8.0
    sc=torch.maximum(pos,neg).clamp_min(1e-8).half().float()
    return torch.round(x/sc[:,None]).clamp(-8,7)*sc[:,None]


def project_model_row_q4_(m):
    with torch.no_grad():
        seen=set()
        for p in m.parameters():
            ptr=p.untyped_storage().data_ptr()
            if ptr in seen:continue
            seen.add(ptr);p.copy_(row_q4_tensor(p))


def row_q4_bytes(rows:int,cols:int)->int:
    return rows*(((cols+1)//2)+2)


def modeled_rowq4_bytes(m)->int:
    total=0;seen=set()
    for p in m.parameters():
        ptr=p.untyped_storage().data_ptr()
        if ptr in seen:continue
        seen.add(ptr)
        total+=row_q4_bytes(p.shape[0],p.shape[1]) if p.ndim==2 else p.numel()*2
    return total


def collect_input_rms(model,ids):
    layers=get_layers(model);out={};handles=[]
    for li,layer in enumerate(layers):
        for site in SITES:
            mod=find_projection(layer,site)
            def hook(_m,inp,_out,li=li,site=site):
                x=inp[0].detach().float().reshape(-1,inp[0].shape[-1])
                out[(site,li)]=x.square().mean(0).sqrt().clamp_min(1e-4).cpu()
            handles.append(mod.register_forward_hook(hook))
    with torch.inference_mode():model(input_ids=ids,use_cache=False)
    for h in handles:h.remove()
    return out


def vectors_and_scale(w:torch.Tensor,rms:torch.Tensor,d:int):
    w=w.detach().float().cpu();rms=rms.float().cpu().clamp_min(1e-4)
    rows,cols=w.shape
    if cols%d:raise ValueError(f"{cols=} not divisible by {d=}")
    nb=cols//d
    rb=rms.reshape(nb,d)
    z=w.reshape(rows,nb,d)*rb.unsqueeze(0)
    scale=z.square().mean(-1).sqrt().clamp_min(1e-8).half().float()
    norm=z/scale.unsqueeze(-1)
    return norm.reshape(-1,d),scale.reshape(-1),rb


def nearest(x:torch.Tensor,c:torch.Tensor,chunk:int=16384):
    ans=[];ct=c.t();c2=c.square().sum(1)
    for s in range(0,x.shape[0],chunk):
        a=x[s:s+chunk]
        dist=a.square().sum(1,keepdim=True)+c2.unsqueeze(0)-2.0*(a@ct)
        ans.append(dist.argmin(1))
    return torch.cat(ans)


def kmeans_pp(x:torch.Tensor,k:int,g:torch.Generator):
    n=x.shape[0];first=int(torch.randint(n,(1,),generator=g))
    centers=[x[first]]
    mind=(x-centers[0]).square().sum(1)
    for _ in range(1,k):
        probs=mind.clamp_min(1e-12);probs=probs/probs.sum()
        idx=int(torch.multinomial(probs,1,generator=g))
        c=x[idx];centers.append(c)
        mind=torch.minimum(mind,(x-c).square().sum(1))
    return torch.stack(centers)


def fit_kmeans(x:torch.Tensor,k:int,iters:int,g:torch.Generator):
    c=kmeans_pp(x,k,g)
    for _ in range(iters):
        idx=nearest(x,c)
        sums=torch.zeros_like(c);counts=torch.zeros(k,dtype=torch.float32)
        sums.index_add_(0,idx,x);counts.index_add_(0,idx,torch.ones(idx.shape[0]))
        nz=counts>0
        new=c.clone();new[nz]=sums[nz]/counts[nz,None]
        c=new
    # These are the actual stored codebook values.
    c=c.half().float()
    return c


def layer_groups(n:int,size:int):
    return [list(range(i,min(n,i+size))) for i in range(0,n,size)]


def sample_group_vectors(base,rms,site,group,d,max_sample,seed):
    layers=get_layers(base);pieces=[];quota=max(1,max_sample//len(group))
    for li in group:
        w=find_projection(layers[li],site).weight
        v,_,_=vectors_and_scale(w,rms[(site,li)],d)
        if v.shape[0]>quota:
            g=torch.Generator().manual_seed(seed+li*1009)
            ix=torch.randperm(v.shape[0],generator=g)[:quota];v=v[ix]
        pieces.append(v)
    x=torch.cat(pieces,0)
    if x.shape[0]>max_sample:
        g=torch.Generator().manual_seed(seed+9176)
        x=x[torch.randperm(x.shape[0],generator=g)[:max_sample]]
    return x


def fit_codebooks(base,rms,d,k,stages,group_size,max_sample,iters):
    layers=get_layers(base);groups=layer_groups(len(layers),group_size)
    books={};diagnostics={}
    for si,site in enumerate(SITES):
        for gi,group in enumerate(groups):
            x=sample_group_vectors(base,rms,site,group,d,max_sample,8000+si*100+gi)
            residual=x.clone();lst=[];diag=[]
            for stage in range(stages):
                g=torch.Generator().manual_seed(810000+si*10000+gi*100+stage)
                c=fit_kmeans(residual,k,iters,g)
                idx=nearest(residual,c);residual=residual-c[idx]
                lst.append(c)
                diag.append({"stage":stage+1,"sample_residual_mse":float(residual.square().mean()),"sample_residual_energy_fraction":float(residual.square().sum()/x.square().sum().clamp_min(1e-30))})
            books[(site,gi)]=lst;diagnostics[f"{site}:{gi}"]={"layers":group,"sample_vectors":x.shape[0],"stages":diag}
    return books,diagnostics,groups


def zero_targets_(m):
    with torch.no_grad():
        for layer in get_layers(m):
            for site in SITES:find_projection(layer,site).weight.zero_()


def target_rowq4_bytes(base):
    total=0;vectors=0
    for layer in get_layers(base):
        for site in SITES:
            w=find_projection(layer,site).weight
            total+=row_q4_bytes(w.shape[0],w.shape[1])
    return total


def vq_bytes(base,stages,d,k,group_size):
    layers=get_layers(base);groups=layer_groups(len(layers),group_size)
    nvec=0;rms_bytes=0
    for li,layer in enumerate(layers):
        for site in SITES:
            w=find_projection(layer,site).weight;rows,cols=w.shape
            nvec+=rows*(cols//d);rms_bytes+=cols*2  # conservative: duplicate RMS by site/layer
    scale_bytes=nvec*2
    index_bytes=nvec*math.ceil(stages*4/8)
    codebook_bytes=len(SITES)*len(groups)*stages*k*d*2
    return {"vectors":nvec,"scale_bytes":scale_bytes,"index_bytes":index_bytes,"codebook_bytes":codebook_bytes,"activation_rms_bytes":rms_bytes,"total_vq_bytes":scale_bytes+index_bytes+codebook_bytes+rms_bytes,"nominal_target_bpw":(scale_bytes+index_bytes)*8/(nvec*d)}


def apply_stage_(base,candidate,rms,books,stage,d,group_size,trace):
    bl=get_layers(base);cl=get_layers(candidate)
    with torch.no_grad():
        for li,(lb,lc) in enumerate(zip(bl,cl)):
            gi=li//group_size
            for site in SITES:
                ow=find_projection(lb,site).weight.detach().float().cpu()
                cw=find_projection(lc,site).weight.detach().float().cpu()
                norm,scale,rb=vectors_and_scale(ow,rms[(site,li)],d)
                rows,cols=ow.shape;nb=cols//d
                curz=cw.reshape(rows,nb,d)*rb.unsqueeze(0)
                curnorm=(curz/scale.reshape(rows,nb).unsqueeze(-1)).reshape(-1,d)
                residual=norm-curnorm
                code=books[(site,gi)][stage-1]
                idx=nearest(residual,code)
                newnorm=curnorm+code[idx]
                newz=newnorm.reshape(rows,nb,d)*scale.reshape(rows,nb).unsqueeze(-1)
                neww=(newz/rb.unsqueeze(0)).reshape(rows,cols)
                find_projection(lc,site).weight.copy_(neww)
                # Assignment trace is deterministic evidence, not a claim that this
                # byte order is the eventual file-format packing order.
                trace.update(site.encode());trace.update(li.to_bytes(2,'little'));trace.update(stage.to_bytes(1,'little'));trace.update(idx.to(torch.uint8).numpy().tobytes())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='HuggingFaceTB/SmolLM2-135M')
    ap.add_argument('--cal-tokens',type=int,default=128)
    ap.add_argument('--eval-tokens',type=int,default=512)
    ap.add_argument('--vector-dim',type=int,default=32)
    ap.add_argument('--codebook-size',type=int,default=16)
    ap.add_argument('--max-stages',type=int,default=8)
    ap.add_argument('--checkpoints',default='2,4,6,8')
    ap.add_argument('--layer-group-size',type=int,default=10)
    ap.add_argument('--sample-vectors',type=int,default=8192)
    ap.add_argument('--kmeans-iters',type=int,default=7)
    ap.add_argument('--out',type=Path,default=Path('benchmarks/run8_additive_vq_real_model.json'))
    a=ap.parse_args()
    checkpoints=tuple(int(x) for x in a.checkpoints.split(','));assert max(checkpoints)<=a.max_stages

    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.manual_seed(8);torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    tok=AutoTokenizer.from_pretrained(a.model)
    base=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.float32).eval().cpu()
    ids=tok(TEXT*8,return_tensors='pt',add_special_tokens=False).input_ids
    need=a.cal_tokens+a.eval_tokens+1
    if ids.shape[1]<need:raise RuntimeError(f'need {need} tokens, got {ids.shape[1]}')
    cal=ids[:,:a.cal_tokens];ev=ids[:,a.cal_tokens:need]

    fp32_nll=nll(base,ev)
    rowq4=copy.deepcopy(base).eval();project_model_row_q4_(rowq4);rowq4_nll=nll(rowq4,ev)
    baseline_bytes=modeled_rowq4_bytes(rowq4);target_q4=target_rowq4_bytes(base)
    rms=collect_input_rms(base,cal)
    books,book_diag,groups=fit_codebooks(base,rms,a.vector_dim,a.codebook_size,a.max_stages,a.layer_group_size,a.sample_vectors,a.kmeans_iters)

    candidate=copy.deepcopy(rowq4).eval();zero_targets_(candidate)
    trace=hashlib.sha256();results=[]
    for stage in range(1,a.max_stages+1):
        apply_stage_(base,candidate,rms,books,stage,a.vector_dim,a.layer_group_size,trace)
        if stage in checkpoints:
            q=nll(candidate,ev);vb=vq_bytes(base,stage,a.vector_dim,a.codebook_size,a.layer_group_size)
            total=baseline_bytes-target_q4+vb['total_vq_bytes']
            results.append({
              'stages':stage,
              'target_nominal_bpw_excluding_codebook_rms':vb['nominal_target_bpw'],
              'vq_payload':vb,
              'nll':q,
              'delta_nats_vs_fp32':q-fp32_nll,
              'ppl_ratio_vs_fp32':math.exp(q-fp32_nll),
              'delta_nats_vs_row_q4':q-rowq4_nll,
              'ppl_ratio_vs_row_q4':math.exp(q-rowq4_nll),
              'modeled_candidate_weight_bytes':total,
              'whole_model_weight_reduction_x':baseline_bytes/total,
              'target_matrix_reduction_x':target_q4/vb['total_vq_bytes'],
              'assignment_trace_sha256_through_stage':trace.copy().hexdigest(),
            })

    out={
      'run':8,'evidence_level':'L3-precheck real pretrained additive-vector-quantization experiment',
      'model':a.model,'model_commit':getattr(base.config,'_commit_hash',None),
      'protocol':{
        'calibration_tokens':a.cal_tokens,'evaluation_tokens':a.eval_tokens,'vector_dim':a.vector_dim,'codebook_size':a.codebook_size,
        'max_stages':a.max_stages,'checkpoints':checkpoints,'layer_group_size':a.layer_group_size,'sample_vectors_per_site_group':a.sample_vectors,'kmeans_iters':a.kmeans_iters,
        'target_sites':SITES,'source_weights':'FP32 pretrained projection matrices','activation_weighting':'per-layer input-feature RMS (diagonal Hessian proxy)',
        'per_vector_magnitude':'FP16 in activation-weighted space','codebooks':'FP16 residual codebooks shared by operator and 10-layer depth group','indices':'4 bits per residual stage per 32-weight vector',
        'non_target_representation':'simple row-Q4 matrices and FP16 1-D parameters','execution':'exact dense reconstruction of stored semantics for quality only'
      },
      'baselines':{'fp32_nll':fp32_nll,'row_q4_nll':rowq4_nll,'row_q4_ppl_ratio_vs_fp32':math.exp(rowq4_nll-fp32_nll),'modeled_row_q4_weight_bytes':baseline_bytes,'target_projection_row_q4_bytes':target_q4},
      'codebook_fit_diagnostics':book_diag,
      'stage_results':results,
      'claim_boundary':'Real-model compressed-representation quality and modeled weight bytes only. No packed lookup kernel, measured RSS/VRAM, Q4_K_M baseline, standard benchmark suite, or complete LARC runtime claim.'
    }
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'baselines':out['baselines'],'stage_results':results},indent=2))
if __name__=='__main__':main()
