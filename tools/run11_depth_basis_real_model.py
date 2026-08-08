#!/usr/bin/env python3
"""Run 11: low-rank decomposition across DEPTH, not within matrices.

For a fixed operator family and a group of n logical layers, flatten each full
matrix W_l and factor the stack along the layer axis:

    W_l ~= sum_j c[l,j] * B_j

B_j remains a full-rank matrix in input/output dimensions. Only the depth axis
is compressed. Direct block sharing is depth-rank 1 with a highly constrained
basis; this experiment tests the optimal SVD depth subspace before adding any
quantization.

This is structural evidence only. It deliberately keeps factors in FP32 for the
quality test so a failure cannot be blamed on an internal weak Q4 baseline.
"""
from __future__ import annotations
import argparse,copy,json,math
from pathlib import Path
import torch

from run6_real_model_falsification import TEXT,find_projection,get_layers,nll

SITES=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")


def parse_ints(s): return tuple(int(x) for x in s.split(',') if x.strip())

def groups(n,size): return [list(range(i,min(i+size,n))) for i in range(0,n,size)]


def depth_decompose(layers,site,group,max_rank):
    mats=[find_projection(layers[i],site).weight.detach().float().cpu().reshape(-1) for i in group]
    M=torch.stack(mats,0) # n x p
    # Eigendecompose only the tiny n x n depth Gram. This yields the exact right
    # singular subspace without an expensive p-dimensional SVD.
    G=M@M.t()
    vals,U=torch.linalg.eigh(G)
    order=torch.argsort(vals,descending=True);vals=vals[order].clamp_min(0);U=U[:,order]
    r=min(max_rank,M.shape[0])
    s=vals[:r].sqrt().clamp_min(1e-12)
    Ur=U[:,:r]
    B=(Ur.t()@M)/s[:,None] # right singular vectors, r x p
    C=Ur*s[None,:]        # n x r
    total_energy=float((M*M).sum().clamp_min(1e-30))
    prefix=[]
    recon=torch.zeros_like(M)
    for j in range(r):
        recon += C[:,j:j+1]*B[j:j+1]
        prefix.append(float(((M-recon)**2).sum()/total_energy))
    return {'shape':tuple(find_projection(layers[group[0]],site).weight.shape),'group':group,'B':B,'C':C,'residual_energy_by_rank':prefix,'original_elements':M.numel(),'matrix_elements':M.shape[1]}


def build_packs(base,group_size,max_rank):
    layers=get_layers(base);packs={}
    for site in SITES:
        packs[site]=[]
        for g in groups(len(layers),group_size):
            packs[site].append(depth_decompose(layers,site,g,max_rank))
    return packs


def apply_rank_(candidate,packs,rank):
    layers=get_layers(candidate);compressed=0;orig=0;weighted_num=0.0;weighted_den=0.0
    with torch.no_grad():
        for site,sitepacks in packs.items():
            for p in sitepacks:
                r=min(rank,p['B'].shape[0],len(p['group']))
                B=p['B'][:r];C=p['C'][:,:r]
                R=C@B
                for row,li in enumerate(p['group']):
                    w=find_projection(layers[li],site).weight
                    w.copy_(R[row].reshape(p['shape']))
                orig += p['original_elements']
                compressed += r*p['matrix_elements'] + len(p['group'])*r
                # Exact relative weight residual for this stack.
                frac=p['residual_energy_by_rank'][r-1]
                weighted_num += frac*p['original_elements']
                weighted_den += p['original_elements']
    return {'original_projection_elements':orig,'compressed_projection_elements':compressed,'projection_element_reduction_x':orig/compressed,'mean_weight_residual_energy_fraction_weighted_by_elements':weighted_num/weighted_den}


def unique_params(m): return sum(p.numel() for p in m.parameters())

def target_projection_elements(m):
    return sum(find_projection(layer,site).weight.numel() for layer in get_layers(m) for site in SITES)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='HuggingFaceTB/SmolLM2-135M');ap.add_argument('--eval-tokens',type=int,default=512);ap.add_argument('--group-sizes',default='10,15,30');ap.add_argument('--ranks',default='1,2,3,4,6,8,12,16,24');ap.add_argument('--out',type=Path,default=Path('benchmarks/run11_depth_basis_real_model.json'));a=ap.parse_args()
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.manual_seed(11);torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    tok=AutoTokenizer.from_pretrained(a.model);base=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.float32).eval().cpu()
    ids=tok(TEXT*8,return_tensors='pt',add_special_tokens=False).input_ids
    if ids.shape[1]<a.eval_tokens+1:raise RuntimeError('insufficient eval text')
    ev=ids[:,-(a.eval_tokens+1):]
    base_nll=nll(base,ev);original_total=unique_params(base);original_proj=target_projection_elements(base)
    candidates=[];diagnostics={}
    ranks=parse_ints(a.ranks)
    for gs in parse_ints(a.group_sizes):
        packs=build_packs(base,gs,max(ranks));diagnostics[str(gs)]={}
        for site,sitepacks in packs.items():
            diagnostics[str(gs)][site]=[{'group':p['group'],'residual_energy_by_rank':p['residual_energy_by_rank']} for p in sitepacks]
        for rank in ranks:
            if rank>gs: continue
            m=copy.deepcopy(base).eval();acct=apply_rank_(m,packs,rank);qnll=nll(m,ev)
            candidate_total=original_total-original_proj+acct['compressed_projection_elements']
            candidates.append({'group_size':gs,'depth_rank':rank,'nll':qnll,'delta_nats_vs_fp32':qnll-base_nll,'ppl_ratio_vs_fp32':math.exp(qnll-base_nll),'whole_parameter_elements':candidate_total,'whole_parameter_reduction_x':original_total/candidate_total,**acct})
            del m
        del packs
    out={'run':11,'evidence_level':'L3-precheck real pretrained cross-depth tensor factorization','model':a.model,'model_commit':getattr(base.config,'_commit_hash',None),'protocol':{'evaluation_tokens':a.eval_tokens,'sites':SITES,'group_sizes':parse_ints(a.group_sizes),'depth_ranks':ranks,'factorization':'exact SVD of flattened full matrices along layer/depth axis via depth Gram eigendecomposition','factor_dtype_for_quality':'FP32','embeddings_norms_and_non_projection_parameters':'unchanged FP32','note':'No activation calibration or quantization is used; this isolates structural depth redundancy.'},'baseline':{'fp32_nll':base_nll,'unique_parameter_elements':original_total,'target_projection_elements':original_proj,'target_projection_fraction':original_proj/original_total},'candidates':candidates,'depth_spectrum_diagnostics':diagnostics,'claim_boundary':'Structural full-matrix depth-sharing test only. Element-count reductions are not packed file/RSS reductions; factors are FP32 and no native depth-basis kernel is claimed.'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'baseline':out['baseline'],'candidates':candidates},indent=2))
if __name__=='__main__':main()
