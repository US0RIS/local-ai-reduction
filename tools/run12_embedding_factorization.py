#!/usr/bin/env python3
"""Run 12: post-hoc tied embedding/LM-head factorization on SmolLM2.

The same low-rank approximation E ~= A B is used for both token lookup and the
tied output head:
  embed(ids) = A[ids] B
  logits(h)  = (h B^T) A^T
No dense E is used by the candidate forward path.
"""
from __future__ import annotations
import argparse,copy,json,math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from run6_real_model_falsification import TEXT,nll

class FactorizedEmbedding(nn.Module):
    def __init__(self,A:nn.Parameter,B:nn.Parameter):
        super().__init__();self.A=A;self.B=B
    @property
    def weight(self):
        # Some HF helpers inspect .weight, but forward does not materialize it.
        return self.A@self.B
    def forward(self,input_ids):
        return F.embedding(input_ids,self.A)@self.B

class FactorizedHead(nn.Module):
    def __init__(self,A:nn.Parameter,B:nn.Parameter):
        super().__init__();self.A=A;self.B=B
    @property
    def weight(self): return self.A@self.B
    def forward(self,h): return (h@self.B.t())@self.A.t()


def factor_embedding(E:torch.Tensor,max_rank:int):
    X=E.detach().float().cpu();G=X.t()@X
    vals,V=torch.linalg.eigh(G);order=torch.argsort(vals,descending=True);V=V[:,order[:max_rank]]
    return X@V,V.t() # A [v,r], B [r,d]

def unique_params(m): return sum(p.numel() for p in m.parameters())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='HuggingFaceTB/SmolLM2-135M');ap.add_argument('--eval-tokens',type=int,default=512);ap.add_argument('--ranks',default='32,48,64,96,128,192,256,384');ap.add_argument('--out',type=Path,default=Path('benchmarks/run12_embedding_factorization.json'));a=ap.parse_args()
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.manual_seed(12);torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    ranks=tuple(int(x) for x in a.ranks.split(','))
    tok=AutoTokenizer.from_pretrained(a.model);base=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.float32).eval().cpu()
    ids=tok(TEXT*8,return_tensors='pt',add_special_tokens=False).input_ids
    if ids.shape[1]<a.eval_tokens+1:raise RuntimeError('insufficient eval text')
    ev=ids[:,-(a.eval_tokens+1):];base_nll=nll(base,ev)
    E=base.model.embed_tokens.weight.detach().float().cpu();vocab,hidden=E.shape;Amax,Bmax=factor_embedding(E,max(ranks))
    orig_total=unique_params(base);orig_emb=E.numel();rows=[]
    total_energy=float((E*E).sum().clamp_min(1e-30))
    for r in ranks:
        A=nn.Parameter(Amax[:,:r].clone(),requires_grad=False);B=nn.Parameter(Bmax[:r].clone(),requires_grad=False)
        m=copy.deepcopy(base).eval();m.model.embed_tokens=FactorizedEmbedding(A,B);m.lm_head=FactorizedHead(A,B)
        q=nll(m,ev);comp=A.numel()+B.numel();candidate_total=orig_total-orig_emb+comp
        recon=A@B;res=float(((E-recon)**2).sum()/total_energy)
        rows.append({'rank':r,'embedding_residual_energy_fraction':res,'nll':q,'delta_nats_vs_fp32':q-base_nll,'ppl_ratio_vs_fp32':math.exp(q-base_nll),'original_embedding_elements':orig_emb,'factor_elements':comp,'embedding_element_reduction_x':orig_emb/comp,'whole_parameter_reduction_x':orig_total/candidate_total})
        del m,recon
    out={'run':12,'evidence_level':'L3-precheck real pretrained tied embedding/head structural factorization','model':a.model,'model_commit':getattr(base.config,'_commit_hash',None),'protocol':{'evaluation_tokens':a.eval_tokens,'ranks':ranks,'factorization':'optimal right-singular subspace of tied embedding via 576x576 Gram eigendecomposition','candidate_forward':'factorized token lookup and factorized output logits; no dense embedding in forward','factor_dtype':'FP32'},'baseline':{'fp32_nll':base_nll,'vocab_size':vocab,'hidden_size':hidden,'embedding_elements':orig_emb,'embedding_parameter_fraction':orig_emb/orig_total,'unique_parameter_elements':orig_total},'ranks':rows,'claim_boundary':'Structural factorization only. Factors remain FP32; no packed bytes/RSS or new tokenizer is claimed.'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'baseline':out['baseline'],'ranks':rows},indent=2))
if __name__=='__main__':main()
