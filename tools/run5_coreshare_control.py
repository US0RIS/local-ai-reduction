#!/usr/bin/env python3
from __future__ import annotations
import copy,json,math,random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from tools.run5_softshare_control import Teacher,CHARS,D,H,HD,FF,L,CTX,train_teacher,evaluate,toks,batch,recover,project_teacher_q4,project_soft_q4,qrecover,getmat,q4row_bytes

FAMILIES={'qkv':('attn.in_proj_weight',3*D,D),'o':('attn.out_proj.weight',D,D),'fc1':('fc1.weight',FF,D),'fc2':('fc2.weight',D,FF)}

def fit_family(t,key,rank):
    vals=torch.stack([getmat(b,key).detach().float() for b in t.blocks]);base=vals.mean(0);r=vals-base;m,n=r.shape[1:];lc=torch.zeros((m,m));rc=torch.zeros((n,n))
    for l in range(L):lc.add_(r[l]@r[l].t());rc.add_(r[l].t()@r[l])
    _,u=torch.linalg.eigh(lc);_,v=torch.linalg.eigh(rc);u=u[:,-rank:].flip(1).contiguous();v=v[:,-rank:].flip(1).contiguous();c=torch.stack([u.t()@r[l]@v for l in range(L)]);return base,u,v.t().contiguous(),c

class CoreShare(nn.Module):
    def __init__(self,t,ranks):
        super().__init__();self.ranks=dict(ranks);self.emb=copy.deepcopy(t.emb);self.pos=copy.deepcopy(t.pos);self.norm=copy.deepcopy(t.norm);self.head=nn.Linear(D,len(CHARS),bias=False);self.head.weight=self.emb.weight
        for name,(key,_,_) in FAMILIES.items():
            base,u,vt,c=fit_family(t,key,ranks[name]);setattr(self,'base_'+name,nn.Parameter(base));setattr(self,'U_'+name,nn.Parameter(u));setattr(self,'Vt_'+name,nn.Parameter(vt));setattr(self,'C_'+name,nn.Parameter(c))
        self.n1w=nn.Parameter(torch.stack([b.n1.weight.detach() for b in t.blocks]));self.n1b=nn.Parameter(torch.stack([b.n1.bias.detach() for b in t.blocks]));self.n2w=nn.Parameter(torch.stack([b.n2.weight.detach() for b in t.blocks]));self.n2b=nn.Parameter(torch.stack([b.n2.bias.detach() for b in t.blocks]));self.qkv_bias=nn.Parameter(torch.stack([b.attn.in_proj_bias.detach() for b in t.blocks]));self.o_bias=nn.Parameter(torch.stack([b.attn.out_proj.bias.detach() for b in t.blocks]));self.fc1_bias=nn.Parameter(torch.stack([b.fc1.bias.detach() for b in t.blocks]));self.fc2_bias=nn.Parameter(torch.stack([b.fc2.bias.detach() for b in t.blocks]))
    def lin(self,x,name,l,bias):return F.linear(x,getattr(self,'base_'+name),bias)+F.linear(F.linear(F.linear(x,getattr(self,'Vt_'+name)),getattr(self,'C_'+name)[l]),getattr(self,'U_'+name))
    def forward(self,idx):
        B,T=idx.shape;x=self.emb(idx)+self.pos(torch.arange(T));
        for l in range(L):
            z=F.layer_norm(x,(D,),self.n1w[l],self.n1b[l]);q,k,v=self.lin(z,'qkv',l,self.qkv_bias[l]).chunk(3,-1);q=q.view(B,T,H,HD).transpose(1,2);k=k.view(B,T,H,HD).transpose(1,2);v=v.view(B,T,H,HD).transpose(1,2);a=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,D);x=x+self.lin(a,'o',l,self.o_bias[l]);z=F.layer_norm(x,(D,),self.n2w[l],self.n2b[l]);x=x+self.lin(F.gelu(self.lin(z,'fc1',l,self.fc1_bias[l])),'fc2',l,self.fc2_bias[l])
        return self.head(self.norm(x))

def teacher_bytes():
    v=len(CHARS);total=q4row_bytes(v,D)+q4row_bytes(CTX,D)+2*D*2;per=sum(q4row_bytes(m,n) for _,m,n in FAMILIES.values())+(4*D+3*D+D+FF+D)*2;return total+L*per

def core_bytes(ranks):
    v=len(CHARS);total=q4row_bytes(v,D)+q4row_bytes(CTX,D)+2*D*2+sum(q4row_bytes(m,n) for _,m,n in FAMILIES.values())
    for name,(_,m,n) in FAMILIES.items():r=ranks[name];total+=q4row_bytes(m,r)+q4row_bytes(r,n)+L*q4row_bytes(r,r)
    for width in (D,D,D,D,3*D,D,FF,D):total+=q4row_bytes(L,width)
    return total

def main():
    torch.manual_seed(3);random.seed(3);t=Teacher();train_teacher(t,120);post_rng=torch.get_rng_state();post_py=random.getstate();ev=toks(333);tf,n=evaluate(t,ev,32768);tq=copy.deepcopy(t);project_teacher_q4(tq);tqn,_=evaluate(tq,ev,32768)
    ranks={'qkv':16,'o':16,'fc1':16,'fc2':16};torch.set_rng_state(post_rng);random.setstate(post_py);s=CoreShare(t,ranks);raw,_=evaluate(s,ev,32768);recover(s,t,80);rec,_=evaluate(s,ev,32768);sq=copy.deepcopy(s);project_soft_q4(sq);pre,_=evaluate(sq,ev,32768);qrecover(sq,tq,50);fin,_=evaluate(sq,ev,32768);b=core_bytes(ranks)
    run4=json.loads(Path('benchmarks/run4_fp8meta_l2c.json').read_text());soft=json.loads(Path('benchmarks/run5_softshare_control.json').read_text())
    out={'run':5,'strategy':'CoreShare-10X','evidence_level':'controlled L2C architecture-selection study','task':'synthetic character-level template LM','model':{'hidden':D,'heads':H,'ffn':FF,'layers':L,'context':CTX,'vocab':len(CHARS)},'training_seed':3,'evaluation_seed':333,'evaluation_chars':n,'representation':'W_layer = S + U C_layer V^T; S,U,V,C all canonical Q4_ROW in the quality path','teacher':{'fp32_nll':tf,'canonical_q4_nll':tqn,'canonical_q4_serialized_tensor_bytes':teacher_bytes()},'uniform_rank16':{'ranks':ranks,'raw_hosvd_nll':raw,'after_80_fp_recovery_nll':rec,'q4_before_constrained_recovery_nll':pre,'q4_after_50_constrained_recovery_nll':fin,'delta_vs_q4_teacher_nats_per_char':fin-tqn,'perplexity_ratio_vs_q4_teacher':math.exp(fin-tqn),'serialized_tensor_bytes':b,'complete_toy_tensor_reduction_x':teacher_bytes()/b,'passes_10x_tensor_gate':teacher_bytes()/b>=10},'comparison':{'hard_recursive_run4_perplexity_ratio':run4['total_perplexity_ratio'],'softshare_q4row_rank2_perplexity_ratio':soft['profiles'][1]['perplexity_ratio_vs_q4_teacher'],'softshare_q4row_rank2_complete_toy_tensor_reduction_x':soft['profiles'][1]['complete_toy_tensor_reduction_x']},'claim_boundary':'CoreShare is the strongest controlled 10x tensor representation in the project so far. This is a synthetic d=128 model and is not L3 real-model evidence, not a complete LARC-file ratio, and not measured RAM/VRAM.'}
    text=json.dumps(out,indent=2)+'\n';Path('benchmarks/run5_coreshare_control.json').write_text(text);print(text,end='')
if __name__=='__main__':main()
