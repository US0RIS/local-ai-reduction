#!/usr/bin/env python3
from __future__ import annotations
import copy,json,math,random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from tools.run5_softshare_control import Teacher,D,H,HD,FF,L,CHARS,train_teacher,evaluate,toks,batch,project_teacher_q4,project_soft_q4,getmat
from tools.run5_coreshare_control import CoreShare,core_bytes,teacher_bytes,FAMILIES

def q4hat(x):
    y=x.detach().float();sh=y.shape;y=y.reshape(-1,sh[-1]);pos=y.amax(1).clamp_min(0)/7.;neg=(-y.amin(1)).clamp_min(0)/8.;s=torch.maximum(pos,neg).clamp_min(1e-8).half().float();return (torch.round(y/s[:,None]).clamp(-8,7)*s[:,None]).reshape(sh)
def init_family(t,key,rank,seed,oversample=16):
    vals=torch.stack([getmat(b,key).detach().float() for b in t.blocks]);s=q4hat(vals.mean(0));r=vals-s;m,n=r.shape[1:];q=min(rank+oversample,m,n);yu=torch.zeros((m,q));yv=torch.zeros((n,q))
    for l in range(L):
        g1=torch.Generator().manual_seed(seed+l*2);g2=torch.Generator().manual_seed(seed+l*2+1);yu.add_(r[l]@torch.randn((n,q),generator=g1));yv.add_(r[l].t()@torch.randn((m,q),generator=g2))
    u=q4hat(torch.linalg.qr(yu,mode='reduced').Q[:,:rank].contiguous());vt=q4hat(torch.linalg.qr(yv,mode='reduced').Q[:,:rank].t().contiguous());v=vt.t().contiguous();gu=u.t()@u;gv=v.t()@v;gu+=torch.eye(rank)*torch.diagonal(gu).mean().clamp_min(1e-8)*1e-5;gv+=torch.eye(rank)*torch.diagonal(gv).mean().clamp_min(1e-8)*1e-5;cores=[]
    for l in range(L):
        mid=u.t()@r[l]@v;left=torch.linalg.solve(gu,mid);cores.append(q4hat(torch.linalg.solve(gv,left.t()).t()))
    return s,u,vt,torch.stack(cores)
class ConverterCoreShare(CoreShare):
    def __init__(self,t,rank=16):
        nn.Module.__init__(self);self.ranks={k:rank for k in FAMILIES};self.emb=copy.deepcopy(t.emb);self.pos=copy.deepcopy(t.pos);self.norm=copy.deepcopy(t.norm);self.head=nn.Linear(D,len(CHARS),bias=False);self.head.weight=self.emb.weight
        for i,(name,(key,_,_)) in enumerate(FAMILIES.items()):
            s,u,vt,c=init_family(t,key,rank,2026+i*10000);setattr(self,'base_'+name,nn.Parameter(s));setattr(self,'U_'+name,nn.Parameter(u));setattr(self,'Vt_'+name,nn.Parameter(vt));setattr(self,'C_'+name,nn.Parameter(c))
        self.n1w=nn.Parameter(torch.stack([b.n1.weight.detach() for b in t.blocks]));self.n1b=nn.Parameter(torch.stack([b.n1.bias.detach() for b in t.blocks]));self.n2w=nn.Parameter(torch.stack([b.n2.weight.detach() for b in t.blocks]));self.n2b=nn.Parameter(torch.stack([b.n2.bias.detach() for b in t.blocks]));self.qkv_bias=nn.Parameter(torch.stack([b.attn.in_proj_bias.detach() for b in t.blocks]));self.o_bias=nn.Parameter(torch.stack([b.attn.out_proj.bias.detach() for b in t.blocks]));self.fc1_bias=nn.Parameter(torch.stack([b.fc1.bias.detach() for b in t.blocks]));self.fc2_bias=nn.Parameter(torch.stack([b.fc2.bias.detach() for b in t.blocks]));project_soft_q4(self)
def hidden_pairs(model,idx):
    B,T=idx.shape;x=model.emb(idx)+model.pos(torch.arange(T));mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1);ins=[];outs=[]
    with torch.inference_mode():
        for b in model.blocks:ins.append(x.detach().clone());x=b(x,mask);outs.append(x.detach().clone())
    return ins,outs
def layer_forward(m,x,l):
    B,T,_=x.shape;z=F.layer_norm(x,(D,),m.n1w[l],m.n1b[l]);q,k,v=m.lin(z,'qkv',l,m.qkv_bias[l]).chunk(3,-1);q=q.view(B,T,H,HD).transpose(1,2);k=k.view(B,T,H,HD).transpose(1,2);v=v.view(B,T,H,HD).transpose(1,2);a=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,D);x=x+m.lin(a,'o',l,m.o_bias[l]);z=F.layer_norm(x,(D,),m.n2w[l],m.n2b[l]);return x+m.lin(F.gelu(m.lin(z,'fc1',l,m.fc1_bias[l])),'fc2',l,m.fc2_bias[l])
def main():
    torch.manual_seed(3);random.seed(3);t=Teacher();train_teacher(t,120);tq=copy.deepcopy(t);project_teacher_q4(tq);ev=toks(333);tn,n=evaluate(tq,ev,32768);torch.manual_seed(123);xcal,_=batch(b=48,T=32);qin,qout=hidden_pairs(tq,xcal);s=ConverterCoreShare(tq,16);raw,_=evaluate(s,ev,32768);opt=torch.optim.AdamW(s.parameters(),lr=2e-4);rng=random.Random(100)
    for _ in range(400):
        l=rng.randrange(L);ids=torch.randint(0,xcal.shape[0],(8,));pred=layer_forward(s,qin[l][ids],l);loss=F.mse_loss(pred,qout[l][ids]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(s.parameters(),1);opt.step();project_soft_q4(s)
    lnll,_=evaluate(s,ev,32768);opt=torch.optim.AdamW(s.parameters(),lr=1.5e-4)
    for _ in range(150):
        x,y=batch();z=s(x);loss=F.cross_entropy(z.reshape(-1,len(CHARS)),y.reshape(-1));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(s.parameters(),1);opt.step();project_soft_q4(s)
    fin,_=evaluate(s,ev,32768);ranks={k:16 for k in FAMILIES};out={'run':5,'strategy':'CoreShare-10X converter-style recovery','evidence_level':'controlled L2C converter-feasibility study','source_teacher_representation':'canonical Q4_ROW','initializer':'streamable randomized shared residual subspaces; shared S/U/V quantized first; C_l least-squares fitted against stored/dequantized U/V','model':{'hidden':D,'layers':L,'context':64,'rank':16},'training_seed':3,'evaluation_seed':333,'evaluation_chars':n,'calibration':{'sequences':48,'tokens_per_sequence':32,'teacher_hidden_pairs_cached':True},'teacher_q4_nll':tn,'raw_converter_initialized_nll':raw,'after_400_layerwise_cached_target_steps_nll':lnll,'after_layerwise_perplexity_ratio':math.exp(lnll-tn),'teacher_free_q4_ce_steps':150,'final_nll':fin,'final_delta_nats_per_char':fin-tn,'final_perplexity_ratio':math.exp(fin-tn),'complete_toy_tensor_reduction_x':teacher_bytes()/core_bytes(ranks),'teacher_required_during_final_ce_recovery':False,'claim_boundary':'This validates the actual bounded-memory randomized CoreShare initialization plus streamed layerwise calibration on the controlled Q4 source. It is not L3/L4 and the sub-1 perplexity ratio reflects extra optimization of a finite-step teacher, not increased general intelligence.'};text=json.dumps(out,indent=2)+'\n';Path('benchmarks/run5_coreshare_converter_recovery.json').write_text(text);print(text,end='')
if __name__=='__main__':main()
