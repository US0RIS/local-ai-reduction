#!/usr/bin/env python3
from __future__ import annotations
import copy,json,math,os,random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(min(4,os.cpu_count() or 2))
NAMES=['Mia','Leo','Ava','Noah','Lily','Ben','Ella','Sam','Ivy','Max'];PLACES=['garden','forest','school','beach','farm','park','shop'];THINGS=['red ball','small key','blue kite','old book','silver coin','green box','toy boat'];ACTIONS=['found','lost','carried','opened','shared','fixed','painted']
D,H,HD,FF,L,CTX=128,4,32,256,16,64

def corpus(n,seed):
 r=random.Random(seed);out=[]
 for _ in range(n):
  a,b=r.sample(NAMES,2);p=r.choice(PLACES);t=r.choice(THINGS);v=r.choice(ACTIONS);out.append(f'Once upon a time, {a} went to the {p}. {a} {v} a {t}. Then {b} came to help. They were kind and went home happy.\n')
 return ''.join(out)
TRAIN_TEXT=corpus(6000,3);CHARS=sorted(set(TRAIN_TEXT));STOI={c:i for i,c in enumerate(CHARS)};TRAIN=torch.tensor([STOI[c] for c in TRAIN_TEXT],dtype=torch.long)
def toks(seed,n=1200):return torch.tensor([STOI[c] for c in corpus(n,seed)],dtype=torch.long)

class Block(nn.Module):
 def __init__(self):super().__init__();self.n1=nn.LayerNorm(D);self.attn=nn.MultiheadAttention(D,H,batch_first=True);self.n2=nn.LayerNorm(D);self.fc1=nn.Linear(D,FF);self.fc2=nn.Linear(FF,D)
 def forward(self,x,mask):z=self.n1(x);a,_=self.attn(z,z,z,attn_mask=mask,need_weights=False);x=x+a;z=self.n2(x);return x+self.fc2(F.gelu(self.fc1(z)))
class Teacher(nn.Module):
 def __init__(self):super().__init__();self.emb=nn.Embedding(len(CHARS),D);self.pos=nn.Embedding(CTX,D);self.blocks=nn.ModuleList([Block() for _ in range(L)]);self.norm=nn.LayerNorm(D);self.head=nn.Linear(D,len(CHARS),bias=False);self.head.weight=self.emb.weight
 def forward(self,idx):
  T=idx.shape[1];x=self.emb(idx)+self.pos(torch.arange(T));mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1)
  for b in self.blocks:x=b(x,mask)
  return self.head(self.norm(x))
def getmat(b,key):
 obj=b;ps=key.split('.')
 for p in ps[:-1]:obj=getattr(obj,p)
 return getattr(obj,ps[-1])
def svd_factors(res,r):
 U,S,Vh=torch.linalg.svd(res.float(),full_matrices=False);s=S[:r].sqrt();return U[:,:r]*s,s[:,None]*Vh[:r]

class SoftShare(nn.Module):
 def __init__(self,t,ranks):
  super().__init__();self.ranks=dict(ranks);self.emb=copy.deepcopy(t.emb);self.pos=copy.deepcopy(t.pos);self.norm=copy.deepcopy(t.norm);self.head=nn.Linear(D,len(CHARS),bias=False);self.head.weight=self.emb.weight
  keys={'qkv':('attn.in_proj_weight',3*D,D),'o':('attn.out_proj.weight',D,D),'fc1':('fc1.weight',FF,D),'fc2':('fc2.weight',D,FF)}
  for name,(key,m,n) in keys.items():
   vals=torch.stack([getmat(b,key).detach().float() for b in t.blocks]);base=vals.mean(0);As=[];Bs=[];r=ranks[name]
   for l in range(L):a,b=svd_factors(vals[l]-base,r);As.append(a);Bs.append(b)
   setattr(self,'base_'+name,nn.Parameter(base));setattr(self,'A_'+name,nn.Parameter(torch.stack(As)));setattr(self,'B_'+name,nn.Parameter(torch.stack(Bs)))
  self.n1w=nn.Parameter(torch.stack([b.n1.weight.detach() for b in t.blocks]));self.n1b=nn.Parameter(torch.stack([b.n1.bias.detach() for b in t.blocks]));self.n2w=nn.Parameter(torch.stack([b.n2.weight.detach() for b in t.blocks]));self.n2b=nn.Parameter(torch.stack([b.n2.bias.detach() for b in t.blocks]));self.qkv_bias=nn.Parameter(torch.stack([b.attn.in_proj_bias.detach() for b in t.blocks]));self.o_bias=nn.Parameter(torch.stack([b.attn.out_proj.bias.detach() for b in t.blocks]));self.fc1_bias=nn.Parameter(torch.stack([b.fc1.bias.detach() for b in t.blocks]));self.fc2_bias=nn.Parameter(torch.stack([b.fc2.bias.detach() for b in t.blocks]))
 def lin(self,x,name,l,bias):
  base=getattr(self,'base_'+name);A=getattr(self,'A_'+name)[l];B=getattr(self,'B_'+name)[l];return F.linear(x,base,bias)+F.linear(F.linear(x,B),A)
 def forward(self,idx):
  Bsz,T=idx.shape;x=self.emb(idx)+self.pos(torch.arange(T))
  for l in range(L):
   z=F.layer_norm(x,(D,),self.n1w[l],self.n1b[l]);q,k,v=self.lin(z,'qkv',l,self.qkv_bias[l]).chunk(3,-1);q=q.view(Bsz,T,H,HD).transpose(1,2);k=k.view(Bsz,T,H,HD).transpose(1,2);v=v.view(Bsz,T,H,HD).transpose(1,2);av=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(Bsz,T,D);x=x+self.lin(av,'o',l,self.o_bias[l]);z=F.layer_norm(x,(D,),self.n2w[l],self.n2b[l]);x=x+self.lin(F.gelu(self.lin(z,'fc1',l,self.fc1_bias[l])),'fc2',l,self.fc2_bias[l])
  return self.head(self.norm(x))

def batch(b=8,T=32):
 ix=torch.randint(0,len(TRAIN)-T-1,(b,));return torch.stack([TRAIN[i:i+T] for i in ix]),torch.stack([TRAIN[i+1:i+T+1] for i in ix])
def evaluate(m,data,max_chars=32768):
 n=min(max_chars//CTX,(len(data)-1)//CTX);tot=count=0;m.eval()
 with torch.inference_mode():
  for i in range(0,n,32):
   bs=min(32,n-i);x=torch.stack([data[(i+j)*CTX:(i+j+1)*CTX] for j in range(bs)]);y=torch.stack([data[(i+j)*CTX+1:(i+j+1)*CTX+1] for j in range(bs)]);z=m(x);tot+=float(F.cross_entropy(z.reshape(-1,len(CHARS)),y.reshape(-1),reduction='sum'));count+=y.numel()
 return tot/count,count
def train_teacher(t,n=120):
 opt=torch.optim.AdamW(t.parameters(),lr=2e-3);t.train()
 for _ in range(n):
  x,y=batch();loss=F.cross_entropy(t(x).reshape(-1,len(CHARS)),y.reshape(-1));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(t.parameters(),1);opt.step()
def recover(s,t,n=80):
 opt=torch.optim.AdamW(s.parameters(),lr=8e-4);t.eval();s.train()
 for _ in range(n):
  x,y=batch();
  with torch.no_grad():tl=t(x)
  sl=s(x);ce=F.cross_entropy(sl.reshape(-1,len(CHARS)),y.reshape(-1));kl=F.kl_div(F.log_softmax(sl/1.5,-1),F.softmax(tl/1.5,-1),reduction='batchmean')*(1.5**2)/sl.shape[1];loss=.55*ce+.45*kl;opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(s.parameters(),1);opt.step()

def row_q4dq(x):
 x=x.detach().float();sh=x.shape;y=x.reshape(-1,sh[-1]);pos=y.amax(1).clamp_min(0)/7.;neg=(-y.amin(1)).clamp_min(0)/8.;sc=torch.maximum(pos,neg).clamp_min(1e-8).half().float();return (torch.round(y/sc[:,None]).clamp(-8,7)*sc[:,None]).reshape(sh)
def group_q4dq(x,g=128):
 y=x.detach().float().reshape(-1);out=torch.empty_like(y)
 for i in range(0,y.numel(),g):
  z=y[i:i+g];pos=z.max().clamp_min(0)/7.;neg=(-z.min()).clamp_min(0)/8.;sc=torch.maximum(pos,neg).clamp_min(1e-8).half().float();out[i:i+g]=torch.round(z/sc).clamp(-8,7)*sc
 return out.reshape_as(x)
def project_teacher_q4(m):
 with torch.no_grad():
  for p in m.parameters():p.copy_(row_q4dq(p) if p.ndim>=2 else p.half().float())
def project_soft_q4(m):
 with torch.no_grad():
  seen=set()
  for name,p in m.named_parameters():
   if p.data_ptr() in seen:continue
   seen.add(p.data_ptr());p.copy_(p.half().float() if p.ndim<2 else group_q4dq(p,128) if name.startswith(('A_','B_')) else row_q4dq(p))
def qrecover(s,t,n=50):
 opt=torch.optim.AdamW(s.parameters(),lr=2e-4);t.eval();s.train()
 for _ in range(n):
  x,y=batch();
  with torch.no_grad():tl=t(x)
  sl=s(x);ce=F.cross_entropy(sl.reshape(-1,len(CHARS)),y.reshape(-1));kl=F.kl_div(F.log_softmax(sl/1.5,-1),F.softmax(tl/1.5,-1),reduction='batchmean')*(1.5**2)/sl.shape[1];loss=.5*ce+.5*kl;opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(s.parameters(),1);opt.step();project_soft_q4(s)

def q4row_bytes(m,n):return m*((n+1)//2+2)
def groupq4_bytes(n,g=128):return math.ceil(n/2)+math.ceil(n/g)*2
def teacher_bytes():
 V=len(CHARS);total=q4row_bytes(V,D)+q4row_bytes(CTX,D)+2*D*2;per=q4row_bytes(3*D,D)+q4row_bytes(D,D)+q4row_bytes(FF,D)+q4row_bytes(D,FF)+(4*D+3*D+D+FF+D)*2;return total+L*per
def soft_bytes(ranks):
 V=len(CHARS);mats={'qkv':(3*D,D),'o':(D,D),'fc1':(FF,D),'fc2':(D,FF)};total=q4row_bytes(V,D)+q4row_bytes(CTX,D)+2*D*2+sum(q4row_bytes(m,n) for m,n in mats.values())
 for name,(m,n) in mats.items():r=ranks[name];total+=groupq4_bytes(L*m*r)+groupq4_bytes(L*r*n)
 for width in (D,D,D,D,3*D,D,FF,D):total+=q4row_bytes(L,width)
 return total

def main():
 torch.manual_seed(3);random.seed(3);t=Teacher();train_teacher(t);rng_t=torch.get_rng_state();rng_p=random.getstate();ev=toks(333);tf,N=evaluate(t,ev);tq=copy.deepcopy(t);project_teacher_q4(tq);tqn,_=evaluate(tq,ev);profiles=[]
 configs=[('scale_normalized_rank3',{'qkv':3,'o':3,'fc1':3,'fc2':3}),('uniform_rank2_exact_toy_10x',{'qkv':2,'o':2,'fc1':2,'fc2':2}),('adaptive_exact_toy_10x',{'qkv':2,'o':1,'fc1':3,'fc2':2})]
 for name,ranks in configs:
  torch.set_rng_state(rng_t);random.setstate(rng_p);s=SoftShare(t,ranks);raw,_=evaluate(s,ev);recover(s,t);rf,_=evaluate(s,ev);sq=copy.deepcopy(s);project_soft_q4(sq);pre,_=evaluate(sq,ev);qrecover(sq,tq);fin,_=evaluate(sq,ev);b=soft_bytes(ranks);profiles.append({'name':name,'ranks':ranks,'raw_svd_nll':raw,'recovered_fp_nll':rf,'q4_before_constrained_recovery_nll':pre,'q4_after_50_constrained_steps_nll':fin,'delta_vs_q4_teacher_nats_per_char':fin-tqn,'perplexity_ratio_vs_q4_teacher':math.exp(fin-tqn),'serialized_bytes':b,'whole_model_reduction_x':teacher_bytes()/b})
 out={'run':5,'evidence_level':'controlled L2C mechanism study','task':'synthetic character-level template LM','model':{'hidden':D,'heads':H,'ffn':FF,'layers':L,'context':CTX,'vocab':len(CHARS)},'training_seed':3,'evaluation_seed':333,'evaluation_chars':N,'teacher':{'fp32_nll':tf,'canonical_q4_nll':tqn,'canonical_q4_serialized_bytes':teacher_bytes()},'profiles':profiles,'claim_boundary':'Controlled mechanism evidence only; not a real-model 10x result.'};text=json.dumps(out,indent=2)+'\n';print(text,end='');Path('benchmarks/run5_softshare_control.json').write_text(text)
if __name__=='__main__':main()
