#!/usr/bin/env python3
"""Canonical controlled Run-5 reference protocol.

CPU-heavy five-seed experiment: train independent 16-block teachers, function-fit
one shared block to all teacher layer roles, hard-project group-64 Q4 during LM
recovery, then evaluate group-3 latent-Q2 KV against the project row-Q4 teacher.
"""
from __future__ import annotations
import copy,json,math,os,random,statistics
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(min(2,os.cpu_count() or 1));torch.use_deterministic_algorithms(True)
NAMES=['Mia','Leo','Ava','Noah','Lily','Ben','Ella','Sam','Ivy','Max'];PLACES=['garden','forest','school','beach','farm','park','shop'];THINGS=['red ball','small key','blue kite','old book','silver coin','green box','toy boat'];ACTIONS=['found','lost','carried','opened','shared','fixed','painted']
def corpus(n,seed):
 r=random.Random(seed);o=[]
 for _ in range(n):
  a,b=r.sample(NAMES,2);p=r.choice(PLACES);t=r.choice(THINGS);v=r.choice(ACTIONS);o.append(f'Once upon a time, {a} went to the {p}. {a} {v} a {t}. Then {b} came to help. They were kind and went home happy.\n')
 return ''.join(o)
text=corpus(6000,3);chars=sorted(set(text));stoi={c:i for i,c in enumerate(chars)};sp=int(len(text)*.9);train=torch.tensor([stoi[c] for c in text[:sp]]);ev=torch.tensor([stoi[c] for c in corpus(1000,999)]);cal=torch.tensor([stoi[c] for c in corpus(200,777)])
D=128;H=4;HD=32;FF=256;DEPTH=16;CTX=64;RANK=16;VOCAB=len(chars)
class Block(nn.Module):
 def __init__(self):super().__init__();self.n1=nn.LayerNorm(D);self.attn=nn.MultiheadAttention(D,H,batch_first=True);self.n2=nn.LayerNorm(D);self.fc1=nn.Linear(D,FF);self.fc2=nn.Linear(FF,D)
 def forward(self,x,mask):z=self.n1(x);a,_=self.attn(z,z,z,attn_mask=mask,need_weights=False);x=x+a;z=self.n2(x);return x+self.fc2(F.gelu(self.fc1(z)))
class Teacher(nn.Module):
 def __init__(self):super().__init__();self.emb=nn.Embedding(VOCAB,D);self.pos=nn.Embedding(CTX,D);self.blocks=nn.ModuleList([Block() for _ in range(DEPTH)]);self.norm=nn.LayerNorm(D);self.head=nn.Linear(D,VOCAB,bias=False);self.head.weight=self.emb.weight
 def forward(self,idx):
  T=idx.shape[1];x=self.emb(idx)+self.pos(torch.arange(T));mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1)
  for b in self.blocks:x=b(x,mask)
  return self.head(self.norm(x))
class Converted(nn.Module):
 def __init__(self,t):
  super().__init__();self.emb=copy.deepcopy(t.emb);self.pos=copy.deepcopy(t.pos);self.block=Block();self.norm=copy.deepcopy(t.norm);self.head=nn.Linear(D,VOCAB,bias=False);self.head.weight=self.emb.weight
  self.block.load_state_dict({k:torch.stack([b.state_dict()[k].float() for b in t.blocks]).mean(0).to(v.dtype) for k,v in self.block.state_dict().items()})
 def forward(self,idx):
  T=idx.shape[1];x=self.emb(idx)+self.pos(torch.arange(T));mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1)
  for _ in range(DEPTH):x=self.block(x,mask)
  return self.head(self.norm(x))
def batch(b=16,T=32):
 ix=torch.randint(0,len(train)-T-1,(b,));return torch.stack([train[i:i+T] for i in ix]),torch.stack([train[i+1:i+T+1] for i in ix])
def train_teacher(seed):
 torch.manual_seed(seed);m=Teacher();opt=torch.optim.AdamW(m.parameters(),lr=2e-3);m.train()
 for _ in range(120):
  x,y=batch();l=F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1));opt.zero_grad();l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 return m.eval()
def block_q4(w,group=64):
 x=w.detach().float();out=torch.empty_like(x)
 for s in range(0,x.shape[1],group):
  a=x[:,s:s+group];pos=a.amax(1).clamp_min(0)/7.;neg=(-a.amin(1)).clamp_min(0)/8.;sc=torch.maximum(pos,neg).clamp_min(1e-8).half().float();out[:,s:s+group]=torch.round(a/sc[:,None]).clamp(-8,7)*sc[:,None]
 return out
def row_q4(w):
 x=w.detach().float();pos=x.amax(1).clamp_min(0)/7.;neg=(-x.amin(1)).clamp_min(0)/8.;sc=torch.maximum(pos,neg).clamp_min(1e-8).half().float();return torch.round(x/sc[:,None]).clamp(-8,7)*sc[:,None]
def project_q4_(m,group=64):
 with torch.no_grad():
  seen=set()
  for p in m.parameters():
   ptr=p.untyped_storage().data_ptr()
   if ptr in seen:continue
   seen.add(ptr);p.copy_(block_q4(p,group) if p.ndim==2 else p.half().float())
def quantized_copy(m,row=False):
 q=copy.deepcopy(m)
 with torch.no_grad():
  seen=set()
  for p in q.parameters():
   ptr=p.untyped_storage().data_ptr()
   if ptr in seen:continue
   seen.add(ptr);p.copy_((row_q4(p) if row else block_q4(p,64)) if p.ndim==2 else p.half().float())
 return q.eval()
def convert_prefit_qat(t,seed):
 torch.manual_seed(seed+4000);m=Converted(t);opt=torch.optim.AdamW(m.block.parameters(),lr=1e-3);t.eval();m.block.train()
 for _ in range(80):
  ids,_=batch(b=8,T=32);T=ids.shape[1];mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1)
  with torch.no_grad():
   x=t.emb(ids)+t.pos(torch.arange(T));ins=[];outs=[]
   for b in t.blocks:ins.append(x);x=b(x,mask);outs.append(x)
  pred=m.block(torch.cat(ins,0),mask);loss=F.mse_loss(pred,torch.cat(outs,0));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.block.parameters(),1);opt.step()
 project_q4_(m,64);opt=torch.optim.AdamW(m.parameters(),lr=1.5e-3);m.train()
 for _ in range(200):
  ids,y=batch();loss=F.cross_entropy(m(ids).reshape(-1,VOCAB),y.reshape(-1));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();project_q4_(m,64)
 return m.eval()
def evaluate(m,data=ev):
 usable=min(100032,((len(data)-1)//CTX)*CTX);starts=list(range(0,usable,CTX));total=0.;count=0;m.eval()
 with torch.inference_mode():
  for j in range(0,len(starts),32):
   ss=starts[j:j+32];x=torch.stack([data[s:s+CTX] for s in ss]);y=torch.stack([data[s+1:s+CTX+1] for s in ss]);total+=float(F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1),reduction='sum'));count+=y.numel()
 return total/count,count
def q4_basis(b):return block_q4(b.reshape(-1,b.shape[-1]),64).reshape_as(b)
def fit_basis(x):_,_,v=torch.pca_lowrank(x.float(),q=min(RANK+4,min(x.shape)),center=False,niter=3);return v[:,:RANK].T.contiguous()
def invgram(B):
 G=B@B.transpose(-1,-2);diag=torch.diagonal(G,dim1=-2,dim2=-1).mean(-1).clamp_min(1e-8);I=torch.eye(RANK).expand_as(G);return torch.linalg.inv(G+I*(diag[:,None,None]*1e-5))
def collect(m,seqs):
 B,T=seqs.shape;CK=[[] for _ in range(DEPTH)];CV=[[] for _ in range(DEPTH)];KC=[[] for _ in range(H)];VC=[[] for _ in range(H)];b=m.block
 with torch.inference_mode():
  for ti in range(T):
   x=m.emb(seqs[:,ti])+m.pos(torch.tensor([ti])).squeeze(0)
   for d in range(DEPTH):
    z=b.n1(x);q,k,v=F.linear(z,b.attn.in_proj_weight,b.attn.in_proj_bias).chunk(3,-1);q=q.view(B,H,HD);k=k.view(B,H,HD);v=v.view(B,H,HD);CK[d].append(k);CV[d].append(v)
    for h in range(H):KC[h].append(k[:,h]);VC[h].append(v[:,h])
    K=torch.stack(CK[d],2);V=torch.stack(CV[d],2);a=torch.softmax(torch.einsum('bhd,bhtd->bht',q,K)/math.sqrt(HD),-1);av=torch.einsum('bht,bhtd->bhd',a,V).reshape(B,D);x=x+F.linear(av,b.attn.out_proj.weight,b.attn.out_proj.bias);x=x+b.fc2(F.gelu(b.fc1(b.n2(x))))
 return KC,VC
def groupdq(a):
 mn=a.amin(dim=(-2,-1),keepdim=True);mx=a.amax(dim=(-2,-1),keepdim=True);sc=((mx-mn)/3).clamp_min(1e-8);return torch.round((a-mn)/sc).clamp(0,3)*sc.half().float()+mn.half().float()
def compressed_logits(m,seqs,KB,VB,KM,VM,group=3):
 B,T=seqs.shape;b=m.block;outs=[];KD=[[] for _ in range(DEPTH)];VD=[[] for _ in range(DEPTH)];KT=[[] for _ in range(DEPTH)];VT=[[] for _ in range(DEPTH)]
 with torch.inference_mode():
  for ti in range(T):
   x=m.emb(seqs[:,ti])+m.pos(torch.tensor([ti])).squeeze(0)
   for d in range(DEPTH):
    z=b.n1(x);q,k,v=F.linear(z,b.attn.in_proj_weight,b.attn.in_proj_bias).chunk(3,-1);q=q.view(B,H,HD);k=k.view(B,H,HD);v=v.view(B,H,HD);KT[d].append(torch.einsum('bhd,hrd->bhr',k,KB));VT[d].append(torch.einsum('bhd,hrd->bhr',v,VB));kd=groupdq(torch.stack(KT[d],2));vd=groupdq(torch.stack(VT[d],2));Kh=torch.cat(KD[d]+[kd],2) if KD[d] else kd;Vh=torch.cat(VD[d]+[vd],2) if VD[d] else vd
    if len(KT[d])==group:KD[d].append(kd);VD[d].append(vd);KT[d]=[];VT[d]=[]
    ql=torch.einsum('bhd,hrd->bhr',q,KB);qc=torch.einsum('bhr,hrs->bhs',ql,KM);a=torch.softmax(torch.einsum('bhtr,bhr->bht',Kh,qc)/math.sqrt(HD),-1);vl=torch.einsum('bht,bhtr->bhr',a,Vh);vc=torch.einsum('bhr,hrs->bhs',vl,VM);av=torch.einsum('bhr,hrd->bhd',vc,VB).reshape(B,D);x=x+F.linear(av,b.attn.out_proj.weight,b.attn.out_proj.bias);x=x+b.fc2(F.gelu(b.fc1(b.n2(x))))
   outs.append(m.head(m.norm(x)))
 return torch.stack(outs,1)
def evaluate_larc(m):
 cs=torch.stack([cal[i*CTX:i*CTX+CTX] for i in range(16)]);KC,VC=collect(m,cs);KB=q4_basis(torch.stack([fit_basis(torch.cat(KC[h],0)) for h in range(H)]));VB=q4_basis(torch.stack([fit_basis(torch.cat(VC[h],0)) for h in range(H)]));KM=invgram(KB);VM=invgram(VB);usable=min(100032,((len(ev)-1)//CTX)*CTX);starts=list(range(0,usable,CTX));total=0.;count=0
 for j in range(0,len(starts),32):
  ss=starts[j:j+32];x=torch.stack([ev[s:s+CTX] for s in ss]);y=torch.stack([ev[s+1:s+CTX+1] for s in ss]);z=compressed_logits(m,x,KB,VB,KM,VM,3);total+=float(F.cross_entropy(z.reshape(-1,VOCAB),y.reshape(-1),reduction='sum'));count+=y.numel()
 return total/count,count
def main():
 rows=[]
 for seed in [3,7,11,19,23]:
  t=train_teacher(seed);baseline=quantized_copy(t,row=True);m=convert_prefit_qat(t,seed);bn,n=evaluate(baseline);fn,_=evaluate_larc(m);fp,_=evaluate(t);rows.append({'seed':seed,'baseline_row_q4_nll':bn,'larc_full_nll':fn,'delta_nats_per_char_vs_row_q4':fn-bn,'perplexity_ratio_vs_row_q4':math.exp(fn-bn),'fp32_teacher_nll':fp,'delta_nats_per_char_vs_fp32':fn-fp,'perplexity_ratio_vs_fp32':math.exp(fn-fp)});print('seed',seed,rows[-1],flush=True)
 d=[x['delta_nats_per_char_vs_row_q4'] for x in rows];p=[x['perplexity_ratio_vs_row_q4'] for x in rows];df=[x['delta_nats_per_char_vs_fp32'] for x in rows];pf=[x['perplexity_ratio_vs_fp32'] for x in rows];out={'evidence_level':'L2C controlled multi-seed','training_corpus_seed':3,'evaluation_stream_seed':999,'calibration_stream_seed':777,'evaluation_chars_per_seed':100032,'seeds':rows,'statistics':{'delta_nats_per_char_vs_row_q4_mean':statistics.mean(d),'delta_nats_per_char_vs_row_q4_sample_std':statistics.stdev(d),'perplexity_ratio_vs_row_q4_mean':statistics.mean(p),'perplexity_ratio_vs_row_q4_sample_std':statistics.stdev(p),'delta_nats_per_char_vs_fp32_mean':statistics.mean(df),'perplexity_ratio_vs_fp32_mean':statistics.mean(pf)}};Path('benchmarks/run5_fullstack_multiseed.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
