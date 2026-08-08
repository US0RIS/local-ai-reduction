#!/usr/bin/env python3
"""Checkpointed reproducer for the promoted Run-4 controlled L2C result.

Phases: teacher -> convert -> q4-recover -> evaluate. Training, checkpoint
selection, latent-basis calibration and final evaluation use disjoint streams.
"""
from __future__ import annotations
import argparse,copy,json,math,os,random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from larc.q4_runtime import q4_rows,dequantize_q4_rows
from larc.latent_kv import fit_basis,quantize_head_basis_q4,pack_q2_rows_fp8meta,unpack_q2_rows_fp8meta

torch.set_num_threads(min(4,os.cpu_count() or 2))
NAMES=['Mia','Leo','Ava','Noah','Lily','Ben','Ella','Sam','Ivy','Max'];PLACES=['garden','forest','school','beach','farm','park','shop'];THINGS=['red ball','small key','blue kite','old book','silver coin','green box','toy boat'];ACTIONS=['found','lost','carried','opened','shared','fixed','painted']
D,H,HD,FF,DEPTH,CTX,RANK=128,4,32,256,16,64,16

def corpus(n,seed):
 r=random.Random(seed);out=[]
 for _ in range(n):
  a,b=r.sample(NAMES,2);p=r.choice(PLACES);t=r.choice(THINGS);v=r.choice(ACTIONS);out.append(f'Once upon a time, {a} went to the {p}. {a} {v} a {t}. Then {b} came to help. They were kind and went home happy.\n')
 return ''.join(out)
TRAIN_TEXT=corpus(6000,3);CHARS=sorted(set(TRAIN_TEXT));STOI={c:i for i,c in enumerate(CHARS)};TRAIN=torch.tensor([STOI[c] for c in TRAIN_TEXT],dtype=torch.long)
def toks(seed,n=1200):return torch.tensor([STOI[c] for c in corpus(n,seed)],dtype=torch.long)
def reset():random.seed(3);torch.manual_seed(3)

class Block(nn.Module):
 def __init__(self):super().__init__();self.n1=nn.LayerNorm(D);self.attn=nn.MultiheadAttention(D,H,batch_first=True);self.n2=nn.LayerNorm(D);self.fc1=nn.Linear(D,FF);self.fc2=nn.Linear(FF,D)
 def forward(self,x,mask):z=self.n1(x);a,_=self.attn(z,z,z,attn_mask=mask,need_weights=False);x=x+a;z=self.n2(x);return x+self.fc2(F.gelu(self.fc1(z)))
class Teacher(nn.Module):
 def __init__(self):super().__init__();self.emb=nn.Embedding(len(CHARS),D);self.pos=nn.Embedding(CTX,D);self.blocks=nn.ModuleList([Block() for _ in range(DEPTH)]);self.norm=nn.LayerNorm(D);self.head=nn.Linear(D,len(CHARS),bias=False);self.head.weight=self.emb.weight
 def forward(self,idx):
  T=idx.shape[1];x=self.emb(idx)+self.pos(torch.arange(T));mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1)
  for b in self.blocks:x=b(x,mask)
  return self.head(self.norm(x))
class Converted(nn.Module):
 def __init__(self,t):
  super().__init__();self.emb=copy.deepcopy(t.emb);self.pos=copy.deepcopy(t.pos);self.block=Block();self.norm=copy.deepcopy(t.norm);self.head=nn.Linear(D,len(CHARS),bias=False);self.head.weight=self.emb.weight
  self.block.load_state_dict({k:torch.stack([b.state_dict()[k].float() for b in t.blocks]).mean(0).to(v.dtype) for k,v in self.block.state_dict().items()})
 def forward(self,idx):
  T=idx.shape[1];x=self.emb(idx)+self.pos(torch.arange(T));mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1)
  for _ in range(DEPTH):x=self.block(x,mask)
  return self.head(self.norm(x))

def batch(b=16,T=32):
 ix=torch.randint(0,len(TRAIN)-T-1,(b,));return torch.stack([TRAIN[i:i+T] for i in ix]),torch.stack([TRAIN[i+1:i+T+1] for i in ix])
def steps(m,opt,n,project=None):
 m.train()
 for _ in range(n):
  x,y=batch();loss=F.cross_entropy(m(x).reshape(-1,len(CHARS)),y.reshape(-1));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();
  if project:project(m)
def eval_model(m,data,max_chars=100032):
 n=min(max_chars//CTX,(len(data)-1)//CTX);tot=count=0;m.eval()
 with torch.inference_mode():
  for i in range(0,n,32):
   bs=min(32,n-i);x=torch.stack([data[(i+j)*CTX:(i+j+1)*CTX] for j in range(bs)]);y=torch.stack([data[(i+j)*CTX+1:(i+j+1)*CTX+1] for j in range(bs)]);z=m(x);tot+=float(F.cross_entropy(z.reshape(-1,len(CHARS)),y.reshape(-1),reduction='sum'));count+=y.numel()
 return tot/count,count
def project_q4(m):
 with torch.no_grad():
  for p in m.parameters():
   if p.ndim==2:
    pk,s,c=q4_rows(p);p.copy_(dequantize_q4_rows(pk,s,c))
   else:p.copy_(p.half().float())
def q4clone(m):q=copy.deepcopy(m);project_q4(q);return q

def collect_bases(m,cal,contexts=16):
 K=[[] for _ in range(H)];V=[[] for _ in range(H)];b=m.block;m.eval()
 with torch.inference_mode():
  for j in range(contexts):
   ids=cal[j*CTX:(j+1)*CTX][None,:];x=m.emb(ids)+m.pos(torch.arange(CTX));mask=torch.triu(torch.ones(CTX,CTX,dtype=torch.bool),1)
   for _ in range(DEPTH):
    z=b.n1(x);_,k,v=F.linear(z,b.attn.in_proj_weight,b.attn.in_proj_bias).chunk(3,-1);k=k.view(CTX,H,HD);v=v.view(CTX,H,HD)
    for h in range(H):K[h].append(k[:,h]);V[h].append(v[:,h])
    a,_=b.attn(z,z,z,attn_mask=mask,need_weights=False);x=x+a;z=b.n2(x);x=x+b.fc2(F.gelu(b.fc1(z)))
 kb=torch.stack([fit_basis(torch.cat(K[h],0),RANK) for h in range(H)]);vb=torch.stack([fit_basis(torch.cat(V[h],0),RANK) for h in range(H)]);return quantize_head_basis_q4(kb),quantize_head_basis_q4(vb)
def rt(x):
 sh=x.shape;p,mn,sc,c=pack_q2_rows_fp8meta(x.reshape(-1,sh[-1]));return unpack_q2_rows_fp8meta(p,mn,sc,c).reshape(sh)
def compressed_forward(m,idx,kq,vq):
 B,T=idx.shape;b=m.block;x=m.emb(idx)+m.pos(torch.arange(T));mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1)
 for _ in range(DEPTH):
  z=b.n1(x);q,k,v=F.linear(z,b.attn.in_proj_weight,b.attn.in_proj_bias).chunk(3,-1);q=q.view(B,T,H,HD).permute(0,2,1,3);k=k.view(B,T,H,HD).permute(0,2,1,3);v=v.view(B,T,H,HD).permute(0,2,1,3)
  kl=rt(torch.einsum('bhtd,hrd->bhtr',k,kq.dequantized));vl=rt(torch.einsum('bhtd,hrd->bhtr',v,vq.dequantized));ql=torch.einsum('bhtd,hrd->bhtr',q,kq.dequantized);qm=torch.einsum('bhtr,hrs->bhts',ql,kq.gram_inv);s=torch.einsum('bhtr,bhsr->bhts',qm,kl)/math.sqrt(HD);s=s.masked_fill(mask[None,None],float('-inf'));a=torch.softmax(s,-1);va=torch.einsum('bhts,bhsr->bhtr',a,vl);vc=torch.einsum('bhtr,hrs->bhts',va,vq.gram_inv);av=torch.einsum('bhtr,hrd->bhtd',vc,vq.dequantized).permute(0,2,1,3).reshape(B,T,D);x=x+F.linear(av,b.attn.out_proj.weight,b.attn.out_proj.bias);z=b.n2(x);x=x+b.fc2(F.gelu(b.fc1(z)))
 return m.head(m.norm(x))
def eval_compressed(m,kq,vq,data,max_chars=100032):
 n=min(max_chars//CTX,(len(data)-1)//CTX);tot=count=0;m.eval()
 with torch.inference_mode():
  for i in range(0,n,32):
   bs=min(32,n-i);x=torch.stack([data[(i+j)*CTX:(i+j+1)*CTX] for j in range(bs)]);y=torch.stack([data[(i+j)*CTX+1:(i+j+1)*CTX+1] for j in range(bs)]);z=compressed_forward(m,x,kq,vq);tot+=float(F.cross_entropy(z.reshape(-1,len(CHARS)),y.reshape(-1),reduction='sum'));count+=y.numel()
 return tot/count,count

def weight_bytes(m):
 seen=set();total=0
 for p in m.parameters():
  ptr=p.untyped_storage().data_ptr()
  if ptr in seen:continue
  seen.add(ptr);total+=p.shape[0]*((p.shape[1]+1)//2+2) if p.ndim==2 else p.numel()*2
 return total

def teacher(ws):reset();m=Teacher();steps(m,torch.optim.AdamW(m.parameters(),lr=2e-3),120);torch.save(m.state_dict(),ws/'teacher.pt')
def convert(ws):reset();t=Teacher();t.load_state_dict(torch.load(ws/'teacher.pt',weights_only=True));m=Converted(t);steps(m,torch.optim.AdamW(m.parameters(),lr=1.5e-3),200);torch.save(m.state_dict(),ws/'converted.pt')
def recover(ws):
 reset();t=Teacher();t.load_state_dict(torch.load(ws/'teacher.pt',weights_only=True));m=Converted(t);m.load_state_dict(torch.load(ws/'converted.pt',weights_only=True));project_q4(m);opt=torch.optim.AdamW(m.parameters(),lr=3e-4);sel=toks(444,300);best=1e9;best_sd=None;curve=[]
 for i in range(200):
  steps(m,opt,1,project_q4)
  if (i+1)%25==0:
   n,_=eval_model(m,sel,8192);curve.append([i+1,n]);
   if n<best:best=n;best_sd=copy.deepcopy(m.state_dict())
 m.load_state_dict(best_sd);torch.save(m.state_dict(),ws/'recovered.pt');(ws/'selection.json').write_text(json.dumps(curve,indent=2))
def evaluate(ws):
 reset();t=Teacher();t.load_state_dict(torch.load(ws/'teacher.pt',weights_only=True));tq=q4clone(t);m=Converted(t);m.load_state_dict(torch.load(ws/'recovered.pt',weights_only=True));ev=toks(333);cal=toks(555,300);tn,N=eval_model(tq,ev);sn,_=eval_model(m,ev);kq,vq=collect_bases(m,cal);cn,_=eval_compressed(m,kq,vq,ev);tw=weight_bytes(t);sw=weight_bytes(m);basekv=DEPTH*CTX*H*HD*4;cache=DEPTH*CTX*H*2*(RANK//4+2);basis=kq.storage_bytes+vq.storage_bytes;base_s=(5*D+FF+H*CTX)*4;larc_s=base_s+4*RANK*4;base=tw+basekv+base_s;larc=sw+cache+basis+larc_s;out={'streams':{'training':3,'selection':444,'calibration':555,'evaluation':333},'evaluation_chars':N,'context':CTX,'teacher_q4_nll':tn,'q4_recovered_shared_nll':sn,'compressed_nll':cn,'total_delta_nats_per_char':cn-tn,'perplexity_ratio':math.exp(cn-tn),'teacher_q4_weight_bytes':tw,'shared_q4_weight_bytes':sw,'fp16_kv_bytes':basekv,'q2_fp8meta_cache_bytes':cache,'basis_and_metrics_bytes':basis,'direct_packed_baseline_scratch_bytes':base_s,'direct_packed_larc_scratch_bytes':larc_s,'direct_packed_baseline_total_bytes':base,'direct_packed_larc_total_bytes':larc,'direct_packed_modeled_reduction_x':base/larc,'memory_measured':False};(ws/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,default=Path('.run4-l2c'));ap.add_argument('--phase',choices=['teacher','convert','recover','evaluate','all'],default='all');a=ap.parse_args();a.workspace.mkdir(parents=True,exist_ok=True);f={'teacher':teacher,'convert':convert,'recover':recover,'evaluate':evaluate}
 if a.phase=='all':
  for p in ('teacher','convert','recover','evaluate'):print('PHASE',p,flush=True);f[p](a.workspace)
 else:f[a.phase](a.workspace)
if __name__=='__main__':main()
