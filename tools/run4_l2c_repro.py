#!/usr/bin/env python3
"""Reproduce the authoritative Run-4 controlled L2C experiment.

Phases intentionally run with reset RNG state and serialized checkpoints so a
long run can be resumed without changing random-number consumption.

  python tools/run4_l2c_repro.py --workspace /tmp/larc-r4 --phase teacher
  python tools/run4_l2c_repro.py --workspace /tmp/larc-r4 --phase convert
  python tools/run4_l2c_repro.py --workspace /tmp/larc-r4 --phase q4-recover
  python tools/run4_l2c_repro.py --workspace /tmp/larc-r4 --phase evaluate

`--phase all` runs all four in order. This is a controlled synthetic
character-LM experiment, not an external pretrained-model benchmark.
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
D=128;H=4;HD=32;FF=256;DEPTH=16;CTX=64;RANK=16;VOCAB_EXPECTED=37

def reset_seed(): random.seed(3);torch.manual_seed(3)
def corpus(n:int,seed:int)->str:
 r=random.Random(seed);o=[]
 for _ in range(n):
  a,b=r.sample(NAMES,2);p=r.choice(PLACES);t=r.choice(THINGS);v=r.choice(ACTIONS);o.append(f'Once upon a time, {a} went to the {p}. {a} {v} a {t}. Then {b} came to help. They were kind and went home happy.\n')
 return ''.join(o)
TRAIN_TEXT=corpus(6000,3);CHARS=sorted(set(TRAIN_TEXT));STOI={c:i for i,c in enumerate(CHARS)};TRAIN=torch.tensor([STOI[c] for c in TRAIN_TEXT],dtype=torch.long)
assert len(CHARS)==VOCAB_EXPECTED

def tokens(seed:int,n=1200):return torch.tensor([STOI[c] for c in corpus(n,seed)],dtype=torch.long)

class Block(nn.Module):
 def __init__(self):
  super().__init__();self.n1=nn.LayerNorm(D);self.attn=nn.MultiheadAttention(D,H,batch_first=True);self.n2=nn.LayerNorm(D);self.fc1=nn.Linear(D,FF);self.fc2=nn.Linear(FF,D)
 def forward(self,x,mask):
  z=self.n1(x);a,_=self.attn(z,z,z,attn_mask=mask,need_weights=False);x=x+a;z=self.n2(x);return x+self.fc2(F.gelu(self.fc1(z)))
class Teacher(nn.Module):
 def __init__(self):
  super().__init__();self.emb=nn.Embedding(len(CHARS),D);self.pos=nn.Embedding(CTX,D);self.blocks=nn.ModuleList([Block() for _ in range(DEPTH)]);self.norm=nn.LayerNorm(D);self.head=nn.Linear(D,len(CHARS),bias=False);self.head.weight=self.emb.weight
 def forward(self,idx):
  T=idx.shape[1];x=self.emb(idx)+self.pos(torch.arange(T,device=idx.device));mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=idx.device),1)
  for b in self.blocks:x=b(x,mask)
  return self.head(self.norm(x))
class Converted(nn.Module):
 def __init__(self,t:Teacher):
  super().__init__();self.emb=copy.deepcopy(t.emb);self.pos=copy.deepcopy(t.pos);self.block=Block();self.norm=copy.deepcopy(t.norm);self.head=nn.Linear(D,len(CHARS),bias=False);self.head.weight=self.emb.weight
  sd=self.block.state_dict();avg={k:torch.stack([b.state_dict()[k].float() for b in t.blocks]).mean(0).to(v.dtype) for k,v in sd.items()};self.block.load_state_dict(avg)
 def forward(self,idx):
  T=idx.shape[1];x=self.emb(idx)+self.pos(torch.arange(T,device=idx.device));mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=idx.device),1)
  for _ in range(DEPTH):x=self.block(x,mask)
  return self.head(self.norm(x))

def sample_batch(b=16,T=32):
 ix=torch.randint(0,len(TRAIN)-T-1,(b,));return torch.stack([TRAIN[i:i+T] for i in ix]),torch.stack([TRAIN[i+1:i+T+1] for i in ix])
def train_steps(m,opt,steps,project=None):
 m.train()
 for _ in range(steps):
  x,y=sample_batch();loss=F.cross_entropy(m(x).reshape(-1,len(CHARS)),y.reshape(-1));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
  if project is not None:project(m)
def eval_tokens(m,data,max_chars=100032,batch_ctx=32):
 n=min(max_chars//CTX,(len(data)-1)//CTX);total=0.;count=0;m.eval()
 with torch.inference_mode():
  for i in range(0,n,batch_ctx):
   bs=min(batch_ctx,n-i);x=torch.stack([data[(i+j)*CTX:(i+j+1)*CTX] for j in range(bs)]);y=torch.stack([data[(i+j)*CTX+1:(i+j+1)*CTX+1] for j in range(bs)]);log=m(x);total+=float(F.cross_entropy(log.reshape(-1,len(CHARS)),y.reshape(-1),reduction='sum'));count+=y.numel()
 return total/count,count

def project_q4(m):
 with torch.no_grad():
  for p in m.parameters():
   if p.ndim==2:
    pk,s,c=q4_rows(p);p.copy_(dequantize_q4_rows(pk,s,c))
   else:p.copy_(p.half().float())
def q4_clone(m):q=copy.deepcopy(m).eval();project_q4(q);return q

def q2_fp8_roundtrip(x):
 shape=x.shape;p,mn,sc,n=pack_q2_rows_fp8meta(x.reshape(-1,shape[-1]));return unpack_q2_rows_fp8meta(p,mn,sc,n).reshape(shape)

class Bases:
 def __init__(self,kb,vb):
  self.kq=quantize_head_basis_q4(kb,store_metric=True);self.vq=quantize_head_basis_q4(vb,store_metric=True);self.kb=self.kq.dequantized;self.vb=self.vq.dequantized
 @property
 def storage_bytes(self):return self.kq.storage_bytes+self.vq.storage_bytes

def collect_bases(m,cal,contexts=16):
 K=[[] for _ in range(H)];V=[[] for _ in range(H)];b=m.block;m.eval()
 with torch.inference_mode():
  for j in range(contexts):
   ids=cal[j*CTX:(j+1)*CTX][None,:];x=m.emb(ids)+m.pos(torch.arange(CTX));mask=torch.triu(torch.ones(CTX,CTX,dtype=torch.bool),1)
   for _ in range(DEPTH):
    z=b.n1(x);_,k,v=F.linear(z,b.attn.in_proj_weight,b.attn.in_proj_bias).chunk(3,-1);k=k.view(CTX,H,HD);v=v.view(CTX,H,HD)
    for h in range(H):K[h].append(k[:,h].clone());V[h].append(v[:,h].clone())
    a,_=b.attn(z,z,z,attn_mask=mask,need_weights=False);x=x+a;z=b.n2(x);x=x+b.fc2(F.gelu(b.fc1(z)))
 kb=torch.stack([fit_basis(torch.cat(K[h],0),RANK) for h in range(H)]);vb=torch.stack([fit_basis(torch.cat(V[h],0),RANK) for h in range(H)]);return Bases(kb,vb)

def compressed_forward(m,idx,bases:Bases):
 B,T=idx.shape;x=m.emb(idx)+m.pos(torch.arange(T));b=m.block;causal=torch.triu(torch.ones(T,T,dtype=torch.bool),1)
 for _ in range(DEPTH):
  z=b.n1(x);q,k,v=F.linear(z,b.attn.in_proj_weight,b.attn.in_proj_bias).chunk(3,-1);q=q.view(B,T,H,HD).permute(0,2,1,3);k=k.view(B,T,H,HD).permute(0,2,1,3);v=v.view(B,T,H,HD).permute(0,2,1,3)
  kl=q2_fp8_roundtrip(torch.einsum('bhtd,hrd->bhtr',k,bases.kb));vl=q2_fp8_roundtrip(torch.einsum('bhtd,hrd->bhtr',v,bases.vb));ql=torch.einsum('bhtd,hrd->bhtr',q,bases.kb);qm=torch.einsum('bhtr,hrs->bhts',ql,bases.kq.metric_inv);scores=torch.einsum('bhtr,bhsr->bhts',qm,kl)/math.sqrt(HD);scores=scores.masked_fill(causal[None,None],float('-inf'));a=torch.softmax(scores,-1);vlat=torch.einsum('bhts,bhsr->bhtr',a,vl);vc=torch.einsum('bhtr,hrs->bhts',vlat,bases.vq.metric_inv);av=torch.einsum('bhtr,hrd->bhtd',vc,bases.vb).permute(0,2,1,3).reshape(B,T,D);x=x+F.linear(av,b.attn.out_proj.weight,b.attn.out_proj.bias);z=b.n2(x);x=x+b.fc2(F.gelu(b.fc1(z)))
 return m.head(m.norm(x))
def eval_compressed(m,bases,data,max_chars=100032,batch_ctx=32):
 n=min(max_chars//CTX,(len(data)-1)//CTX);total=0.;count=0;m.eval()
 with torch.inference_mode():
  for i in range(0,n,batch_ctx):
   bs=min(batch_ctx,n-i);x=torch.stack([data[(i+j)*CTX:(i+j+1)*CTX] for j in range(bs)]);y=torch.stack([data[(i+j)*CTX+1:(i+j+1)*CTX+1] for j in range(bs)]);log=compressed_forward(m,x,bases);total+=float(F.cross_entropy(log.reshape(-1,len(CHARS)),y.reshape(-1),reduction='sum'));count+=y.numel()
 return total/count,count

def qb(p):return p.shape[0]*((p.shape[1]+1)//2)+2*p.shape[0] if p.ndim==2 else p.numel()*2
def weight_bytes(m):
 seen=set();total=0
 for p in m.parameters():
  ptr=p.untyped_storage().data_ptr()
  if ptr not in seen:seen.add(ptr);total+=qb(p)
 return total

def teacher_phase(ws):
 reset_seed();m=Teacher();opt=torch.optim.AdamW(m.parameters(),lr=2e-3);train_steps(m,opt,120);torch.save(m.state_dict(),ws/'teacher120.pt')
def convert_phase(ws):
 reset_seed();t=Teacher();t.load_state_dict(torch.load(ws/'teacher120.pt',weights_only=True));m=Converted(t);opt=torch.optim.AdamW(m.parameters(),lr=1.5e-3);train_steps(m,opt,200);torch.save(m.state_dict(),ws/'converted.pt')
def q4_recover_phase(ws):
 reset_seed();t=Teacher();t.load_state_dict(torch.load(ws/'teacher120.pt',weights_only=True));m=Converted(t);m.load_state_dict(torch.load(ws/'converted.pt',weights_only=True));project_q4(m);opt=torch.optim.AdamW(m.parameters(),lr=3e-4);selection=tokens(444,300);best=float('inf');best_sd=None;curve=[]
 for step in range(200):
  train_steps(m,opt,1,project_q4)
  if (step+1)%25==0:
   nll,_=eval_tokens(m,selection,8192);curve.append([step+1,nll]);
   if nll<best:best=nll;best_sd=copy.deepcopy(m.state_dict())
 m.load_state_dict(best_sd);torch.save(m.state_dict(),ws/'converted_q4_recovered.pt');(ws/'recovery_selection.json').write_text(json.dumps({'selection_seed':444,'curve':curve,'selected':min(curve,key=lambda x:x[1])},indent=2)+'\n')
def evaluate_phase(ws):
 reset_seed();t=Teacher();t.load_state_dict(torch.load(ws/'teacher120.pt',weights_only=True));teacher_q4=q4_clone(t);m=Converted(t);m.load_state_dict(torch.load(ws/'converted_q4_recovered.pt',weights_only=True));final=tokens(333);cal=tokens(555,300);tn,N=eval_tokens(teacher_q4,final);sn,_=eval_tokens(m,final);bases=collect_bases(m,cal);cn,N2=eval_compressed(m,bases,final)
 teacher_w=weight_bytes(t);shared_w=weight_bytes(m);basekv=DEPTH*CTX*H*HD*4;cache=DEPTH*CTX*H*2*(math.ceil(RANK*2/8)+2);base_scratch=(5*D+FF+H*CTX)*4;larc_scratch=base_scratch+4*RANK*4;base_total=teacher_w+basekv+base_scratch;larc_total=shared_w+cache+bases.storage_bytes+larc_scratch
 out={'streams':{'training_seed':3,'selection_seed':444,'calibration_seed':555,'final_eval_seed':333},'evaluation_chars':N2,'context':CTX,'teacher_q4_nll':tn,'q4_recovered_shared_nll':sn,'compressed_nll':cn,'structural_delta_nats':sn-tn,'kv_delta_nats':cn-sn,'total_delta_nats':cn-tn,'ppl_ratio':math.exp(cn-tn),'teacher_q4_weight_bytes':teacher_w,'shared_q4_weight_bytes':shared_w,'fp16_kv_bytes':basekv,'q2_fp8meta_cache_bytes':cache,'basis_and_metrics_bytes':bases.storage_bytes,'direct_packed_baseline_scratch_bytes':base_scratch,'direct_packed_larc_scratch_bytes':larc_scratch,'direct_packed_baseline_total_bytes':base_total,'direct_packed_larc_total_bytes':larc_total,'direct_packed_modeled_reduction_x':base_total/larc_total,'memory_measured':False}
 (ws/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,default=Path('.run4-repro'));ap.add_argument('--phase',choices=['teacher','convert','q4-recover','evaluate','all'],default='all');a=ap.parse_args();a.workspace.mkdir(parents=True,exist_ok=True)
 phases={'teacher':teacher_phase,'convert':convert_phase,'q4-recover':q4_recover_phase,'evaluate':evaluate_phase}
 if a.phase=='all':
  for name in ['teacher','convert','q4-recover','evaluate']:print('PHASE',name,flush=True);phases[name](a.workspace)
 else:phases[a.phase](a.workspace)
if __name__=='__main__':main()
