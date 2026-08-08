#!/usr/bin/env python3
from __future__ import annotations
import copy, json, os, random, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from larc.q4_runtime import q4_rows

random.seed(0); torch.manual_seed(0); torch.set_num_threads(min(4,os.cpu_count() or 2))
NAMES=['Mia','Leo','Ava','Noah','Lily','Ben','Ella','Sam']; PLACES=['garden','forest','school','beach','farm','park']; THINGS=['red ball','small key','blue kite','old book','silver coin','green box']; ACTIONS=['found','lost','carried','opened','shared','fixed']
def make_corpus(n=2500):
 out=[]
 for _ in range(n):
  a,b=random.sample(NAMES,2); p=random.choice(PLACES); t=random.choice(THINGS); v=random.choice(ACTIONS)
  out.append(f"Once upon a time, {a} went to the {p}. {a} {v} a {t}. Then {b} came to help. They were kind and went home happy.\n")
 return ''.join(out)
text=make_corpus(); split=int(len(text)*.9); train_text=text[:split]; val_text=text[split:]; chars=sorted(set(text)); stoi={c:i for i,c in enumerate(chars)}
train=torch.tensor([stoi[c] for c in train_text],dtype=torch.long); val=torch.tensor([stoi[c] for c in val_text],dtype=torch.long)

class Block(nn.Module):
 def __init__(self,d=64,h=4,ff=128):
  super().__init__(); self.n1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,h,batch_first=True); self.n2=nn.LayerNorm(d); self.fc1=nn.Linear(d,ff); self.fc2=nn.Linear(ff,d)
 def forward(self,x,mask):
  z=self.n1(x); a,_=self.attn(z,z,z,attn_mask=mask,need_weights=False); x=x+a; z=self.n2(x); return x+self.fc2(F.gelu(self.fc1(z)))
class SharedLM(nn.Module):
 def __init__(self,vocab,depth=16,d=64):
  super().__init__(); self.depth=depth; self.emb=nn.Embedding(vocab,d); self.pos=nn.Embedding(64,d); self.block=Block(d); self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,vocab,bias=False); self.head.weight=self.emb.weight
 def forward(self,idx):
  _,T=idx.shape; x=self.emb(idx)+self.pos(torch.arange(T,device=idx.device)); mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=idx.device),1)
  for _ in range(self.depth): x=self.block(x,mask)
  return self.head(self.norm(x))
class DuplicatedLM(nn.Module):
 def __init__(self,shared):
  super().__init__(); self.depth=shared.depth; self.emb=copy.deepcopy(shared.emb); self.pos=copy.deepcopy(shared.pos); self.blocks=nn.ModuleList([copy.deepcopy(shared.block) for _ in range(shared.depth)]); self.norm=copy.deepcopy(shared.norm); self.head=nn.Linear(shared.head.in_features,shared.head.out_features,bias=False); self.head.weight=self.emb.weight
 def forward(self,idx):
  _,T=idx.shape; x=self.emb(idx)+self.pos(torch.arange(T,device=idx.device)); mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=idx.device),1)
  for b in self.blocks: x=b(x,mask)
  return self.head(self.norm(x))

def batch(data,b=24,T=64):
 ix=torch.randint(0,len(data)-T-1,(b,)); return torch.stack([data[i:i+T] for i in ix]),torch.stack([data[i+1:i+T+1] for i in ix])
def loss_on(m,data,n=20,T=64):
 m.eval(); z=0
 with torch.inference_mode():
  for i in range(n):
   start=(i*T)%max(1,len(data)-T-1); x=data[start:start+T][None,:]; y=data[start+1:start+T+1][None,:]; z+=F.cross_entropy(m(x).reshape(-1,len(chars)),y.reshape(-1)).item()
 return z/n
def unique_bytes(m):
 seen=set(); n=0
 for t in list(m.parameters())+list(m.buffers()):
  p=t.untyped_storage().data_ptr()
  if p not in seen: seen.add(p); n+=t.untyped_storage().nbytes()
 return n
def q4_module_bytes(mod):
 total=0
 for p in mod.parameters(recurse=True):
  if p.ndim==2:
   pk,s,_=q4_rows(p.detach()); total+=pk.numel()+s.numel()*s.element_size()
  else: total+=p.numel()*2
 return total
def q4_logical_file_bytes(m,duplicate_blocks):
 outside=q4_module_bytes(m.emb)+q4_module_bytes(m.pos)+q4_module_bytes(m.norm); block=q4_module_bytes(m.block)
 return outside+block*(m.depth if duplicate_blocks else 1)+256

def main():
 m=SharedLM(len(chars)); opt=torch.optim.AdamW(m.parameters(),lr=2e-3); t0=time.time(); m.train()
 for step in range(220):
  x,y=batch(train); loss=F.cross_entropy(m(x).reshape(-1,len(chars)),y.reshape(-1)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
  if (step+1)%55==0: print('step',step+1,'loss',float(loss.detach()),flush=True)
 train_s=time.time()-t0; sl=loss_on(m,val); dup=DuplicatedLM(m).eval(); dl=loss_on(dup,val); x,_=batch(val,4)
 with torch.inference_mode(): diff=float((m.eval()(x)-dup(x)).abs().max())
 sb=unique_bytes(m); db=unique_bytes(dup); fq=q4_logical_file_bytes(m,True); fl=q4_logical_file_bytes(m,False); scratch=4*64*128*6
 result={'logical_depth':m.depth,'physical_block_bundles_larc':1,'physical_block_bundles_naive':m.depth,'vocab':len(chars),'train_seconds':train_s,'shared_val_nll':sl,'duplicated_val_nll':dl,'max_logit_abs_diff':diff,'shared_fp32_unique_bytes':sb,'duplicated_fp32_unique_bytes':db,'resident_weight_reduction':db/sb,'naive_q4_logical_file_bytes':fq,'larc_q4_reference_file_bytes':fl,'q4_file_reduction':fq/fl,'bounded_shared_scratch_bytes':scratch,'naive_modeled_peak_bytes':db+scratch,'larc_modeled_peak_bytes':sb+scratch,'modeled_total_memory_reduction':(db+scratch)/(sb+scratch),'quality_retention_exact':diff<1e-6 and abs(sl-dl)<1e-7}
 Path('benchmarks/run2_recurrent_conformance.json').parent.mkdir(exist_ok=True); Path('benchmarks/run2_recurrent_conformance.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
