#!/usr/bin/env python3
"""Canonical Run-4 reconstruction of the documented Run-3 control protocol.

This script intentionally owns its corpus seeds, split, model definition, optimizer
steps, evaluation stream and Q4 dequantization so the resulting JSON has a
committed generator. It is a reconstruction because the original generator for
run3_posttrain_corrected_100k.json was not committed.
"""
import copy,json,os,random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(min(4,os.cpu_count() or 2))
NAMES=['Mia','Leo','Ava','Noah','Lily','Ben','Ella','Sam','Ivy','Max'];PLACES=['garden','forest','school','beach','farm','park','shop'];THINGS=['red ball','small key','blue kite','old book','silver coin','green box','toy boat'];ACTIONS=['found','lost','carried','opened','shared','fixed','painted']
def corpus(n,seed):
 r=random.Random(seed);out=[]
 for _ in range(n):
  a,b=r.sample(NAMES,2);p=r.choice(PLACES);t=r.choice(THINGS);v=r.choice(ACTIONS);out.append(f'Once upon a time, {a} went to the {p}. {a} {v} a {t}. Then {b} came to help. They were kind and went home happy.\n')
 return ''.join(out)
train_text=corpus(6000,3);chars=sorted(set(train_text));stoi={c:i for i,c in enumerate(chars)};split=int(len(train_text)*.9);train=torch.tensor([stoi[c] for c in train_text[:split]]);eval_text=corpus(1000,999);ev=torch.tensor([stoi[c] for c in eval_text])
D=128;H=4;FF=256;DEPTH=16;CTX=64
class Block(nn.Module):
 def __init__(self):super().__init__();self.n1=nn.LayerNorm(D);self.attn=nn.MultiheadAttention(D,H,batch_first=True);self.n2=nn.LayerNorm(D);self.fc1=nn.Linear(D,FF);self.fc2=nn.Linear(FF,D)
 def forward(self,x,mask):z=self.n1(x);a,_=self.attn(z,z,z,attn_mask=mask,need_weights=False);x=x+a;z=self.n2(x);return x+self.fc2(F.gelu(self.fc1(z)))
class Teacher(nn.Module):
 def __init__(self):super().__init__();self.emb=nn.Embedding(len(chars),D);self.pos=nn.Embedding(CTX,D);self.blocks=nn.ModuleList([Block() for _ in range(DEPTH)]);self.norm=nn.LayerNorm(D);self.head=nn.Linear(D,len(chars),bias=False);self.head.weight=self.emb.weight
 def forward(self,idx):
  T=idx.shape[1];x=self.emb(idx)+self.pos(torch.arange(T));mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1)
  for b in self.blocks:x=b(x,mask)
  return self.head(self.norm(x))
class Shared(nn.Module):
 def __init__(self):super().__init__();self.emb=nn.Embedding(len(chars),D);self.pos=nn.Embedding(CTX,D);self.block=Block();self.norm=nn.LayerNorm(D);self.head=nn.Linear(D,len(chars),bias=False);self.head.weight=self.emb.weight
 def forward(self,idx):
  T=idx.shape[1];x=self.emb(idx)+self.pos(torch.arange(T));mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1)
  for _ in range(DEPTH):x=self.block(x,mask)
  return self.head(self.norm(x))
class Converted(Shared):
 def __init__(self,t):
  nn.Module.__init__(self);self.emb=copy.deepcopy(t.emb);self.pos=copy.deepcopy(t.pos);self.block=Block();self.norm=copy.deepcopy(t.norm);self.head=nn.Linear(D,len(chars),bias=False);self.head.weight=self.emb.weight
  self.block.load_state_dict({k:torch.stack([b.state_dict()[k].float() for b in t.blocks]).mean(0).to(v.dtype) for k,v in self.block.state_dict().items()})
def batch(b=16,T=32):
 ix=torch.randint(0,len(train)-T-1,(b,));return torch.stack([train[i:i+T] for i in ix]),torch.stack([train[i+1:i+T+1] for i in ix])
def train_model(m,steps,lr):
 opt=torch.optim.AdamW(m.parameters(),lr=lr);m.train()
 for _ in range(steps):
  x,y=batch();loss=F.cross_entropy(m(x).reshape(-1,len(chars)),y.reshape(-1));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
def evaluate(m):
 m.eval();usable=min(100032,((len(ev)-1)//CTX)*CTX);starts=list(range(0,usable,CTX));total=0.;count=0
 with torch.inference_mode():
  for j in range(0,len(starts),32):
   ss=starts[j:j+32];x=torch.stack([ev[s:s+CTX] for s in ss]);y=torch.stack([ev[s+1:s+CTX+1] for s in ss]);total+=float(F.cross_entropy(m(x).reshape(-1,len(chars)),y.reshape(-1),reduction='sum'));count+=y.numel()
 return total/count,count
def q4dq(w):
 x=w.detach().float();pos=x.amax(1).clamp_min(0)/7.;neg=(-x.amin(1)).clamp_min(0)/8.;s=torch.maximum(pos,neg).clamp_min(1e-8).half().float();return torch.round(x/s[:,None]).clamp(-8,7)*s[:,None]
def quantized_copy(m):
 q=copy.deepcopy(m)
 with torch.no_grad():
  for p in q.parameters():p.copy_(q4dq(p) if p.ndim==2 else p.half().float())
 return q
def main():
 torch.manual_seed(3);teacher=Teacher();train_model(teacher,120,2e-3);student=Converted(teacher);train_model(student,200,1.5e-3)
 torch.manual_seed(3);scratch=Shared();train_model(scratch,320,2e-3)
 tn,n=evaluate(teacher);sn,_=evaluate(student);rn,_=evaluate(scratch);tqn,_=evaluate(quantized_copy(teacher));sqn,_=evaluate(quantized_copy(student))
 control={"status":"canonical Run-4 reconstruction, not proof of exact Run-3 provenance","training_seed":3,"evaluation_stream_seed":999,"evaluation_chars":n,"training_fraction":0.9,"teacher_steps":120,"recovery_steps":200,"scratch_steps":320,"teacher_fp32_nll":tn,"converted_recovered_fp32_nll":sn,"scratch_recurrent_320_fp32_nll":rn,"converted_minus_teacher_nats_per_char":sn-tn}
 q4={"evaluation_stream_seed":999,"evaluation_chars":n,"teacher_fp32_nll":tn,"teacher_q4_dequant_nll":tqn,"teacher_q4_delta_nats_per_char":tqn-tn,"converted_fp32_nll":sn,"converted_q4_dequant_nll":sqn,"converted_q4_delta_nats_per_char":sqn-sn,"structural_fp32_delta_nats_per_char":sn-tn,"structural_q4_delta_nats_per_char":sqn-tqn,"excess_q4_damage_in_reused_block_nats_per_char":(sqn-sn)-(tqn-tn)}
 Path('benchmarks/run4_control_reproduction.json').write_text(json.dumps(control,indent=2)+'\n');Path('benchmarks/run4_q4_weight_quality.json').write_text(json.dumps(q4,indent=2)+'\n');print(json.dumps({"control":control,"q4":q4},indent=2))
if __name__=='__main__':main()
