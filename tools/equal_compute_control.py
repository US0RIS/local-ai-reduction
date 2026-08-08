#!/usr/bin/env python3
import copy,json,os,random,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

random.seed(3);torch.manual_seed(3);torch.set_num_threads(min(4,os.cpu_count() or 2))
NAMES=['Mia','Leo','Ava','Noah','Lily','Ben','Ella','Sam','Ivy','Max'];PLACES=['garden','forest','school','beach','farm','park','shop'];THINGS=['red ball','small key','blue kite','old book','silver coin','green box','toy boat'];ACTIONS=['found','lost','carried','opened','shared','fixed','painted']
def corpus(n=6000):
 o=[]
 for _ in range(n):
  a,b=random.sample(NAMES,2);p=random.choice(PLACES);t=random.choice(THINGS);v=random.choice(ACTIONS);o.append(f'Once upon a time, {a} went to the {p}. {a} {v} a {t}. Then {b} came to help. They were kind and went home happy.\n')
 return ''.join(o)
text=corpus();sp=int(len(text)*.9);chars=sorted(set(text));stoi={c:i for i,c in enumerate(chars)};train=torch.tensor([stoi[c] for c in text[:sp]]);val=torch.tensor([stoi[c] for c in text[sp:]])
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
  av={};sd=self.block.state_dict()
  for k,v in sd.items():av[k]=torch.stack([b.state_dict()[k].float() for b in t.blocks]).mean(0).to(v.dtype)
  self.block.load_state_dict(av)
def batch(data,b=16,T=32):
 ix=torch.randint(0,len(data)-T-1,(b,));return torch.stack([data[i:i+T] for i in ix]),torch.stack([data[i+1:i+T+1] for i in ix])
def evaluate(m,n=32):
 m.eval();z=[]
 with torch.inference_mode():
  for j in range(n):
   s=j*CTX;seq=val[s:s+CTX+1];z.append(F.cross_entropy(m(seq[:-1][None,:])[0],seq[1:]).item())
 return sum(z)/len(z)
def train_model(m,steps,lr):
 opt=torch.optim.AdamW(m.parameters(),lr=lr);m.train();t=time.time()
 for _ in range(steps):
  x,y=batch(train);l=F.cross_entropy(m(x).reshape(-1,len(chars)),y.reshape(-1));opt.zero_grad();l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 return time.time()-t

def main():
 torch.manual_seed(3);teacher=Teacher();tt=train_model(teacher,120,2e-3);tn=evaluate(teacher);converted=Converted(teacher);pre=evaluate(converted);rt=train_model(converted,200,1.5e-3);cn=evaluate(converted)
 torch.manual_seed(3);scratch=Shared();st=train_model(scratch,320,2e-3);sn=evaluate(scratch)
 res={'experiment':'equal_compute_recurrent_control','seed':3,'model':{'d':D,'heads':H,'ff':FF,'logical_depth':DEPTH,'context':CTX,'vocab':len(chars)},'evaluation_contexts':32,'evaluation_chars':32*CTX,'teacher_independent_blocks':DEPTH,'teacher_steps':120,'teacher_eval_nll':tn,'teacher_train_seconds':tt,'converted_pre_recovery_nll':pre,'converted_recovery_steps':200,'converted_post_recovery_nll':cn,'converted_recovery_seconds':rt,'teacher_plus_recovery_seconds':tt+rt,'scratch_recurrent_steps':320,'scratch_recurrent_eval_nll':sn,'scratch_recurrent_train_seconds':st,'converted_vs_scratch_nll_ratio':cn/sn}
 Path('benchmarks').mkdir(exist_ok=True);Path('benchmarks/run3_equal_compute_control.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
if __name__=='__main__':main()
