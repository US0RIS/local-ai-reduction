import math,random,json,os,copy
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
D=128;H=4;HD=32;FF=256;DEPTH=16;CTX=64;RANK=16
class Block(nn.Module):
 def __init__(self):super().__init__();self.n1=nn.LayerNorm(D);self.attn=nn.MultiheadAttention(D,H,batch_first=True);self.n2=nn.LayerNorm(D);self.fc1=nn.Linear(D,FF);self.fc2=nn.Linear(FF,D)
 def forward(self,x,mask):z=self.n1(x);a,_=self.attn(z,z,z,attn_mask=mask,need_weights=False);x=x+a;z=self.n2(x);return x+self.fc2(F.gelu(self.fc1(z)))
class Teacher(nn.Module):
 def __init__(self):super().__init__();self.emb=nn.Embedding(len(chars),D);self.pos=nn.Embedding(CTX,D);self.blocks=nn.ModuleList([Block() for _ in range(DEPTH)]);self.norm=nn.LayerNorm(D);self.head=nn.Linear(D,len(chars),bias=False);self.head.weight=self.emb.weight
 def forward(self,idx):
  T=idx.shape[1];x=self.emb(idx)+self.pos(torch.arange(T));mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1)
  for b in self.blocks:x=b(x,mask)
  return self.head(self.norm(x))
class Student(nn.Module):
 def __init__(self,t):
  super().__init__();self.emb=copy.deepcopy(t.emb);self.pos=copy.deepcopy(t.pos);self.block=Block();self.norm=copy.deepcopy(t.norm);self.head=nn.Linear(D,len(chars),bias=False);self.head.weight=self.emb.weight
  av={};sd=self.block.state_dict()
  for k,v in sd.items():av[k]=torch.stack([b.state_dict()[k].float() for b in t.blocks]).mean(0).to(v.dtype)
  self.block.load_state_dict(av)
 def forward(self,idx):
  T=idx.shape[1];x=self.emb(idx)+self.pos(torch.arange(T));mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1)
  for _ in range(DEPTH):x=self.block(x,mask)
  return self.head(self.norm(x))
def batch(data,b=16,T=32):ix=torch.randint(0,len(data)-T-1,(b,));return torch.stack([data[i:i+T] for i in ix]),torch.stack([data[i+1:i+T+1] for i in ix])
def eval_full(m,n=8):
 m.eval();z=0
 with torch.inference_mode():
  for j in range(n):
   s=j*CTX;seq=val[s:s+CTX+1];z+=F.cross_entropy(m(seq[:-1][None,:])[0],seq[1:]).item()
 return z/n
teacher=Teacher();opt=torch.optim.AdamW(teacher.parameters(),lr=2e-3);teacher.train()
for step in range(120):
 x,y=batch(train);l=F.cross_entropy(teacher(x).reshape(-1,len(chars)),y.reshape(-1));opt.zero_grad();l.backward();torch.nn.utils.clip_grad_norm_(teacher.parameters(),1);opt.step()
 if (step+1)%30==0:print('teacher',step+1,float(l.detach()),flush=True)
teacher_nll=eval_full(teacher);student=Student(teacher);pre_nll=eval_full(student)
# Recovery happens only after conventional teacher pretraining and conversion.
opt=torch.optim.AdamW(student.parameters(),lr=1.5e-3);student.train()
for step in range(200):
 x,y=batch(train);sl=student(x);loss=F.cross_entropy(sl.reshape(-1,len(chars)),y.reshape(-1));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1);opt.step()
 if (step+1)%45==0:print('student',step+1,float(loss.detach()),flush=True)
student_nll=eval_full(student);student.eval()
def fit(x,r):_,_,v=torch.pca_lowrank(x.float(),q=min(r+4,min(x.shape)),center=False,niter=3);return v[:,:r].T.contiguous()
def baseline_inc(tokens,collect=False):
 caches=[([],[]) for _ in range(DEPTH)];outs=[];KC=[[] for _ in range(H)];VC=[[] for _ in range(H)];b=student.block
 with torch.inference_mode():
  for t,tid in enumerate(tokens):
   x=student.emb(tid.view(1))+student.pos(torch.tensor([t]))
   for d in range(DEPTH):
    z=b.n1(x);q,k,v=F.linear(z,b.attn.in_proj_weight,b.attn.in_proj_bias).chunk(3,-1);q=q.view(H,HD);k=k.view(H,HD);v=v.view(H,HD);caches[d][0].append(k.clone());caches[d][1].append(v.clone())
    if collect:
     for h in range(H):KC[h].append(k[h].clone());VC[h].append(v[h].clone())
    K=torch.stack(caches[d][0]).permute(1,0,2);V=torch.stack(caches[d][1]).permute(1,0,2);a=torch.softmax(torch.einsum('hd,htd->ht',q,K)/math.sqrt(HD),-1);av=torch.einsum('ht,htd->hd',a,V).reshape(1,D);x=x+F.linear(av,b.attn.out_proj.weight,b.attn.out_proj.bias);x=x+b.fc2(F.gelu(b.fc1(b.n2(x))))
   outs.append(student.head(student.norm(x)).squeeze(0))
 return torch.stack(outs),KC,VC
cal=val[:CTX];inc,KC,VC=baseline_inc(cal,True);KB=torch.stack([fit(torch.stack(KC[h]),RANK) for h in range(H)]);VB=torch.stack([fit(torch.stack(VC[h]),RANK) for h in range(H)])
with torch.inference_mode():inc_err=float((student(cal[None,:])[0]-inc).abs().max())
def pack_rows(x):
 mn=x.amin(-1);mx=x.amax(-1);sc=((mx-mn)/3).clamp_min(1e-8);q=torch.round((x-mn[...,None])/sc[...,None]).clamp(0,3).byte();pad=(-q.shape[-1])%4
 if pad:q=torch.cat([q,torch.zeros((*q.shape[:-1],pad),dtype=torch.uint8)],-1)
 return (q[...,0::4]|q[...,1::4]<<2|q[...,2::4]<<4|q[...,3::4]<<6).contiguous(),mn.half(),sc.half()
def unpack_rows(p,mn,sc,n):
 q=torch.empty((*p.shape[:-1],p.shape[-1]*4),dtype=torch.uint8);q[...,0::4]=p&3;q[...,1::4]=(p>>2)&3;q[...,2::4]=(p>>4)&3;q[...,3::4]=(p>>6)&3
 return q[...,:n].float()*sc.float()[...,None]+mn.float()[...,None]
def comp_inc(tokens):
 T=len(tokens);pb=(RANK+3)//4;PK=torch.empty(DEPTH,H,T,pb,dtype=torch.uint8);PV=torch.empty_like(PK);MK=torch.empty(DEPTH,H,T,dtype=torch.float16);SK=torch.empty_like(MK);MV=torch.empty_like(MK);SV=torch.empty_like(MK);outs=[];b=student.block
 with torch.inference_mode():
  for t,tid in enumerate(tokens):
   x=student.emb(tid.view(1))+student.pos(torch.tensor([t]))
   for d in range(DEPTH):
    z=b.n1(x);q,k,v=F.linear(z,b.attn.in_proj_weight,b.attn.in_proj_bias).chunk(3,-1);q=q.view(H,HD);k=k.view(H,HD);v=v.view(H,HD)
    kl=torch.einsum('hd,hrd->hr',k,KB);vl=torch.einsum('hd,hrd->hr',v,VB);pk,mk,sk=pack_rows(kl);pv,mv,sv=pack_rows(vl);PK[d,:,t]=pk;PV[d,:,t]=pv;MK[d,:,t]=mk;SK[d,:,t]=sk;MV[d,:,t]=mv;SV[d,:,t]=sv
    K=unpack_rows(PK[d,:,:t+1],MK[d,:,:t+1],SK[d,:,:t+1],RANK);V=unpack_rows(PV[d,:,:t+1],MV[d,:,:t+1],SV[d,:,:t+1],RANK);ql=torch.einsum('hd,hrd->hr',q,KB);scores=torch.einsum('htr,hr->ht',K,ql)/math.sqrt(HD);a=torch.softmax(scores,-1);vlat=torch.einsum('ht,htr->hr',a,V);av=torch.einsum('hr,hrd->hd',vlat,VB).reshape(1,D)
    x=x+F.linear(av,b.attn.out_proj.weight,b.attn.out_proj.bias);x=x+b.fc2(F.gelu(b.fc1(b.n2(x))))
   outs.append(student.head(student.norm(x)).squeeze(0))
 cb=sum(t.numel()*t.element_size() for t in (PK,PV,MK,SK,MV,SV));bb=(KB.numel()+VB.numel())//2+(KB.shape[0]*KB.shape[1]*2)*2;return torch.stack(outs),cb,bb
seq=val[CTX:2*CTX+1];inp=seq[:-1];target=seq[1:];c,cache,basis=comp_inc(inp);latent_nll=F.cross_entropy(c,target).item()
def qb(t):return t.shape[0]*((t.shape[1]+1)//2)+2*t.shape[0] if t.ndim==2 else t.numel()*2
def mod(mod):return sum(qb(p) for p in mod.parameters(recurse=True))
outside=mod(teacher.emb)+mod(teacher.pos)+mod(teacher.norm);teacher_w=outside+sum(mod(b) for b in teacher.blocks);student_w=mod(student.emb)+mod(student.pos)+mod(student.norm)+mod(student.block);basekv=DEPTH*CTX*D*4;scratch=(D+D+3*D+H*CTX+FF+CTX*RANK)*4;base_total=teacher_w+basekv+scratch;larc_total=student_w+cache+basis+scratch
res={'teacher_pretrained_nll':teacher_nll,'student_pre_recovery_nll':pre_nll,'student_post_recovery_nll':student_nll,'student_latent_q2_nll':latent_nll,'final_nll_increase_vs_teacher_pct':(latent_nll/teacher_nll-1)*100,'incremental_logit_error':inc_err,'teacher_q4_weight_bytes':teacher_w,'larc_q4_weight_bytes':student_w,'weight_reduction':teacher_w/student_w,'baseline_fp16_kv_bytes':basekv,'larc_packed_kv_bytes':cache,'larc_kv_basis_bytes':basis,'kv_reduction':basekv/(cache+basis),'scratch_bytes':scratch,'baseline_total_bytes':base_total,'larc_total_bytes':larc_total,'total_memory_reduction':base_total/larc_total,'passes_15pct_quality':latent_nll<=teacher_nll*1.15,'passes_10x_total_memory':base_total/larc_total>=10,'teacher_steps':120,'recovery_steps':200,'config':{'d':D,'heads':H,'depth':DEPTH,'context':CTX,'latent_rank':RANK}}
print(json.dumps(res,indent=2));os.makedirs('benchmarks',exist_ok=True);open('benchmarks/run2_posttrain_conversion.json','w').write(json.dumps(res,indent=2)+'\n')
