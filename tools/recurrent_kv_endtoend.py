import math, random, time, json, os
import torch
from torch import nn
import torch.nn.functional as F
random.seed(1); torch.manual_seed(1); torch.set_num_threads(min(4,os.cpu_count() or 2))
NAMES=['Mia','Leo','Ava','Noah','Lily','Ben','Ella','Sam','Ivy','Max']; PLACES=['garden','forest','school','beach','farm','park','shop']; THINGS=['red ball','small key','blue kite','old book','silver coin','green box','toy boat']; ACTIONS=['found','lost','carried','opened','shared','fixed','painted']
def corpus(n=4000):
 o=[]
 for _ in range(n):
  a,b=random.sample(NAMES,2);p=random.choice(PLACES);t=random.choice(THINGS);v=random.choice(ACTIONS)
  o.append(f'Once upon a time, {a} went to the {p}. {a} {v} a {t}. Then {b} came to help. They were kind and went home happy.\n')
 return ''.join(o)
text=corpus(); sp=int(.9*len(text)); chars=sorted(set(text)); stoi={c:i for i,c in enumerate(chars)}; train=torch.tensor([stoi[c] for c in text[:sp]],dtype=torch.long); val=torch.tensor([stoi[c] for c in text[sp:]],dtype=torch.long)
D=128;H=4;HD=D//H;FF=256;DEPTH=16;CTX=64;RANK=12
class Block(nn.Module):
 def __init__(self):
  super().__init__(); self.n1=nn.LayerNorm(D); self.attn=nn.MultiheadAttention(D,H,batch_first=True); self.n2=nn.LayerNorm(D); self.fc1=nn.Linear(D,FF); self.fc2=nn.Linear(FF,D)
 def forward(self,x,mask):
  z=self.n1(x);a,_=self.attn(z,z,z,attn_mask=mask,need_weights=False);x=x+a;z=self.n2(x);return x+self.fc2(F.gelu(self.fc1(z)))
class LM(nn.Module):
 def __init__(self):
  super().__init__();self.emb=nn.Embedding(len(chars),D);self.pos=nn.Embedding(CTX,D);self.block=Block();self.norm=nn.LayerNorm(D);self.head=nn.Linear(D,len(chars),bias=False);self.head.weight=self.emb.weight
 def forward(self,idx):
  T=idx.shape[1];x=self.emb(idx)+self.pos(torch.arange(T));mask=torch.triu(torch.ones(T,T,dtype=torch.bool),1)
  for _ in range(DEPTH):x=self.block(x,mask)
  return self.head(self.norm(x))
def batch(data,b=16,T=32):
 ix=torch.randint(0,len(data)-T-1,(b,));return torch.stack([data[i:i+T] for i in ix]),torch.stack([data[i+1:i+T+1] for i in ix])
m=LM();opt=torch.optim.AdamW(m.parameters(),lr=2e-3);t0=time.time();m.train()
for step in range(100):
 x,y=batch(train); logits=m(x);loss=F.cross_entropy(logits.reshape(-1,len(chars)),y.reshape(-1));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 if (step+1)%25==0: print('train',step+1,float(loss.detach()),flush=True)
print('train_seconds',time.time()-t0);m.eval()
def fit(x,r):
 _,_,v=torch.pca_lowrank(x.float(),q=min(r+4,min(x.shape)),center=False,niter=3);return v[:,:r].T.contiguous()
def baseline_incremental(tokens,collect=False):
 caches=[([],[]) for _ in range(DEPTH)]; outs=[]; KCOL=[[] for _ in range(H)];VCOL=[[] for _ in range(H)];b=m.block
 with torch.inference_mode():
  for t,tid in enumerate(tokens):
   x=m.emb(tid.view(1))+m.pos(torch.tensor([t]))
   for depth in range(DEPTH):
    z=b.n1(x);qkv=F.linear(z,b.attn.in_proj_weight,b.attn.in_proj_bias);q,k,v=qkv.chunk(3,-1);q=q.view(H,HD);k=k.view(H,HD);v=v.view(H,HD);caches[depth][0].append(k.clone());caches[depth][1].append(v.clone())
    if collect:
     for h in range(H):KCOL[h].append(k[h].clone());VCOL[h].append(v[h].clone())
    K=torch.stack(caches[depth][0],dim=0).permute(1,0,2);V=torch.stack(caches[depth][1],dim=0).permute(1,0,2);a=torch.softmax(torch.einsum('hd,htd->ht',q,K)/math.sqrt(HD),-1);av=torch.einsum('ht,htd->hd',a,V).reshape(1,D);x=x+F.linear(av,b.attn.out_proj.weight,b.attn.out_proj.bias);zz=b.n2(x);x=x+b.fc2(F.gelu(b.fc1(zz)))
   outs.append(m.head(m.norm(x)).squeeze(0))
 return torch.stack(outs),KCOL,VCOL,caches
cal=val[:CTX];base_cal,KCOL,VCOL,_=baseline_incremental(cal,True);KB=torch.stack([fit(torch.stack(KCOL[h]),RANK) for h in range(H)]);VB=torch.stack([fit(torch.stack(VCOL[h]),RANK) for h in range(H)])
with torch.inference_mode(): full=m(cal[None,:])[0]
inc_err=float((full-base_cal).abs().max());print('incremental_err',inc_err,flush=True)
def pack_vec(v):
 mn=v.min(); mx=v.max(); sc=((mx-mn)/3).clamp_min(1e-8); q=torch.round((v-mn)/sc).clamp(0,3).to(torch.uint8);pad=(-q.numel())%4
 if pad: q=torch.cat([q,torch.zeros(pad,dtype=torch.uint8)])
 return (q[0::4]|(q[1::4]<<2)|(q[2::4]<<4)|(q[3::4]<<6)).contiguous(),mn.half(),sc.half()
def unpack_mat(p,mn,sc,n):
 q=torch.empty((*p.shape[:-1],p.shape[-1]*4),dtype=torch.uint8);q[...,0::4]=p&3;q[...,1::4]=(p>>2)&3;q[...,2::4]=(p>>4)&3;q[...,3::4]=(p>>6)&3
 return q[...,:n].float()*sc.float()[...,None]+mn.float()[...,None]
def compressed_incremental(tokens):
 T=len(tokens); pb=(RANK+3)//4;PK=torch.empty((DEPTH,H,T,pb),dtype=torch.uint8);PV=torch.empty_like(PK);MK=torch.empty((DEPTH,H,T),dtype=torch.float16);SK=torch.empty_like(MK);MV=torch.empty_like(MK);SV=torch.empty_like(MK);outs=[];b=m.block
 with torch.inference_mode():
  for t,tid in enumerate(tokens):
   x=m.emb(tid.view(1))+m.pos(torch.tensor([t]))
   for depth in range(DEPTH):
    z=b.n1(x);qkv=F.linear(z,b.attn.in_proj_weight,b.attn.in_proj_bias);q,k,v=qkv.chunk(3,-1);q=q.view(H,HD);k=k.view(H,HD);v=v.view(H,HD);avh=[]
    for h in range(H):
     kl=k[h]@KB[h].T;vl=v[h]@VB[h].T;pk,mk,sk=pack_vec(kl);pv,mv,sv=pack_vec(vl);PK[depth,h,t]=pk;PV[depth,h,t]=pv;MK[depth,h,t]=mk;SK[depth,h,t]=sk;MV[depth,h,t]=mv;SV[depth,h,t]=sv;K=unpack_mat(PK[depth,h,:t+1],MK[depth,h,:t+1],SK[depth,h,:t+1],RANK);V=unpack_mat(PV[depth,h,:t+1],MV[depth,h,:t+1],SV[depth,h,:t+1],RANK);ql=q[h]@KB[h].T;a=torch.softmax((K@ql)/math.sqrt(HD),0);avh.append((a@V)@VB[h])
    av=torch.stack(avh).reshape(1,D);x=x+F.linear(av,b.attn.out_proj.weight,b.attn.out_proj.bias);zz=b.n2(x);x=x+b.fc2(F.gelu(b.fc1(zz)))
   outs.append(m.head(m.norm(x)).squeeze(0))
 cache_bytes=sum(t.numel()*t.element_size() for t in (PK,PV,MK,SK,MV,SV));basis_bytes=(KB.numel()+VB.numel())//2+(KB.shape[0]*KB.shape[1]*2)*2
 return torch.stack(outs),cache_bytes,basis_bytes
seq=val[CTX:2*CTX+1];inp=seq[:-1];target=seq[1:];bl,_,_,_=baseline_incremental(inp,False);t1=time.time();cl,actual_cache,basis_bytes=compressed_incremental(inp);comp_seconds=time.time()-t1;base_nll=F.cross_entropy(bl,target).item();comp_nll=F.cross_entropy(cl,target).item()
def q4_bytes_tensor(t):
 if t.ndim==2:return t.shape[0]*((t.shape[1]+1)//2)+2*t.shape[0]
 return t.numel()*2
def mod_bytes(mod):return sum(q4_bytes_tensor(p.detach()) for p in mod.parameters(recurse=True))
outside=mod_bytes(m.emb)+mod_bytes(m.pos)+mod_bytes(m.norm);block=mod_bytes(m.block);naive_w=outside+DEPTH*block;larc_w=outside+block;baseline_kv=DEPTH*CTX*D*2*2;scratch=(D+D+3*D+H*CTX+FF+CTX*RANK)*4;baseline_total=naive_w+baseline_kv+scratch;larc_total=larc_w+actual_cache+basis_bytes+scratch
res=dict(d=D,heads=H,depth=DEPTH,context=CTX,rank=RANK,incremental_max_logit_error=inc_err,baseline_nll=base_nll,latent_q2_nll=comp_nll,nll_increase_pct=(comp_nll/base_nll-1)*100,naive_q4_weight_bytes=naive_w,larc_shared_q4_weight_bytes=larc_w,weight_reduction=naive_w/larc_w,baseline_fp16_kv_bytes=baseline_kv,actual_packed_latent_q2_cache_bytes=actual_cache,shared_q4_kv_basis_bytes=basis_bytes,kv_reduction=baseline_kv/(actual_cache+basis_bytes),scratch_bytes=scratch,baseline_total_bytes=baseline_total,larc_total_bytes=larc_total,total_memory_reduction=baseline_total/larc_total,quality_gate_15pct=(comp_nll<=base_nll*1.15),total_memory_gate_10x=(baseline_total/larc_total>=10),compressed_eval_seconds=comp_seconds)
print(json.dumps(res,indent=2));os.makedirs('benchmarks',exist_ok=True);open('benchmarks/run2_recurrent_kv_endtoend.json','w').write(json.dumps(res,indent=2)+'\n')
