#!/usr/bin/env python3
from __future__ import annotations
import argparse,gc,json,math,re,struct
from pathlib import Path
import torch
from larc.paged_container import LARCv2StreamWriter,CODEC_RAW,CODEC_Q4_ROW,FLAG_REQUIRED,FLAG_SHARED,FLAG_STREAMABLE
from larc.q4_runtime import q4_rows,dequantize_q4_rows
from larc.safetensors_range import ShardedSafeTensorSource

MATRIX_TYPES={'q_proj':'self_attn.q_proj.weight','k_proj':'self_attn.k_proj.weight','v_proj':'self_attn.v_proj.weight','o_proj':'self_attn.o_proj.weight','gate_proj':'mlp.gate_proj.weight','up_proj':'mlp.up_proj.weight','down_proj':'mlp.down_proj.weight'}
LAYER_RE=re.compile(r'^model\.layers\.(\d+)\.(.+)$');Q4_HEAD=struct.Struct('<IIQ')
def layer_name(layer:int,suffix:str)->str:return f'model.layers.{layer}.{suffix}'
def q4_parts(x:torch.Tensor):
 if x.ndim!=2:raise ValueError(f'Q4 page expects 2-D matrix, got {tuple(x.shape)}')
 p,s,c=q4_rows(x.detach().float().contiguous());return p.contiguous(),s.contiguous(),c
def q4_blob_from_parts(p:torch.Tensor,s:torch.Tensor,cols:int)->bytes:
 rows=p.shape[0];pb=p.cpu().numpy().tobytes();sb=s.cpu().numpy().tobytes();return Q4_HEAD.pack(rows,cols,len(pb))+pb+sb
def q4_blob(x:torch.Tensor):
 p,s,c=q4_parts(x);return q4_blob_from_parts(p,s,c),dequantize_q4_rows(p,s,c)
def fp16_blob(x:torch.Tensor)->bytes:return x.detach().to(torch.float16).contiguous().cpu().numpy().tobytes()
def discover(src:ShardedSafeTensorSource):
 names=set(src.names());layers=sorted({int(m.group(1)) for n in names if (m:=LAYER_RE.match(n))})
 if not layers or layers!=list(range(max(layers)+1)):raise ValueError(f'non-contiguous/no layers: {layers[:8]}')
 L=len(layers)
 for n in ('model.embed_tokens.weight','model.norm.weight','lm_head.weight'):
  if n not in names:raise KeyError(n)
 for l in range(L):
  for suf in MATRIX_TYPES.values():
   if layer_name(l,suf) not in names:raise KeyError(layer_name(l,suf))
  for suf in ('input_layernorm.weight','post_attention_layernorm.weight'):
   if layer_name(l,suf) not in names:raise KeyError(layer_name(l,suf))
 return L
def make_page_plan(src,L,ranks):
 pages=[];pid=1
 def add(role,name,codec,flags,shape=None,layer=None,matrix_type=None):
  nonlocal pid;d={'page_id':pid,'role':role,'source_name':name,'codec_id':codec,'flags':flags}
  if shape is not None:d['shape']=list(shape)
  if layer is not None:d['layer']=layer
  if matrix_type is not None:d['matrix_type']=matrix_type
  pages.append(d);pid+=1
 for n,role in [('model.embed_tokens.weight','embedding'),('lm_head.weight','lm_head')]:add(role,n,CODEC_Q4_ROW,FLAG_REQUIRED|FLAG_STREAMABLE,src.info(n).shape)
 add('final_norm','model.norm.weight',CODEC_RAW,FLAG_REQUIRED|FLAG_STREAMABLE,src.info('model.norm.weight').shape)
 for l in range(L):
  for suf,role in [('input_layernorm.weight','input_norm'),('post_attention_layernorm.weight','post_attention_norm')]:
   n=layer_name(l,suf);add(role,n,CODEC_RAW,FLAG_REQUIRED|FLAG_STREAMABLE,src.info(n).shape,layer=l)
 for typ,suf in MATRIX_TYPES.items():
  shape=src.info(layer_name(0,suf)).shape;r=ranks[typ];add('shared_base',f'@shared/{typ}',CODEC_Q4_ROW,FLAG_REQUIRED|FLAG_SHARED|FLAG_STREAMABLE,shape,matrix_type=typ)
  for l in range(L):
   m,n=shape;add('residual_A',layer_name(l,suf),CODEC_Q4_ROW,FLAG_REQUIRED|FLAG_STREAMABLE,(m,r),layer=l,matrix_type=typ);add('residual_B',layer_name(l,suf),CODEC_Q4_ROW,FLAG_REQUIRED|FLAG_STREAMABLE,(r,n),layer=l,matrix_type=typ)
 return pages
def fit_residual(residual,rank,oversample,niter,seed):
 rank=min(rank,*residual.shape);q=min(min(residual.shape),rank+oversample);torch.manual_seed(seed)
 if q>=min(residual.shape):U,S,Vh=torch.linalg.svd(residual,full_matrices=False);U=U[:,:rank];S=S[:rank];V=Vh[:rank].t()
 else:U,S,V=torch.svd_lowrank(residual,q=q,niter=niter);U=U[:,:rank];S=S[:rank];V=V[:,:rank]
 root=S.clamp_min(0).sqrt();return (U*root).contiguous(),(root[:,None]*V.t()).contiguous()
def parse_ranks(spec):
 if spec.isdigit():return {k:int(spec) for k in MATRIX_TYPES}
 out={}
 for item in spec.split(','):k,v=item.split('=',1);out[k.strip()]=int(v)
 missing=set(MATRIX_TYPES)-set(out)
 if missing:raise ValueError(f'missing ranks for {sorted(missing)}')
 return out
def convert(index,out_path,ranks,oversample=8,niter=2,seed=1234):
 src=ShardedSafeTensorSource(index);L=discover(src);plan=make_page_plan(src,L,ranks);manifest={'architecture':'mistral','conversion':'SoftShare-10X two-pass initialization','source':{'safetensors_index':str(index),'full_source_file_required_locally':False},'softshare':{'layers':L,'ranks':ranks,'equation':'W_layer = shared_Q4 + A_Q4 @ B_Q4','residual_fit_reference':'dequantized stored shared_Q4','recovery_applied':False},'pages':plan};peak_source=peak_shared=0;page_lookup={(p['role'],p.get('matrix_type'),p.get('layer')):p for p in plan}
 with LARCv2StreamWriter(out_path,len(plan),manifest) as wr:
  for n,role in [('model.embed_tokens.weight','embedding'),('lm_head.weight','lm_head')]:
   x=src.read_tensor(n);peak_source=max(peak_source,x.numel()*x.element_size());blob,_=q4_blob(x);p=page_lookup[(role,None,None)];wr.add_page(p['page_id'],p['codec_id'],p['flags'],blob,logical_length=x.numel()*2);del x,blob
  x=src.read_tensor('model.norm.weight');peak_source=max(peak_source,x.numel()*x.element_size());p=page_lookup[('final_norm',None,None)];wr.add_page(p['page_id'],p['codec_id'],p['flags'],fp16_blob(x),logical_length=x.numel()*2);del x
  for l in range(L):
   for suf,role in [('input_layernorm.weight','input_norm'),('post_attention_layernorm.weight','post_attention_norm')]:
    n=layer_name(l,suf);x=src.read_tensor(n);peak_source=max(peak_source,x.numel()*x.element_size());p=page_lookup[(role,None,l)];wr.add_page(p['page_id'],p['codec_id'],p['flags'],fp16_blob(x),logical_length=x.numel()*2);del x
  for ti,(typ,suf) in enumerate(MATRIX_TYPES.items()):
   shared=None
   for l in range(L):
    x=src.read_tensor(layer_name(l,suf));peak_source=max(peak_source,x.numel()*x.element_size());xf=x.float();del x
    if shared is None:shared=xf
    else:shared.add_(xf);del xf
   shared.div_(L);peak_shared=max(peak_shared,shared.numel()*shared.element_size());logical=shared.numel()*2;parts=q4_parts(shared);del shared;shared_hat=dequantize_q4_rows(*parts);blob=q4_blob_from_parts(*parts);p=page_lookup[('shared_base',typ,None)];wr.add_page(p['page_id'],p['codec_id'],p['flags'],blob,logical_length=logical);del blob,parts
   for l in range(L):
    x=src.read_tensor(layer_name(l,suf));peak_source=max(peak_source,x.numel()*x.element_size());res=x.float().sub_(shared_hat);del x;A,B=fit_residual(res,ranks[typ],oversample,niter,seed+ti*1000+l);del res
    for role,t in [('residual_A',A),('residual_B',B)]:
     blob,_=q4_blob(t);p=page_lookup[(role,typ,l)];wr.add_page(p['page_id'],p['codec_id'],p['flags'],blob,logical_length=t.numel()*2);del blob
    del A,B;gc.collect()
   del shared_hat;gc.collect()
 return {'output':str(out_path),'page_count':len(plan),'layers':L,'ranks':ranks,'source_bytes_fetched_or_read':src.bytes_fetched,'largest_single_source_tensor_bytes':peak_source,'largest_shared_fp32_base_bytes':peak_shared,'bounded_local_source_file_required':False}
def main():
 ap=argparse.ArgumentParser(description='Two-pass tensor-range SoftShare converter; never requires a complete source checkpoint file locally.');ap.add_argument('--index',required=True);ap.add_argument('--output',required=True);ap.add_argument('--ranks',default='96');ap.add_argument('--oversample',type=int,default=8);ap.add_argument('--niter',type=int,default=2);ap.add_argument('--seed',type=int,default=1234);ap.add_argument('--report');a=ap.parse_args();report=convert(a.index,a.output,parse_ranks(a.ranks),a.oversample,a.niter,a.seed);text=json.dumps(report,indent=2)+'\n';print(text,end='');
 if a.report:Path(a.report).write_text(text)
if __name__=='__main__':main()
