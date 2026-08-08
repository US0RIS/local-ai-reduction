#!/usr/bin/env python3
from __future__ import annotations
import argparse,gc,json,math
from pathlib import Path
import torch
from larc.paged_container import LARCv2StreamWriter,CODEC_RAW,CODEC_Q4_ROW,FLAG_REQUIRED,FLAG_SHARED,FLAG_STREAMABLE
from larc.q4_runtime import dequantize_q4_rows
from tools.stream_softshare_convert import MATRIX_TYPES,_open_source,discover,parse_aux,_read_aux,q4_parts,q4_blob_from_parts,q4_blob,fp16_blob,layer_name,DEFAULT_MISTRAL_Q4KM_BYTES


def parse_ranks(spec,src,L):
    if spec.isdigit():
        r=int(spec);return {k:min(r,*src.info(layer_name(0,suf)).shape) for k,suf in MATRIX_TYPES.items()}
    out={}
    for item in spec.split(','):k,v=item.split('=',1);out[k.strip()]=int(v)
    missing=set(MATRIX_TYPES)-set(out)
    if missing:raise ValueError(f'missing ranks for {sorted(missing)}')
    return {k:min(v,*src.info(layer_name(0,MATRIX_TYPES[k])).shape) for k,v in out.items()}

def make_plan(src,L,ranks,aux,source_metadata):
    pages=[]
    def add(role,name,codec,flags,shape=None,layer=None,matrix_type=None,resource_name=None):
        d={'page_id':len(pages)+1,'role':role,'source_name':name,'codec_id':codec,'flags':flags}
        if shape is not None:d['shape']=list(shape)
        if layer is not None:d['layer']=layer
        if matrix_type is not None:d['matrix_type']=matrix_type
        if resource_name is not None:d['resource_name']=resource_name
        pages.append(d)
    req,shared,stream=FLAG_REQUIRED,FLAG_SHARED,FLAG_STREAMABLE
    for n,role in [('model.embed_tokens.weight','embedding'),('lm_head.weight','lm_head')]:add(role,n,CODEC_Q4_ROW,req|stream,src.info(n).shape)
    add('final_norm','model.norm.weight',CODEC_RAW,req|stream,src.info('model.norm.weight').shape)
    for l in range(L):
        for suf,role in [('input_layernorm.weight','input_norm'),('post_attention_layernorm.weight','post_attention_norm')]:
            n=layer_name(l,suf);add(role,n,CODEC_RAW,req|stream,src.info(n).shape,l)
    for typ,suf in MATRIX_TYPES.items():
        m,n=src.info(layer_name(0,suf)).shape;r=ranks[typ]
        add('shared_base',f'@shared/{typ}/S',CODEC_Q4_ROW,req|shared|stream,(m,n),matrix_type=typ)
        add('residual_U',f'@shared/{typ}/U',CODEC_Q4_ROW,req|shared|stream,(m,r),matrix_type=typ)
        add('residual_Vt',f'@shared/{typ}/Vt',CODEC_Q4_ROW,req|shared|stream,(r,n),matrix_type=typ)
        for l in range(L):add('residual_core',layer_name(l,suf),CODEC_Q4_ROW,req|stream,(r,r),l,typ)
    if source_metadata:add('source_metadata','@source/gguf_header',CODEC_RAW,req|stream,resource_name='source.gguf.header')
    for name,loc in aux:add('aux_resource',loc,CODEC_RAW,req|stream,resource_name=name)
    return pages

def _stored_q4(x):
    parts=q4_parts(x);blob=q4_blob_from_parts(*parts);hat=dequantize_q4_rows(*parts);return blob,hat

def _residual(src,name,shared_hat):
    x=src.read_tensor(name);r=x.float() if x.dtype!=torch.float32 else x;r.sub_(shared_hat);return r

def _orth(x):return torch.linalg.qr(x,mode='reduced').Q

def fit_subspaces(src,L,suf,shared_hat,rank,oversample,seed,power_iters=0):
    m,n=shared_hat.shape;q=min(rank+oversample,m,n);yu=torch.zeros((m,q),dtype=torch.float32);yv=torch.zeros((n,q),dtype=torch.float32)
    for l in range(L):
        r=_residual(src,layer_name(l,suf),shared_hat);g1=torch.Generator().manual_seed(seed+l*2);g2=torch.Generator().manual_seed(seed+l*2+1);on=torch.randn((n,q),generator=g1);om=torch.randn((m,q),generator=g2);yu.add_(r@on);yv.add_(r.t()@om);del r,on,om
    u=_orth(yu)[:,:rank].contiguous();v=_orth(yv)[:,:rank].contiguous();del yu,yv
    for _ in range(power_iters):
        nu=torch.zeros_like(u);nv=torch.zeros_like(v)
        for l in range(L):
            r=_residual(src,layer_name(l,suf),shared_hat);nu.add_(r@(r.t()@u));nv.add_(r.t()@(r@v));del r
        u=_orth(nu)[:,:rank].contiguous();v=_orth(nv)[:,:rank].contiguous()
    return u,v

def convert(source_location,out_path,rank_spec='1024',kv_rank=96,oversample=16,power_iters=0,seed=2026,aux=None,baseline_bytes=DEFAULT_MISTRAL_Q4KM_BYTES,source_format='auto',preserve_source_metadata=True):
    aux=parse_aux(aux);src,fmt=_open_source(source_location,source_format);L=discover(src);ranks=parse_ranks(rank_spec,src,L);source_meta=fmt=='gguf' and preserve_source_metadata;plan=make_plan(src,L,ranks,aux,source_meta);lookup={(p['role'],p.get('matrix_type'),p.get('layer'),p.get('resource_name')):p for p in plan};hist=src.type_histogram() if hasattr(src,'type_histogram') else None
    manifest={'architecture':'mistral','conversion':'CoreShare-10X randomized shared-subspace initialization','source':{'location':str(source_location),'format':fmt,'full_source_file_required_locally':False,'type_histogram':hist},'coreshare':{'layers':L,'ranks':ranks,'equation':'W_layer = S_Q4 + U_Q4 @ C_layer_Q4 @ Vt_Q4','factor_codec':'Q4_ROW','shared_base_reference':'dequantized stored S_Q4','core_fit_reference':'dequantized stored U_Q4/Vt_Q4','kv_rank_planned':kv_rank},'source_metadata_preserved':source_meta,'aux_resources':[{'name':n,'source':loc} for n,loc in aux],'pages':plan}
    peak=0;source_meta_bytes=aux_bytes=0
    with LARCv2StreamWriter(out_path,len(plan),manifest) as wr:
        for n,role in [('model.embed_tokens.weight','embedding'),('lm_head.weight','lm_head')]:
            x=src.read_tensor(n);peak=max(peak,x.numel()*x.element_size());blob=q4_blob(x);p=lookup[(role,None,None,None)];wr.add_page(p['page_id'],p['codec_id'],p['flags'],blob,x.numel()*2);del x,blob
        x=src.read_tensor('model.norm.weight');p=lookup[('final_norm',None,None,None)];wr.add_page(p['page_id'],p['codec_id'],p['flags'],fp16_blob(x),x.numel()*2);del x
        for l in range(L):
            for suf,role in [('input_layernorm.weight','input_norm'),('post_attention_layernorm.weight','post_attention_norm')]:
                x=src.read_tensor(layer_name(l,suf));p=lookup[(role,None,l,None)];wr.add_page(p['page_id'],p['codec_id'],p['flags'],fp16_blob(x),x.numel()*2);del x
        for ti,(typ,suf) in enumerate(MATRIX_TYPES.items()):
            shared=None
            for l in range(L):
                x=src.read_tensor(layer_name(l,suf));peak=max(peak,x.numel()*x.element_size());
                if shared is None:shared=x.float() if x.dtype!=torch.float32 else x
                else:shared.add_(x);del x
            shared.div_(L);sblob,shat=_stored_q4(shared);del shared;p=lookup[('shared_base',typ,None,None)];wr.add_page(p['page_id'],p['codec_id'],p['flags'],sblob,shat.numel()*2);del sblob
            rnk=ranks[typ];u,v=fit_subspaces(src,L,suf,shat,rnk,oversample,seed+ti*10000,power_iters);ublob,uhat=_stored_q4(u);vblob,vthat=_stored_q4(v.t().contiguous());del u,v;p=lookup[('residual_U',typ,None,None)];wr.add_page(p['page_id'],p['codec_id'],p['flags'],ublob,uhat.numel()*2);p=lookup[('residual_Vt',typ,None,None)];wr.add_page(p['page_id'],p['codec_id'],p['flags'],vblob,vthat.numel()*2);del ublob,vblob
            vv=vthat.t().contiguous();gu=uhat.t()@uhat;gv=vv.t()@vv;eu=torch.eye(rnk)*torch.diagonal(gu).mean().clamp_min(1e-8)*1e-5;ev=torch.eye(rnk)*torch.diagonal(gv).mean().clamp_min(1e-8)*1e-5;gu=gu+eu;gv=gv+ev
            for l in range(L):
                r=_residual(src,layer_name(l,suf),shat);mid=uhat.t()@r@vv;left=torch.linalg.solve(gu,mid);core=torch.linalg.solve(gv,left.t()).t().contiguous();del r,mid,left;blob=q4_blob(core);p=lookup[('residual_core',typ,l,None)];wr.add_page(p['page_id'],p['codec_id'],p['flags'],blob,core.numel()*2);del core,blob;gc.collect()
            del shat,uhat,vthat,vv,gu,gv,eu,ev;gc.collect()
        if source_meta:
            data=src.raw_source_metadata();source_meta_bytes=len(data);p=lookup[('source_metadata',None,None,'source.gguf.header')];wr.add_page(p['page_id'],p['codec_id'],p['flags'],data,len(data));del data
        for name,loc in aux:
            data=_read_aux(loc);aux_bytes+=len(data);p=lookup[('aux_resource',None,None,name)];wr.add_page(p['page_id'],p['codec_id'],p['flags'],data,len(data));del data
    final=Path(out_path).stat().st_size;limit=baseline_bytes//10
    return {'output':str(out_path),'source_format':fmt,'source_type_histogram':hist,'representation':'CoreShare-10X','ranks':ranks,'planned_kv_rank':kv_rank,'page_count':len(plan),'source_bytes_fetched_or_read':src.bytes_fetched,'largest_single_source_tensor_bytes':peak,'source_metadata_bytes':source_meta_bytes,'aux_resource_bytes':aux_bytes,'final_larc_bytes':final,'baseline_bytes_for_file_gate':baseline_bytes,'ten_x_max_integer_file_bytes':limit,'passes_10x_file_gate':final<=limit,'file_reduction_x':baseline_bytes/final,'conversion_peak_memory_measured':False,'note':'Randomized shared-subspace fitting is bounded by one residual tensor plus U/V/sketch/core workspaces; exact process peak is not yet measured.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--source-format',choices=['auto','gguf','safetensors'],default='auto');ap.add_argument('--output',required=True);ap.add_argument('--ranks',default='1024');ap.add_argument('--kv-rank',type=int,default=96);ap.add_argument('--oversample',type=int,default=16);ap.add_argument('--power-iters',type=int,default=0);ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--aux',action='append',default=[]);ap.add_argument('--baseline-bytes',type=int,default=DEFAULT_MISTRAL_Q4KM_BYTES);ap.add_argument('--report');a=ap.parse_args();rep=convert(a.source,a.output,a.ranks,a.kv_rank,a.oversample,a.power_iters,a.seed,a.aux,a.baseline_bytes,a.source_format);text=json.dumps(rep,indent=2)+'\n';print(text,end='');
    if a.report:Path(a.report).write_text(text)
if __name__=='__main__':main()
