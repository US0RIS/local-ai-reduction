#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path

BASELINE_Q4KM_BYTES=4_368_438_912
BASELINE_SHA256='ce6253d2e91adea0c35924b38411b0434fa18fcb90c52980ce68187dbcbbe40c'
TARGET_MAX_FILE_BYTES=BASELINE_Q4KM_BYTES//10
ALIGNMENT=4096
Q4_PAGE_HEADER_BYTES=16
AUX_DEPLOYMENT_RESERVE_BYTES=4*1024*1024
CANONICAL_SOURCE_INDEX='https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/model.safetensors.index.json'
CFG={'model':'mistralai/Mistral-7B-v0.1','layers':32,'hidden':4096,'intermediate':14336,'heads':32,'kv_heads':8,'head_dim':128,'vocab':32000,'tied_embeddings':False,'sliding_window':4096}

def align_up(n:int,a:int=ALIGNMENT)->int:return ((n+a-1)//a)*a
def q4row_bytes(rows:int,cols:int)->int:return rows*((cols+1)//2+2)
def matrices():
 d=CFG['hidden'];ff=CFG['intermediate'];kv=CFG['kv_heads']*CFG['head_dim'];return {'q_proj':(d,d),'k_proj':(kv,d),'v_proj':(kv,d),'o_proj':(d,d),'gate_proj':(ff,d),'up_proj':(ff,d),'down_proj':(d,ff)}
MATRIX_SUFFIX={'q_proj':'self_attn.q_proj.weight','k_proj':'self_attn.k_proj.weight','v_proj':'self_attn.v_proj.weight','o_proj':'self_attn.o_proj.weight','gate_proj':'mlp.gate_proj.weight','up_proj':'mlp.up_proj.weight','down_proj':'mlp.down_proj.weight'}

def weight_bytes(rank:int):
 d=CFG['hidden'];L=CFG['layers'];v=CFG['vocab'];mats=matrices();common=(1 if CFG['tied_embeddings'] else 2)*q4row_bytes(v,d);shared=sum(q4row_bytes(m,n) for m,n in mats.values());residual=L*sum(q4row_bytes(m,rank)+q4row_bytes(rank,n) for m,n in mats.values());norms=(2*L*d+d)*2;return {'total':common+shared+residual+norms,'embedding_head':common,'shared_bases':shared,'layer_residuals':residual,'norms_fp16':norms}

def _page_plan(rank:int):
 pages=[]
 def add(role,name,codec,flags,shape,layer=None,matrix_type=None):
  d={'page_id':len(pages)+1,'role':role,'source_name':name,'codec_id':codec,'flags':flags,'shape':list(shape)}
  if layer is not None:d['layer']=layer
  if matrix_type is not None:d['matrix_type']=matrix_type
  pages.append(d)
 req,shared,stream=1,2,8
 d=CFG['hidden'];v=CFG['vocab'];L=CFG['layers']
 add('embedding','model.embed_tokens.weight',1,req|stream,(v,d));add('lm_head','lm_head.weight',1,req|stream,(v,d));add('final_norm','model.norm.weight',0,req|stream,(d,))
 for l in range(L):
  add('input_norm',f'model.layers.{l}.input_layernorm.weight',0,req|stream,(d,),l);add('post_attention_norm',f'model.layers.{l}.post_attention_layernorm.weight',0,req|stream,(d,),l)
 for typ,(m,n) in matrices().items():
  add('shared_base',f'@shared/{typ}',1,req|shared|stream,(m,n),matrix_type=typ)
  for l in range(L):
   src=f'model.layers.{l}.{MATRIX_SUFFIX[typ]}';add('residual_A',src,1,req|stream,(m,rank),l,typ);add('residual_B',src,1,req|stream,(rank,n),l,typ)
 return pages

def larc_weight_file_estimate(rank:int):
 pages=_page_plan(rank);ranks={k:rank for k in matrices()};manifest={'architecture':'mistral','conversion':'SoftShare-10X two-pass initialization','source':{'safetensors_index':CANONICAL_SOURCE_INDEX,'full_source_file_required_locally':False},'softshare':{'layers':CFG['layers'],'ranks':ranks,'equation':'W_layer = shared_Q4 + A_Q4 @ B_Q4','residual_fit_reference':'dequantized stored shared_Q4','recovery_applied':False},'pages':pages,'larc':{'major':2,'minor':0,'alignment':ALIGNMENT}}
 mb=json.dumps(manifest,separators=(',',':'),sort_keys=True).encode();table_off=((64+len(mb)+63)//64)*64;data_off=align_up(table_off+64*len(pages));off=data_off;payload=0
 for p in pages:
  m=math.prod(p['shape'])
  if p['codec_id']==0:plen=m*2
  else:
   rows,cols=p['shape'];plen=Q4_PAGE_HEADER_BYTES+q4row_bytes(rows,cols)
  payload+=plen;off=align_up(off)+plen
 return {'serialized_weight_file_bytes':off,'page_count':len(pages),'manifest_bytes':len(mb),'stored_payload_bytes':payload,'container_alignment_and_metadata_bytes':off-payload,'with_aux_deployment_reserve_bytes':off+AUX_DEPLOYMENT_RESERVE_BYTES,'file_reduction_with_aux_reserve_x':BASELINE_Q4KM_BYTES/(off+AUX_DEPLOYMENT_RESERVE_BYTES),'bytes_remaining_to_10x_file_limit_after_aux_reserve':TARGET_MAX_FILE_BYTES-(off+AUX_DEPLOYMENT_RESERVE_BYTES)}

def fp16_kv_bytes(seq:int|None=None)->int:
 seq=CFG['sliding_window'] if seq is None else seq;return CFG['layers']*seq*CFG['kv_heads']*CFG['head_dim']*4

def latent_kv_bytes(rank:int,seq:int|None=None):
 seq=CFG['sliding_window'] if seq is None else seq;L=CFG['layers'];h=CFG['kv_heads'];d=CFG['head_dim'];units=L*seq*h;coeff=units*2*(math.ceil(rank*2/8)+2);rows=L*h*2*rank;bases=rows*(math.ceil(d/2)+2);metrics=L*h*2*rank*rank*2;return {'total':coeff+bases+metrics,'coeff_and_fp8_metadata':coeff,'q4_bases':bases,'fp16_inverse_grams':metrics}

def candidate(weight_rank:int,kv_rank:int,breakdown:bool=True):
 w=weight_bytes(weight_rank);file_est=larc_weight_file_estimate(weight_rank);k=latent_kv_bytes(kv_rank);bkv=fp16_kv_bytes();base=BASELINE_Q4KM_BYTES+bkv;compressed=w['total']+k['total'];out={'weight_residual_rank':weight_rank,'kv_latent_rank':kv_rank,'resident_weight_tensor_bytes':w['total'],'resident_weight_tensor_reduction_vs_q4km_file_x':BASELINE_Q4KM_BYTES/w['total'],'larc_weight_file':file_est,'kv_bytes_at_4096':k['total'],'kv_reduction_vs_fp16_at_4096':bkv/k['total'],'weights_plus_kv_tensor_reduction_at_4096':base/compressed,'equal_common_scratch_headroom_bytes_before_falling_below_10x':max(0,(base-10*compressed)/9)}
 if breakdown:out['breakdown']={'weights':w,'kv':k}
 return out

def build():
 return {'run':5,'strategy':'soft cross-layer sharing: one shared Q4 base per matrix type + depth-specific low-rank Q4 residuals; Q2/FP8 latent KV retained','target':'10x, not 50x','reference_model':CFG,'baseline':{'gguf':'TheBloke/Mistral-7B-v0.1-GGUF / mistral-7b-v0.1.Q4_K_M.gguf','exact_bytes':BASELINE_Q4KM_BYTES,'sha256':BASELINE_SHA256,'exact_size_source':'Hugging Face raw LFS pointer','ten_x_max_integer_file_bytes':TARGET_MAX_FILE_BYTES,'fp16_kv_bytes_at_sliding_window_4096':fp16_kv_bytes()},'planning_reserves':{'aux_tokenizer_config_bytes':AUX_DEPLOYMENT_RESERVE_BYTES,'note':'Conservative planning reserve only. Final 10x gate uses actual complete serialized .larc bytes.'},'rank_sweep':[candidate(r,64,False) for r in (64,80,96,112,128,136,140,144)],'recommended_core':candidate(96,64,True),'quality_max_candidate':candidate(128,72,True),'design_note':'Start at rank96 because 96/4096 equals the 2.34% relative rank of the successful rank3/d128 controlled probe. Spend the remaining complete-file/runtime 10x budget on validation-selected rescue ranks/pages; do not force harder sharing.'}

def _semantic_equal(a,b,path='root'):
 if isinstance(a,dict) and isinstance(b,dict):
  if set(a)!=set(b):raise AssertionError(f'{path}: key mismatch {set(a)^set(b)}')
  for k in a:_semantic_equal(a[k],b[k],f'{path}.{k}')
 elif isinstance(a,list) and isinstance(b,list):
  if len(a)!=len(b):raise AssertionError(f'{path}: length {len(a)} != {len(b)}')
  for i,(x,y) in enumerate(zip(a,b)):_semantic_equal(x,y,f'{path}[{i}]')
 elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
  if not math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-9):raise AssertionError(f'{path}: {a} != {b}')
 elif a!=b:raise AssertionError(f'{path}: {a!r} != {b!r}')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='benchmarks/run5_mistral7b_budget.json');ap.add_argument('--check',action='store_true');a=ap.parse_args();obj=build();p=Path(a.output)
 if a.check:
  _semantic_equal(json.loads(p.read_text()),obj);print(f'OK {p}')
 else:
  text=json.dumps(obj,indent=2)+'\n';p.write_text(text);print(text,end='')
if __name__=='__main__':main()
