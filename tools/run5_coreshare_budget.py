#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
BASE=4_368_438_912;LIMIT=BASE//10;AUX=4*1024*1024;ALIGN=4096
CFG={'layers':32,'hidden':4096,'intermediate':14336,'kv_heads':8,'head_dim':128,'vocab':32000,'sliding_window':4096}
MATS={'q_proj':(4096,4096),'k_proj':(1024,4096),'v_proj':(1024,4096),'o_proj':(4096,4096),'gate_proj':(14336,4096),'up_proj':(14336,4096),'down_proj':(4096,14336)}
SOURCE='https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf'
def q4(m,n):return m*((n+1)//2+2)
def align(n):return ((n+ALIGN-1)//ALIGN)*ALIGN
def weight_bytes(rank):
 d=CFG['hidden'];L=CFG['layers'];v=CFG['vocab'];total=2*q4(v,d)+(2*L*d+d)*2+sum(q4(m,n) for m,n in MATS.values())
 for m,n in MATS.values():
  r=min(rank,m,n);total+=q4(m,r)+q4(r,n)+L*q4(r,r)
 return total
def kv_bytes(rank):
 L=CFG['layers'];seq=CFG['sliding_window'];h=CFG['kv_heads'];d=CFG['head_dim'];units=L*seq*h;coeff=units*2*(math.ceil(rank*2/8)+2);rows=L*h*2*rank;bases=rows*(math.ceil(d/2)+2);metrics=L*h*2*rank*rank*2;return coeff+bases+metrics
def pages(rank):
 p=[]
 def add(role,shape=None):p.append((role,shape))
 add('embedding',(32000,4096));add('lm_head',(32000,4096));add('final_norm',(4096,))
 for _ in range(32):add('input_norm',(4096,));add('post_attention_norm',(4096,))
 for m,n in MATS.values():
  r=min(rank,m,n);add('shared_base',(m,n));add('U',(m,r));add('Vt',(r,n))
  for _ in range(32):add('core',(r,r))
 add('source_metadata',None);return p
def file_est(rank):
 ps=pages(rank);ranks={k:min(rank,*s) for k,s in MATS.items()};manifest={'architecture':'mistral','conversion':'CoreShare-10X randomized shared-subspace initialization','source':{'location':SOURCE,'format':'gguf','full_source_file_required_locally':False},'coreshare':{'layers':32,'ranks':ranks,'equation':'W_layer = shared_Q4 + U_Q4 @ C_layer_Q4 @ Vt_Q4','factor_codec':'Q4_ROW','residual_fit_reference':'dequantized stored shared_Q4'},'source_metadata_preserved':True,'page_count':len(ps),'larc':{'major':2,'minor':0,'alignment':ALIGN}}
 mb=json.dumps(manifest,separators=(',',':'),sort_keys=True).encode();off=align(((64+len(mb)+63)//64)*64+64*len(ps));payload=0
 for role,shape in ps:
  if role=='source_metadata':plen=0
  elif role.endswith('norm'):plen=math.prod(shape)*2
  else:
   m,n=shape;plen=16+q4(m,n)
  payload+=plen;off=align(off)+plen
 return {'serialized_weight_file_bytes_before_aux':off,'page_count':len(ps),'manifest_bytes':len(mb),'stored_payload_bytes_excluding_source_metadata':payload,'container_alignment_and_metadata_bytes':off-payload,'with_4mib_aux_reserve_bytes':off+AUX,'file_reduction_x':BASE/(off+AUX),'bytes_remaining_to_10x_file_limit':LIMIT-(off+AUX)}
def candidate(wr,kr):
 w=weight_bytes(wr);kv=kv_bytes(kr);base=BASE+32*4096*8*128*4;comp=w+kv;return {'weight_subspace_rank':wr,'kv_rank':kr,'resident_weight_tensor_bytes':w,'weight_tensor_reduction_x':BASE/w,'file':file_est(wr),'kv_bytes_4096':kv,'weights_plus_kv_tensor_reduction_x':base/comp,'equal_common_scratch_headroom_bytes_before_10x':max(0,(base-10*comp)/9)}
def build():return {'run':5,'strategy':'CoreShare-10X: W_l=S+U C_l V^T','baseline':{'q4_k_m_bytes':BASE,'ten_x_file_limit_bytes':LIMIT},'rank_sweep':[candidate(r,112) for r in (768,896,960,1024,1088,1120,1152)],'recommended_core':candidate(960,112),'higher_weight_capacity':candidate(1024,96),'design_note':'Use rank960/KV112 as the initial real-model point. It spends far more capacity on layer-specific residual structure than per-layer SoftShare while retaining complete-file and runtime headroom for validation-selected rescue.'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='benchmarks/run5_coreshare_mistral_budget.json');a=ap.parse_args();obj=build();text=json.dumps(obj,indent=2)+'\n';Path(a.output).write_text(text);print(text,end='')
if __name__=='__main__':main()
