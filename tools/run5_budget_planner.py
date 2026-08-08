#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path

# Exact size from the Hugging Face raw LFS pointer for
# mistral-7b-v0.1.Q4_K_M.gguf (SHA256 ce6253d2...bcbbe40c).
BASELINE_Q4KM_BYTES=4_368_438_912
BASELINE_SHA256='ce6253d2e91adea0c35924b38411b0434fa18fcb90c52980ce68187dbcbbe40c'
CFG={'model':'mistralai/Mistral-7B-v0.1','layers':32,'hidden':4096,'intermediate':14336,'heads':32,'kv_heads':8,'head_dim':128,'vocab':32000,'tied_embeddings':False,'sliding_window':4096}

def q4row_bytes(rows:int,cols:int)->int:return rows*((cols+1)//2+2)
def matrices():
 d=CFG['hidden'];ff=CFG['intermediate'];kv=CFG['kv_heads']*CFG['head_dim'];return {'q_proj':(d,d),'k_proj':(kv,d),'v_proj':(kv,d),'o_proj':(d,d),'gate_proj':(ff,d),'up_proj':(ff,d),'down_proj':(d,ff)}
def weight_bytes(rank:int):
 d=CFG['hidden'];L=CFG['layers'];v=CFG['vocab'];mats=matrices();common=(1 if CFG['tied_embeddings'] else 2)*q4row_bytes(v,d);shared=sum(q4row_bytes(m,n) for m,n in mats.values());residual=L*sum(q4row_bytes(m,rank)+q4row_bytes(rank,n) for m,n in mats.values());norms=(2*L*d+d)*2;return {'total':common+shared+residual+norms,'embedding_head':common,'shared_bases':shared,'layer_residuals':residual,'norms_fp16':norms}
def fp16_kv_bytes(seq:int|None=None)->int:
 seq=CFG['sliding_window'] if seq is None else seq;return CFG['layers']*seq*CFG['kv_heads']*CFG['head_dim']*4
def latent_kv_bytes(rank:int,seq:int|None=None):
 seq=CFG['sliding_window'] if seq is None else seq;L=CFG['layers'];h=CFG['kv_heads'];d=CFG['head_dim'];units=L*seq*h;coeff=units*2*(math.ceil(rank*2/8)+2);rows=L*h*2*rank;bases=rows*(math.ceil(d/2)+2);metrics=L*h*2*rank*rank*2;return {'total':coeff+bases+metrics,'coeff_and_fp8_metadata':coeff,'q4_bases':bases,'fp16_inverse_grams':metrics}
def candidate(weight_rank:int,kv_rank:int,breakdown:bool=True):
 w=weight_bytes(weight_rank);k=latent_kv_bytes(kv_rank);bkv=fp16_kv_bytes();base=BASELINE_Q4KM_BYTES+bkv;compressed=w['total']+k['total'];out={'weight_residual_rank':weight_rank,'kv_latent_rank':kv_rank,'weight_bytes':w['total'],'weight_file_reduction_vs_exact_q4km':BASELINE_Q4KM_BYTES/w['total'],'kv_bytes_at_4096':k['total'],'kv_reduction_vs_fp16_at_4096':bkv/k['total'],'weights_plus_kv_reduction_at_4096':base/compressed,'equal_common_scratch_headroom_bytes_before_falling_below_10x':max(0,(base-10*compressed)/9)}
 if breakdown:out['breakdown']={'weights':w,'kv':k}
 return out
def build():
 return {'run':5,'strategy':'soft cross-layer sharing: one shared Q4 base per matrix type + depth-specific low-rank Q4 residuals; Q2/FP8 latent KV retained','target':'10x, not 50x','reference_model':CFG,'baseline':{'gguf':'TheBloke/Mistral-7B-v0.1-GGUF / mistral-7b-v0.1.Q4_K_M.gguf','exact_bytes':BASELINE_Q4KM_BYTES,'sha256':BASELINE_SHA256,'exact_size_source':'Hugging Face raw LFS pointer','fp16_kv_bytes_at_sliding_window_4096':fp16_kv_bytes()},'rank_sweep':[candidate(r,64,False) for r in (64,80,96,112,128,136,140,144)],'recommended_core':candidate(96,64,True),'quality_max_candidate':candidate(128,72,True),'design_note':'Start at rank96 because 96/4096 equals the 2.34% relative rank of the successful rank3/d128 controlled probe. Spend remaining 10x file budget only on validation-selected rescue ranks/pages; do not force harder sharing.'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='benchmarks/run5_mistral7b_budget.json');a=ap.parse_args();text=json.dumps(build(),indent=2)+'\n';Path(a.output).write_text(text);print(text,end='')
if __name__=='__main__':main()
