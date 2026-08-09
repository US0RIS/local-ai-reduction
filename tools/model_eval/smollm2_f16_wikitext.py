#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys,time
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
from remote_gguf import HTTPRangeSource,inspect_remote_gguf
URL='https://registry.ollama.ai/v2/library/smollm2/blobs/sha256:f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57'
EXPECTED_SHA='f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57'
EXPECTED_BYTES=270_885_952
WIKI_REV='b08601e04326c79dfdd32d625aee71d232d685c3'
WIKI_URL=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{WIKI_REV}/wikitext-2-raw-v1/validation-00000-of-00001.parquet?download=true'
H=576;KV=192;FF=1536;L=30;VOC=49_152

def config_model():
    from transformers import LlamaConfig,LlamaForCausalLM
    cfg=LlamaConfig(vocab_size=VOC,hidden_size=H,intermediate_size=FF,num_hidden_layers=L,num_attention_heads=9,num_key_value_heads=3,max_position_embeddings=8192,rms_norm_eps=1e-5,rope_theta=100000.0,attention_bias=False,attention_dropout=0.0,hidden_act='silu',tie_word_embeddings=True,bos_token_id=0,eos_token_id=0)
    return LlamaForCausalLM(cfg)

def tensor_nelems(shape):
    n=1
    for x in shape:n*=int(x)
    return n

def fetch_tensor(src,t):
    # GGML type 0=F32, 1=F16. Reject anything else: this must remain the exact F16 teacher.
    if int(t.ggml_type)==1:dt='<f2';item=2
    elif int(t.ggml_type)==0:dt='<f4';item=4
    else:raise ValueError((t.name,t.ggml_type,'teacher tensor is not F16/F32'))
    need=tensor_nelems(t.shape)*item;raw=bytearray();off=0
    while off<need:
        m=min(4<<20,need-off);raw+=src.read(t.absolute_offset+off,m);off+=m
    x=np.frombuffer(raw,dtype=dt).astype(np.float32)
    if len(t.shape)==1:return torch.from_numpy(x.copy()),need
    if len(t.shape)!=2:raise ValueError((t.name,t.shape))
    return torch.from_numpy(x.reshape(int(t.shape[1]),int(t.shape[0])).copy()),need

def load_teacher():
    idx=inspect_remote_gguf(URL,chunk_bytes=256<<10,max_header_bytes=32<<20,max_response_bytes=4<<20)
    if idx.remote_size!=EXPECTED_BYTES:raise RuntimeError(('teacher byte size mismatch',idx.remote_size,EXPECTED_BYTES))
    src=HTTPRangeSource(URL,max_response_bytes=4<<20);tm={t.name:t for t in idx.tensors};m=config_model();sd=m.state_dict();loaded=0;types={}
    mp={'model.embed_tokens.weight':'token_embd.weight','model.norm.weight':'output_norm.weight'}
    for l in range(L):
        mp.update({f'model.layers.{l}.self_attn.q_proj.weight':f'blk.{l}.attn_q.weight',f'model.layers.{l}.self_attn.k_proj.weight':f'blk.{l}.attn_k.weight',f'model.layers.{l}.self_attn.v_proj.weight':f'blk.{l}.attn_v.weight',f'model.layers.{l}.self_attn.o_proj.weight':f'blk.{l}.attn_output.weight',f'model.layers.{l}.mlp.gate_proj.weight':f'blk.{l}.ffn_gate.weight',f'model.layers.{l}.mlp.up_proj.weight':f'blk.{l}.ffn_up.weight',f'model.layers.{l}.mlp.down_proj.weight':f'blk.{l}.ffn_down.weight',f'model.layers.{l}.input_layernorm.weight':f'blk.{l}.attn_norm.weight',f'model.layers.{l}.post_attention_layernorm.weight':f'blk.{l}.ffn_norm.weight'})
    with torch.no_grad():
        for dst,srcn in mp.items():
            t=tm[srcn];types[int(t.ggml_type)]=types.get(int(t.ggml_type),0)+1;x,n=fetch_tensor(src,t);loaded+=n
            if tuple(sd[dst].shape)!=tuple(x.shape):raise ValueError((dst,tuple(sd[dst].shape),tuple(x.shape)))
            sd[dst].copy_(x)
        if 'output.weight' in tm:
            t=tm['output.weight'];types[int(t.ggml_type)]=types.get(int(t.ggml_type),0)+1;x,n=fetch_tensor(src,t);loaded+=n;sd['lm_head.weight'].copy_(x)
        else:sd['lm_head.weight'].copy_(sd['model.embed_tokens.weight'])
    m.load_state_dict(sd);m.tie_weights();m.eval();m.config.use_cache=False
    for p in m.parameters():p.requires_grad=False
    return m,{'url':URL,'remote_bytes':idx.remote_size,'header_bytes_fetched':idx.header_bytes_fetched,'tensor_payload_bytes_loaded':loaded,'gguf_version':idx.version,'tensor_count':len(idx.tensors),'mapped_tensor_type_counts':types,'expected_blob_sha256':EXPECTED_SHA}

def wiki_text():
    import requests,pyarrow.parquet as pq
    r=requests.get(WIKI_URL,timeout=180);r.raise_for_status();table=pq.read_table(BytesIO(r.content),columns=['text']);rows=[str(x) if x is not None else '' for x in table.column('text').to_pylist()];rows=[x for x in rows if x.strip()];return '\n'.join(rows)

@torch.inference_mode()
def nll(model,ids,seq=256,batch_chunks=8,max_tokens=None):
    if max_tokens is not None:ids=ids[:max_tokens]
    chunks=[]
    for st in range(0,len(ids)-1,seq):
        z=ids[st:st+seq+1]
        if len(z)>=2:chunks.append(z)
    total=0.;count=0
    # Full equal-sized chunks can batch without changing reset boundaries. Handle tail serially.
    i=0
    while i<len(chunks):
        n=min(batch_chunks,len(chunks)-i);group=chunks[i:i+n]
        same=len({len(z) for z in group})==1
        if same:
            x=torch.tensor([z[:-1] for z in group],dtype=torch.long);y=torch.tensor([z[1:] for z in group],dtype=torch.long);log=model(x,use_cache=False).logits.float();total+=float(F.cross_entropy(log.reshape(-1,log.shape[-1]),y.reshape(-1),reduction='sum'));count+=int(y.numel());i+=n
        else:
            z=chunks[i];x=torch.tensor(z[:-1],dtype=torch.long)[None];y=torch.tensor(z[1:],dtype=torch.long);log=model(x,use_cache=False).logits[0].float();total+=float(F.cross_entropy(log,y,reduction='sum'));count+=len(y);i+=1
    if not count:raise ValueError('no NLL tokens')
    return total/count,count

def main():
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())));t0=time.time();model,meta=load_teacher();from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct',use_fast=True);text=wiki_text();ids=tok(text,add_special_tokens=False)['input_ids'];short,ns=nll(model,ids,max_tokens=3072);full,nf=nll(model,ids);out={'teacher':meta,'wikitext':{'revision':WIKI_REV,'joined_sha256':hashlib.sha256(text.encode()).hexdigest(),'characters':len(text),'token_count':len(ids)},'nll':{'first_3072_reset256':short,'first_3072_tokens_scored':ns,'full_reset256':full,'full_tokens_scored':nf},'candidate_threshold_for_1p10':1.10*full,'elapsed_sec':time.time()-t0};Path('results').mkdir(exist_ok=True);Path('results/smollm2_f16_teacher_wikitext.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
