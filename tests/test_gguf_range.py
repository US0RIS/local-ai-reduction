from __future__ import annotations
import json,struct
from pathlib import Path
import numpy as np
import torch
from larc.gguf_range import GGUFRangeFile,MistralGGUFSource,GGML_F32,GGML_F16,GGML_Q4_K,GGML_Q5_K,GGML_Q6_K
from larc.paged_container import LARCv2File
from tools.stream_softshare_convert import MATRIX_TYPES,convert

ALIGN=32

def _align(n,a=ALIGN):return ((n+a-1)//a)*a
def _s(s:str):
 b=s.encode();return struct.pack('<Q',len(b))+b

def _write_gguf(path:Path,tensors:list[tuple[str,tuple[int,...],int,bytes]],arch='mistral'):
    # Tensor ne tuples use GGUF order (fastest dimension first).
    data=bytearray();entries=[]
    for name,ne,typ,raw in tensors:
        off=_align(len(data));data+=b'\0'*(off-len(data));entries.append((name,ne,typ,off));data+=raw
    h=bytearray(b'GGUF'+struct.pack('<IQQ',3,len(tensors),2))
    h+=_s('general.alignment')+struct.pack('<I',4)+struct.pack('<I',ALIGN)
    h+=_s('general.architecture')+struct.pack('<I',8)+_s(arch)
    for name,ne,typ,off in entries:
        h+=_s(name)+struct.pack('<I',len(ne))+b''.join(struct.pack('<Q',x) for x in ne)+struct.pack('<IQ',typ,off)
    data_start=_align(len(h));blob=h+b'\0'*(data_start-len(h))+data
    # GGUFRangeFile buffers a 1 MiB header chunk. Pad synthetic fixtures past it.
    if len(blob)<(1<<20)+1:blob+=b'\0'*((1<<20)+1-len(blob))
    path.write_bytes(blob)

def _k_scales():return bytes([1,1,1,1,0,0,0,0,1,1,1,1])
def _q4_block():return struct.pack('<e',1.0)+struct.pack('<e',0.0)+_k_scales()+bytes([0x21])*128
def _q5_block():return struct.pack('<e',1.0)+struct.pack('<e',0.0)+_k_scales()+bytes([0xFF])*32+bytes([0x21])*128
def _q6_block():return bytes(128)+bytes(64)+bytes([1])*16+struct.pack('<e',1.0)

def test_known_q4_q5_q6_k_dequantization(tmp_path):
    f=tmp_path/'kquants.gguf';f32=np.array([1.25,-2.5,3.0,0.0],dtype='<f4')
    _write_gguf(f,[('q4',(256,),GGML_Q4_K,_q4_block()),('q5',(256,),GGML_Q5_K,_q5_block()),('q6',(256,),GGML_Q6_K,_q6_block()),('f32',(4,),GGML_F32,f32.tobytes())])
    g=GGUFRangeFile(f);assert g.version==3;assert g.metadata['general.alignment']==32;assert g.metadata['general.architecture']=='mistral';assert g.type_histogram()=={'Q4_K':1,'Q5_K':1,'Q6_K':1,'F32':1}
    q4=g.read_tensor('q4').numpy();q5=g.read_tensor('q5').numpy();q6=g.read_tensor('q6').numpy();got=g.read_tensor('f32').numpy()
    expected4=np.tile(np.r_[np.ones(32),np.full(32,2,dtype=np.float32)],4)
    expected5=np.tile(np.r_[np.full(32,17,dtype=np.float32),np.full(32,18,dtype=np.float32)],4)
    assert np.array_equal(q4,expected4);assert np.array_equal(q5,expected5);assert np.array_equal(q6,np.full(256,-32,dtype=np.float32));assert np.array_equal(got,f32)
    assert g.raw_header_bytes().startswith(b'GGUF')

def _tiny_mistral_gguf(path:Path,layers=1):
    torch.manual_seed(31);d=16;kv=8;ff=24;vocab=20;t=[]
    def add(name,x):
        x=x.detach().to(torch.float16).contiguous();shape=tuple(reversed(x.shape));t.append((name,shape,GGML_F16,x.numpy().tobytes()))
    add('token_embd.weight',torch.randn(vocab,d)*.1);add('output.weight',torch.randn(vocab,d)*.1);add('output_norm.weight',torch.ones(d))
    gg={'q_proj':'attn_q.weight','k_proj':'attn_k.weight','v_proj':'attn_v.weight','o_proj':'attn_output.weight','gate_proj':'ffn_gate.weight','up_proj':'ffn_up.weight','down_proj':'ffn_down.weight'}
    shapes={'q_proj':(d,d),'k_proj':(kv,d),'v_proj':(kv,d),'o_proj':(d,d),'gate_proj':(ff,d),'up_proj':(ff,d),'down_proj':(d,ff)}
    shared={k:torch.randn(*sh)*.1 for k,sh in shapes.items()}
    for l in range(layers):
        add(f'blk.{l}.attn_norm.weight',torch.ones(d)+.01*l);add(f'blk.{l}.ffn_norm.weight',torch.ones(d)-.01*l)
        for typ,name in gg.items():
            m,n=shapes[typ];A=torch.randn(m,2)*.03;B=torch.randn(2,n)*.03;add(f'blk.{l}.{name}',shared[typ]+A@B)
    _write_gguf(path,t);return len(t)

def test_gguf_source_maps_mistral_names_and_converts(tmp_path):
    src_path=tmp_path/'tiny-mistral.gguf';tensor_count=_tiny_mistral_gguf(src_path);src=MistralGGUFSource(src_path)
    assert tensor_count==12;assert src.type_histogram()=={'F16':12};assert src.info('model.layers.0.self_attn.k_proj.weight').shape==(8,16)
    out=tmp_path/'from-gguf.larc';baseline=10_000_000;report=convert(src_path,out,{k:2 for k in MATRIX_TYPES},oversample=2,niter=1,seed=9,baseline_bytes=baseline,source_format='gguf')
    expected_pages=3+2+len(MATRIX_TYPES)*3+1
    assert report['source_format']=='gguf';assert report['source_type_histogram']=={'F16':12};assert report['page_count']==expected_pages==27;assert report['source_metadata_bytes']>0;assert report['passes_10x_file_gate'];assert report['file_reduction_x']>=10
    with LARCv2File(out) as f:
        assert len(f.pages)==27;assert f.manifest['source']['format']=='gguf';assert f.manifest['source_metadata_preserved'] is True
        meta=[p for p in f.manifest['pages'] if p['role']=='source_metadata'];assert len(meta)==1;assert bytes(f.page_view(meta[0]['page_id'],verify=True)).startswith(b'GGUF')
        for page_id in f.pages:
            v=f.page_view(page_id,verify=True);assert len(v)==f.pages[page_id].stored_length;del v
