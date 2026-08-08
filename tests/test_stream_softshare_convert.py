from __future__ import annotations
import json,struct
from pathlib import Path
import torch
from larc.paged_container import LARCv2File
from larc.safetensors_range import SafeTensorFile,ShardedSafeTensorSource
from tools.stream_softshare_convert import MATRIX_TYPES,convert


def _write_safetensors(path:Path,tensors:dict[str,torch.Tensor]):
    entries={};chunks=[];off=0
    for name,t in sorted(tensors.items()):
        x=t.detach().to(torch.float16).contiguous();b=x.numpy().tobytes();entries[name]={'dtype':'F16','shape':list(x.shape),'data_offsets':[off,off+len(b)]};chunks.append(b);off+=len(b)
    h=json.dumps(entries,separators=(',',':')).encode();h+=b' ' *((8-len(h)%8)%8)
    with open(path,'wb') as f:f.write(struct.pack('<Q',len(h)));f.write(h);[f.write(b) for b in chunks]

def _source(tmp_path:Path,layers=2):
    torch.manual_seed(19);d=16;kv=8;ff=24;vocab=20;all_t={
      'model.embed_tokens.weight':torch.randn(vocab,d)*.1,'lm_head.weight':torch.randn(vocab,d)*.1,'model.norm.weight':torch.ones(d)}
    shapes={'q_proj':(d,d),'k_proj':(kv,d),'v_proj':(kv,d),'o_proj':(d,d),'gate_proj':(ff,d),'up_proj':(ff,d),'down_proj':(d,ff)}
    shared={k:torch.randn(*sh)*.1 for k,sh in shapes.items()}
    for l in range(layers):
        all_t[f'model.layers.{l}.input_layernorm.weight']=torch.ones(d)+.01*l
        all_t[f'model.layers.{l}.post_attention_layernorm.weight']=torch.ones(d)-.01*l
        for typ,suf in MATRIX_TYPES.items():
            m,n=shapes[typ];A=torch.randn(m,2)*.03;B=torch.randn(2,n)*.03;all_t[f'model.layers.{l}.{suf}']=shared[typ]+A@B
    shards=[{},{}];weight_map={}
    for i,(name,t) in enumerate(sorted(all_t.items())):shards[i%2][name]=t;weight_map[name]=f'model-{i%2+1:05d}-of-00002.safetensors'
    for i,s in enumerate(shards):_write_safetensors(tmp_path/f'model-{i+1:05d}-of-00002.safetensors',s)
    index=tmp_path/'model.safetensors.index.json';index.write_text(json.dumps({'metadata':{},'weight_map':weight_map}));return index,all_t

def test_local_safetensors_tensor_range_reader(tmp_path):
    p=tmp_path/'one.safetensors';x=torch.arange(24,dtype=torch.float16).reshape(4,6);_write_safetensors(p,{'x':x});f=SafeTensorFile(p);got=f.read_tensor('x');assert torch.equal(got,x);assert f.bytes_fetched < p.stat().st_size+64

def test_sharded_softshare_conversion_is_streamed_self_contained_and_valid(tmp_path):
    index,src_tensors=_source(tmp_path);src=ShardedSafeTensorSource(index);assert 'model.layers.0.self_attn.q_proj.weight' in set(src.names())
    config=tmp_path/'config.json';config.write_bytes(b'{"model_type":"mistral"}\n');tokenizer=tmp_path/'tokenizer.json';tokenizer.write_bytes(b'{"version":"1.0","model":{}}\n')
    out=tmp_path/'tiny.larc';synthetic_baseline=10_000_000
    report=convert(index,out,{k:2 for k in MATRIX_TYPES},oversample=2,niter=1,seed=7,aux=[('config.json',str(config)),('tokenizer.json',str(tokenizer))],baseline_bytes=synthetic_baseline)
    expected_pages=3+2*2+len(MATRIX_TYPES)*(1+2*2)+2
    assert report['source_format']=='safetensors'
    assert report['page_count']==expected_pages==44
    assert report['bounded_local_source_file_required'] is False
    assert report['largest_single_source_tensor_bytes']==max(t.numel()*2 for t in src_tensors.values())
    assert report['conversion_peak_explicit_tensor_lower_bound_excluding_internal_svd_and_q4_temporaries_bytes']>=report['largest_single_source_tensor_bytes']
    assert report['internal_svd_workspace_peak_measured'] is False
    assert report['internal_q4_temporary_peak_measured'] is False
    assert report['aux_resource_count']==2
    assert report['aux_resource_bytes']==config.stat().st_size+tokenizer.stat().st_size
    assert report['final_larc_bytes']==out.stat().st_size
    assert report['ten_x_max_integer_file_bytes']==synthetic_baseline//10
    assert report['passes_10x_file_gate'] is True
    assert report['file_reduction_x']>=10.0
    with LARCv2File(out) as f:
        assert len(f.pages)==expected_pages
        assert f.manifest['softshare']['equation']=='W_layer = shared_Q4 + A_Q4 @ B_Q4'
        assert f.manifest['softshare']['factor_codec']=='Q4_ROW'
        assert f.manifest['softshare']['residual_fit_reference']=='dequantized stored shared_Q4'
        aux_by_name={p['resource_name']:p['page_id'] for p in f.manifest['pages'] if p['role']=='aux_resource'}
        assert bytes(f.page_view(aux_by_name['config.json'],verify=True))==config.read_bytes()
        assert bytes(f.page_view(aux_by_name['tokenizer.json'],verify=True))==tokenizer.read_bytes()
        for page_id in sorted(f.pages):
            v=f.page_view(page_id,verify=True);assert len(v)==f.pages[page_id].stored_length;del v
