#!/usr/bin/env python3
"""Reproduce Run-6 Wq weight-space geometry from public paLLM TS260Q8 chunks.

Fetches only chunks 000..003, SHA-256 verifies them, parses the complete Wq
weight tensor for all five layers, then measures singular-energy and adjacent
layer diagnostics. This is external-pretrained WEIGHT geometry, not activation
geometry and not a LARC conversion.
"""
from __future__ import annotations
import hashlib,json,math,struct,urllib.request
from pathlib import Path
import torch

COMMIT='ce6b82233bbff55a7876f09dcee35f5fa1f69535'
HASHES={
0:'4a7d25f7fe7d918d28894c9995229c5013f3f5b12cce0184692e1e51b4bdde96',
1:'f85edac2dd0179e4fd99c1cd79170ae4e37ca45ad4cdf6225f5d3e2c538a529e',
2:'823521ff257c14e41ee03f03d07bf86e3a5cbf4f107023270d5c8a481b1ea7a1',
3:'d81a42f61d76769493b692980f38378a14e9462f42db63564b0a037345ef7d18'}
RANKS=(4,8,16,24,32,48,64)

def fetch():
 out=bytearray();meta=[]
 for i in range(4):
  url=f'https://raw.githubusercontent.com/maddiedreese/paLLM/{COMMIT}/Src/q8_chunk_{i:03d}.bin'
  with urllib.request.urlopen(url,timeout=60) as r:b=r.read()
  h=hashlib.sha256(b).hexdigest()
  if len(b)!=16384 or h!=HASHES[i]:raise RuntimeError(f'chunk {i} integrity failed')
  out+=b;meta.append({'index':i,'bytes':len(b),'sha256':h})
 return bytes(out),meta

def parse_wq(data):
 if data[:8]!=b'TS260Q8\0':raise ValueError('bad magic')
 off=8;cfg=struct.unpack_from('<7i',data,off);off+=28;count,=struct.unpack_from('<I',data,off);off+=4
 complete={}
 for _ in range(count):
  if off+2>len(data):break
  nl,=struct.unpack_from('<H',data,off);off+=2
  if off+nl+8>len(data):break
  name=data[off:off+nl].decode();off+=nl;rows,cols=struct.unpack_from('<II',data,off);off+=8
  if off+4*rows+rows*cols>len(data):break
  scales=torch.tensor(struct.unpack_from(f'<{rows}I',data,off),dtype=torch.float32)/65536.;off+=4*rows
  q=torch.frombuffer(bytearray(data[off:off+rows*cols]),dtype=torch.uint8).view(torch.int8).float().reshape(rows,cols);off+=rows*cols
  complete[name]=q*scales[:,None]
 if 'wq' not in complete:raise RuntimeError('verified prefix did not contain complete wq')
 return cfg,complete['wq'].reshape(cfg[2],cfg[0],cfg[0]),list(complete)

def main():
 data,chunks=fetch();cfg,wq,names=parse_wq(data);spectra=[];thresholds=[]
 for l,W in enumerate(wq):
  s=torch.linalg.svdvals(W).double();e=(s*s).cumsum(0)/(s*s).sum();spectra.append({'layer':l,**{f'energy_r{r}':float(e[r-1]) for r in RANKS}});thresholds.append({'layer':l,'rank_95':int((e>=.95).nonzero()[0])+1,'rank_98':int((e>=.98).nonzero()[0])+1,'rank_99':int((e>=.99).nonzero()[0])+1})
 pairs=[]
 for l in range(len(wq)-1):
  A,B=wq[l],wq[l+1];fa,fb=A.flatten(),B.flatten();M=(A+B)/2;_,_,Va=torch.linalg.svd(A,full_matrices=False);_,_,Vb=torch.linalg.svd(B,full_matrices=False);Qa,Qb=Va[:16].T,Vb[:16].T
  pairs.append({'layers':[l,l+1],'weight_cosine':float(torch.dot(fa,fb)/(fa.norm()*fb.norm())),'normalized_frob_distance':float((A-B).norm()/torch.sqrt(.5*(A.square().sum()+B.square().sum()))),'mean_isotropic_nmse_if_average_shared':float(.5*((M-A).square().sum()/A.square().sum()+(M-B).square().sum()/B.square().sum())),'top16_input_subspace_overlap':float((Qa.T@Qb).square().sum()/16)})
 out={'run':6,'evidence_level':'external_pretrained_quantized_weight_geometry_partial','model':'TinyStories-260K (llama2.c family), paLLM row-Q8 payload','source_repository':'maddiedreese/paLLM','source_commit':COMMIT,'checkpoint_format':'TS260Q8','checkpoint_config':{'dim':cfg[0],'hidden_dim':cfg[1],'n_layers':cfg[2],'n_heads':cfg[3],'n_kv_heads':cfg[4],'vocab_size':cfg[5],'max_seq_len':cfg[6]},'verified_source_prefix':{'bytes':len(data),'chunks':chunks,'complete_tensors_in_prefix':names},'wq_weight_spectral_energy':spectra,'wq_energy_threshold_ranks':thresholds,'adjacent_wq_diagnostics':pairs,'interpretation':{'rank16_fraction_of_width':.25,'rank16_weight_energy_range':[min(x['energy_r16'] for x in spectra),max(x['energy_r16'] for x in spectra)],'rank32_weight_energy_range':[min(x['energy_r32'] for x in spectra),max(x['energy_r32'] for x in spectra)],'rank98_required_range':[min(x['rank_98'] for x in thresholds),max(x['rank_98'] for x in thresholds)],'naive_parameter_sharing_supported':False,'activation_low_rank_thesis_decided':False},'limitations':['Weight-space geometry, not activation-space geometry.','Checkpoint is row-Q8 rather than original FP32.','Only the first 65,536 verified source bytes are required; Wq is complete for all five layers.','No LARC conversion, recovery, quality evaluation, or memory measurement.'],'claim_boundary':'First genuine external-pretrained parameter-geometry evidence for Run 6. It disfavors naive parameter averaging/sharing for Wq, but does not determine whether real activation distributions admit low-rank projection.'};Path('benchmarks/run6_pallm_wq_geometry.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
