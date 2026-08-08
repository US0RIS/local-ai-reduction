from __future__ import annotations
from dataclasses import dataclass
import math,struct
from pathlib import Path
from urllib.request import Request,urlopen
import numpy as np
import torch

GGUF_MAGIC=b'GGUF'
GGML_F32=0;GGML_F16=1;GGML_Q4_0=2;GGML_Q4_1=3;GGML_Q5_0=6;GGML_Q5_1=7;GGML_Q8_0=8
GGML_Q2_K=10;GGML_Q3_K=11;GGML_Q4_K=12;GGML_Q5_K=13;GGML_Q6_K=14
TYPE_LAYOUT={
 GGML_F32:(1,4,'F32'),GGML_F16:(1,2,'F16'),GGML_Q4_0:(32,18,'Q4_0'),GGML_Q4_1:(32,20,'Q4_1'),
 GGML_Q5_0:(32,22,'Q5_0'),GGML_Q5_1:(32,24,'Q5_1'),GGML_Q8_0:(32,34,'Q8_0'),
 GGML_Q2_K:(256,84,'Q2_K'),GGML_Q3_K:(256,110,'Q3_K'),GGML_Q4_K:(256,144,'Q4_K'),GGML_Q5_K:(256,176,'Q5_K'),GGML_Q6_K:(256,210,'Q6_K'),
}
META_FIXED={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

class RangeUnsupportedError(RuntimeError):pass
class UnsupportedGGMLTypeError(NotImplementedError):pass

class _RangeReader:
 def __init__(self,location:str|Path):self.location=str(location);self.remote=self.location.startswith(('http://','https://'));self.bytes_fetched=0
 def read_range(self,start:int,end:int)->bytes:
  n=end-start
  if n<0:raise ValueError('invalid range')
  if not self.remote:
   with open(self.location,'rb') as f:f.seek(start);data=f.read(n)
   if len(data)!=n:raise EOFError(f'short read: wanted {n}, got {len(data)}')
  else:
   req=Request(self.location,headers={'Range':f'bytes={start}-{end-1}','Accept-Encoding':'identity'})
   with urlopen(req) as r:
    status=getattr(r,'status',None);cr=r.headers.get('Content-Range')
    if status!=206 or not cr:raise RangeUnsupportedError(f'{self.location} did not honor Range (status={status}, Content-Range={cr!r})')
    data=r.read(n+1)
   if len(data)!=n:raise EOFError(f'range read returned {len(data)} bytes, expected {n}')
  self.bytes_fetched+=len(data);return data

class _Cursor:
 def __init__(self,reader:_RangeReader,chunk_bytes:int=1<<20):self.r=reader;self.pos=0;self.chunk_bytes=chunk_bytes;self._start=-1;self._data=b''
 def tell(self):return self.pos
 def skip(self,n):self.pos+=n
 def _chunk(self):
  start=(self.pos//self.chunk_bytes)*self.chunk_bytes
  if not (self._start<=self.pos<self._start+len(self._data)):
   self._start=start;self._data=self.r.read_range(start,start+self.chunk_bytes)
 def read(self,n):
  out=bytearray()
  while n:
   self._chunk();i=self.pos-self._start;take=min(n,len(self._data)-i)
   if take<=0:raise EOFError('GGUF cursor ran past source')
   out+=self._data[i:i+take];self.pos+=take;n-=take
  return bytes(out)
 def u32(self):return struct.unpack('<I',self.read(4))[0]
 def u64(self):return struct.unpack('<Q',self.read(8))[0]
 def string(self):
  n=self.u64();return self.read(n).decode('utf-8')

def _skip_meta(c:_Cursor,t:int):
 if t in META_FIXED:c.skip(META_FIXED[t]);return
 if t==8:
  n=c.u64();c.skip(n);return
 if t==9:
  et=c.u32();n=c.u64()
  if et in META_FIXED:c.skip(META_FIXED[et]*n);return
  for _ in range(n):_skip_meta(c,et)
  return
 raise ValueError(f'unknown GGUF metadata type {t}')

def _read_meta(c:_Cursor,t:int):
 if t==4:return c.u32()
 if t==10:return c.u64()
 if t==8:return c.string()
 if t==7:return bool(c.read(1)[0])
 if t==6:return struct.unpack('<f',c.read(4))[0]
 # Only a small scalar subset is needed by the range source. Skip everything else.
 _skip_meta(c,t);return None

@dataclass(frozen=True)
class GGUFTensorInfo:
 name:str;ggml_type:int;ne:tuple[int,...];relative_offset:int;absolute_offset:int;nbytes:int
 @property
 def type_name(self):return TYPE_LAYOUT.get(self.ggml_type,(0,0,f'TYPE_{self.ggml_type}'))[2]
 @property
 def shape(self):return tuple(reversed(self.ne))

class GGUFRangeFile:
 """Parse GGUF metadata/tensor descriptors and range-load individual tensors.

 Supported execution dequantizers cover the types expected in classic Q4_K_M files:
 F32, F16, Q4_K, Q5_K and Q6_K. Other known types can still be indexed and
 reported but raise when tensor values are requested.
 """
 def __init__(self,location:str|Path):
  self.location=str(location);self.reader=_RangeReader(location);c=_Cursor(self.reader)
  if c.read(4)!=GGUF_MAGIC:raise ValueError('not a GGUF file')
  self.version=c.u32();self.tensor_count=c.u64();self.kv_count=c.u64();self.metadata={};alignment=32
  for _ in range(self.kv_count):
   key=c.string();t=c.u32();keep=key in ('general.alignment','general.architecture','general.name')
   if keep:
    value=_read_meta(c,t);self.metadata[key]=value
    if key=='general.alignment' and value is not None:alignment=int(value)
   else:_skip_meta(c,t)
  infos=[]
  for _ in range(self.tensor_count):
   name=c.string();nd=c.u32();ne=tuple(c.u64() for _ in range(nd));typ=c.u32();off=c.u64()
   if typ not in TYPE_LAYOUT:raise UnsupportedGGMLTypeError(f'unknown GGML type id {typ} for {name}')
   bs,ts,_=TYPE_LAYOUT[typ];count=math.prod(ne)
   if count%bs:raise ValueError(f'{name}: element count {count} not divisible by block size {bs}')
   infos.append((name,typ,ne,off,count//bs*ts))
  self.alignment=alignment;self.header_end=c.tell();self.data_offset=((self.header_end+alignment-1)//alignment)*alignment
  self.tensors={name:GGUFTensorInfo(name,typ,ne,off,self.data_offset+off,nbytes) for name,typ,ne,off,nbytes in infos}
 def raw_header_bytes(self):return self.reader.read_range(0,self.data_offset)
 @property
 def bytes_fetched(self):return self.reader.bytes_fetched
 def type_histogram(self):
  out={}
  for t in self.tensors.values():out[t.type_name]=out.get(t.type_name,0)+1
  return out
 def read_raw_tensor(self,name):
  i=self.tensors[name];return self.reader.read_range(i.absolute_offset,i.absolute_offset+i.nbytes)
 def read_tensor(self,name,device='cpu'):
  i=self.tensors[name];raw=self.read_raw_tensor(name);count=math.prod(i.ne)
  if i.ggml_type==GGML_F32:arr=np.frombuffer(raw,dtype='<f4',count=count).copy()
  elif i.ggml_type==GGML_F16:arr=np.frombuffer(raw,dtype='<f2',count=count).astype(np.float32)
  elif i.ggml_type==GGML_Q4_K:arr=_dequant_q4_k(raw)
  elif i.ggml_type==GGML_Q5_K:arr=_dequant_q5_k(raw)
  elif i.ggml_type==GGML_Q6_K:arr=_dequant_q6_k(raw)
  else:raise UnsupportedGGMLTypeError(f'dequantization not implemented for {i.type_name} tensor {name}')
  t=torch.from_numpy(arr.reshape(i.shape))
  return t.to(device) if device!='cpu' else t

def _half_col(raw2):return np.frombuffer(raw2.copy().reshape(-1).tobytes(),dtype='<f2').astype(np.float32)
def _scale_min_k4(s):
 b=s.shape[0];sc=np.empty((b,8),np.uint8);mn=np.empty((b,8),np.uint8)
 for j in range(8):
  if j<4:
   sc[:,j]=s[:,j]&63;mn[:,j]=s[:,j+4]&63
  else:
   sc[:,j]=(s[:,j+4]&15)|((s[:,j-4]>>6)<<4);mn[:,j]=(s[:,j+4]>>4)|((s[:,j]>>6)<<4)
 return sc.astype(np.float32),mn.astype(np.float32)

def _dequant_q4_k(raw:bytes):
 u=np.frombuffer(raw,dtype=np.uint8)
 if u.size%144:raise ValueError('Q4_K payload not block aligned')
 b=u.reshape(-1,144);d=_half_col(b[:,0:2]);dm=_half_col(b[:,2:4]);sc,mn=_scale_min_k4(b[:,4:16]);q=b[:,16:144];out=np.empty((b.shape[0],256),np.float32)
 for g in range(4):
  z=q[:,32*g:32*(g+1)];out[:,64*g:64*g+32]=d[:,None]*sc[:,2*g,None]*(z&15)-dm[:,None]*mn[:,2*g,None];out[:,64*g+32:64*g+64]=d[:,None]*sc[:,2*g+1,None]*(z>>4)-dm[:,None]*mn[:,2*g+1,None]
 return out.reshape(-1)

def _dequant_q5_k(raw:bytes):
 u=np.frombuffer(raw,dtype=np.uint8)
 if u.size%176:raise ValueError('Q5_K payload not block aligned')
 b=u.reshape(-1,176);d=_half_col(b[:,0:2]);dm=_half_col(b[:,2:4]);sc,mn=_scale_min_k4(b[:,4:16]);qh=b[:,16:48];q=b[:,48:176];out=np.empty((b.shape[0],256),np.float32)
 for g in range(4):
  z=q[:,32*g:32*(g+1)];m1=1<<(2*g);m2=2<<(2*g);hi1=((qh&m1)!=0).astype(np.uint8)*16;hi2=((qh&m2)!=0).astype(np.uint8)*16
  out[:,64*g:64*g+32]=d[:,None]*sc[:,2*g,None]*((z&15)+hi1)-dm[:,None]*mn[:,2*g,None];out[:,64*g+32:64*g+64]=d[:,None]*sc[:,2*g+1,None]*((z>>4)+hi2)-dm[:,None]*mn[:,2*g+1,None]
 return out.reshape(-1)

def _dequant_q6_k(raw:bytes):
 u=np.frombuffer(raw,dtype=np.uint8)
 if u.size%210:raise ValueError('Q6_K payload not block aligned')
 b=u.reshape(-1,210);ql=b[:,0:128];qh=b[:,128:192];sc=b[:,192:208].view(np.int8).astype(np.float32);d=_half_col(b[:,208:210]);out=np.empty((b.shape[0],256),np.float32);idx=np.arange(32)//16
 for h in range(2):
  zl=ql[:,64*h:64*h+64];zh=qh[:,32*h:32*h+32];ss=sc[:,8*h:8*h+8];base=128*h
  q1=((zl[:,0:32]&15)|(((zh>>0)&3)<<4)).astype(np.int16)-32;q2=((zl[:,32:64]&15)|(((zh>>2)&3)<<4)).astype(np.int16)-32;q3=((zl[:,0:32]>>4)|(((zh>>4)&3)<<4)).astype(np.int16)-32;q4=((zl[:,32:64]>>4)|(((zh>>6)&3)<<4)).astype(np.int16)-32
  out[:,base:base+32]=d[:,None]*ss[:,idx+0]*q1;out[:,base+32:base+64]=d[:,None]*ss[:,idx+2]*q2;out[:,base+64:base+96]=d[:,None]*ss[:,idx+4]*q3;out[:,base+96:base+128]=d[:,None]*ss[:,idx+6]*q4
 return out.reshape(-1)

GGUF_TO_HF_FIXED={'model.embed_tokens.weight':'token_embd.weight','lm_head.weight':'output.weight','model.norm.weight':'output_norm.weight'}
GGUF_LAYER_SUFFIX={'self_attn.q_proj.weight':'attn_q.weight','self_attn.k_proj.weight':'attn_k.weight','self_attn.v_proj.weight':'attn_v.weight','self_attn.o_proj.weight':'attn_output.weight','mlp.gate_proj.weight':'ffn_gate.weight','mlp.up_proj.weight':'ffn_up.weight','mlp.down_proj.weight':'ffn_down.weight','input_layernorm.weight':'attn_norm.weight','post_attention_layernorm.weight':'ffn_norm.weight'}
HF_LAYER_RE=__import__('re').compile(r'^model\.layers\.(\d+)\.(.+)$')

class MistralGGUFSource:
 """Expose Mistral GGUF tensors through the same logical names as the SafeTensors converter."""
 def __init__(self,location:str|Path):
  self.gguf=GGUFRangeFile(location);self.location=str(location);self._map={}
  for hf,gg in GGUF_TO_HF_FIXED.items():
   if gg in self.gguf.tensors:self._map[hf]=gg
  for gg in self.gguf.tensors:
   m=__import__('re').match(r'^blk\.(\d+)\.(.+)$',gg)
   if not m:continue
   layer=int(m.group(1));tail=m.group(2)
   for hf_tail,gg_tail in GGUF_LAYER_SUFFIX.items():
    if tail==gg_tail:self._map[f'model.layers.{layer}.{hf_tail}']=gg;break
 def names(self):return self._map.keys()
 def _gg(self,name):
  if name not in self._map:raise KeyError(name)
  return self._map[name]
 def info(self,name):return self.gguf.tensors[self._gg(name)]
 def read_tensor(self,name,device='cpu'):return self.gguf.read_tensor(self._gg(name),device)
 @property
 def bytes_fetched(self):return self.gguf.bytes_fetched
 def raw_source_metadata(self):return self.gguf.raw_header_bytes()
 def type_histogram(self):return self.gguf.type_histogram()
