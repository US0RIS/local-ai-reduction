"""Bounded HTTP-range GGUF metadata/tensor-table reader for public model evaluation."""
from __future__ import annotations
from dataclasses import dataclass
import struct
from typing import Any
import requests
GGUF_MAGIC=b"GGUF";DEFAULT_ALIGNMENT=32
class GGUFError(RuntimeError): pass
@dataclass(frozen=True)
class TensorInfo:
    name:str;shape:tuple[int,...];ggml_type:int;relative_offset:int;absolute_offset:int;byte_length:int|None
class HTTPRangeSource:
    def __init__(self,url:str,*,timeout:float=30.0,max_response_bytes:int=8<<20):
        self.url=url;self.timeout=timeout;self.max_response_bytes=max_response_bytes;self.session=requests.Session();self.size=self._discover_size()
    def _discover_size(self):
        try:
            r=self.session.head(self.url,allow_redirects=True,timeout=self.timeout)
            if r.ok and r.headers.get('content-length'):return int(r.headers['content-length'])
        except requests.RequestException:pass
        r=self.session.get(self.url,headers={'Range':'bytes=0-0'},allow_redirects=True,timeout=self.timeout,stream=True)
        try:
            if r.status_code==206:
                cr=r.headers.get('content-range','')
                if '/' in cr and cr.rsplit('/',1)[1].isdigit():return int(cr.rsplit('/',1)[1])
            return int(r.headers['content-length']) if r.headers.get('content-length') else None
        finally:r.close()
    def read(self,start:int,length:int)->bytes:
        if length<=0:return b''
        if length>self.max_response_bytes:raise GGUFError(f'requested range {length} exceeds cap {self.max_response_bytes}')
        r=self.session.get(self.url,headers={'Range':f'bytes={start}-{start+length-1}'},allow_redirects=True,timeout=self.timeout,stream=True)
        try:
            if r.status_code!=206:raise GGUFError(f'HTTP {r.status_code}: range not honored')
            data=r.raw.read(length+1)
            if len(data)!=length:raise GGUFError(f'short/long range: {len(data)} != {length}')
            return data
        finally:r.close()
class ProgressiveReader:
    def __init__(self,source,*,chunk_bytes=256<<10,max_header_bytes=64<<20):self.source=source;self.chunk_bytes=chunk_bytes;self.max_header_bytes=max_header_bytes;self.buf=bytearray();self.pos=0
    def _ensure(self,end):
        while len(self.buf)<end:
            if len(self.buf)>=self.max_header_bytes:raise GGUFError('header limit')
            n=min(self.chunk_bytes,self.max_header_bytes-len(self.buf))
            if self.source.size is not None:n=min(n,self.source.size-len(self.buf))
            if n<=0:raise GGUFError('EOF')
            self.buf.extend(self.source.read(len(self.buf),n))
    def take(self,n):
        end=self.pos+n;self._ensure(end);out=bytes(self.buf[self.pos:end]);self.pos=end;return out
    def unpack(self,fmt):return struct.unpack(fmt,self.take(struct.calcsize(fmt)))
    def u8(self):return self.unpack('<B')[0]
    def i8(self):return self.unpack('<b')[0]
    def u16(self):return self.unpack('<H')[0]
    def i16(self):return self.unpack('<h')[0]
    def u32(self):return self.unpack('<I')[0]
    def i32(self):return self.unpack('<i')[0]
    def u64(self):return self.unpack('<Q')[0]
    def i64(self):return self.unpack('<q')[0]
    def f32(self):return self.unpack('<f')[0]
    def f64(self):return self.unpack('<d')[0]
    def string(self):
        n=self.u64()
        if n>self.max_header_bytes:raise GGUFError(f'bad string length {n}')
        return self.take(n).decode('utf-8')
UINT8,INT8,UINT16,INT16,UINT32,INT32,FLOAT32,BOOL,STRING,ARRAY,UINT64,INT64,FLOAT64=range(13)
def _metadata_value(r,value_type,depth=0):
    if depth>8:raise GGUFError('metadata nesting')
    if value_type==UINT8:return r.u8()
    if value_type==INT8:return r.i8()
    if value_type==UINT16:return r.u16()
    if value_type==INT16:return r.i16()
    if value_type==UINT32:return r.u32()
    if value_type==INT32:return r.i32()
    if value_type==FLOAT32:return r.f32()
    if value_type==BOOL:return bool(r.u8())
    if value_type==STRING:return r.string()
    if value_type==UINT64:return r.u64()
    if value_type==INT64:return r.i64()
    if value_type==FLOAT64:return r.f64()
    if value_type==ARRAY:
        et=r.u32();n=r.u64()
        if n>100_000_000:raise GGUFError('bad array length')
        return [_metadata_value(r,et,depth+1) for _ in range(n)]
    raise GGUFError(f'unsupported metadata type {value_type}')
def _align(n,a):return ((n+a-1)//a)*a
@dataclass(frozen=True)
class GGUFIndex:
    version:int;metadata:dict[str,Any];tensors:tuple[TensorInfo,...];tensor_data_offset:int;remote_size:int|None;header_bytes_fetched:int
def inspect_remote_gguf(url,*,chunk_bytes=256<<10,max_header_bytes=64<<20,max_response_bytes=8<<20):
    source=HTTPRangeSource(url,max_response_bytes=max_response_bytes);r=ProgressiveReader(source,chunk_bytes=chunk_bytes,max_header_bytes=max_header_bytes)
    if r.take(4)!=GGUF_MAGIC:raise GGUFError('magic')
    ver=r.u32()
    if ver not in (2,3):raise GGUFError(f'version {ver}')
    tc=r.u64();mc=r.u64();metadata={}
    for _ in range(mc):metadata[r.string()]=_metadata_value(r,r.u32())
    raw=[]
    for _ in range(tc):
        name=r.string();nd=r.u32();shape=tuple(r.u64() for _ in range(nd));typ=r.u32();rel=r.u64();raw.append((name,shape,typ,rel))
    alignment=int(metadata.get('general.alignment',DEFAULT_ALIGNMENT));base=_align(r.pos,alignment);starts=sorted({x[3] for x in raw});nxt={starts[i]:starts[i+1] for i in range(len(starts)-1)};tensors=[]
    for name,shape,typ,rel in raw:
        abs_start=base+rel;length=nxt[rel]-rel if rel in nxt else (source.size-abs_start if source.size is not None else None);tensors.append(TensorInfo(name,shape,typ,rel,abs_start,length))
    return GGUFIndex(ver,metadata,tuple(tensors),base,source.size,len(r.buf))
