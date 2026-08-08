from __future__ import annotations
from dataclasses import dataclass
import json,struct
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request,urlopen
import torch

_HEADER_LEN=struct.Struct('<Q')
_DTYPE={'F16':torch.float16,'BF16':torch.bfloat16,'F32':torch.float32}

class RangeUnsupportedError(RuntimeError):pass

@dataclass(frozen=True)
class TensorInfo:
    name:str;dtype:str;shape:tuple[int,...];start:int;end:int
    @property
    def nbytes(self):return self.end-self.start

class _Reader:
    def __init__(self,location:str|Path):
        self.location=str(location);self.remote=self.location.startswith(('http://','https://'));self.bytes_fetched=0
    def read_range(self,start:int,end_exclusive:int)->bytes:
        if end_exclusive<start:raise ValueError('invalid range')
        n=end_exclusive-start
        if not self.remote:
            with open(self.location,'rb') as f:f.seek(start);data=f.read(n)
            if len(data)!=n:raise EOFError(f'short read {self.location}: wanted {n}, got {len(data)}')
        else:
            req=Request(self.location,headers={'Range':f'bytes={start}-{end_exclusive-1}','Accept-Encoding':'identity'})
            with urlopen(req) as r:
                status=getattr(r,'status',None);cr=r.headers.get('Content-Range')
                # Never silently accept a server that ignores Range: doing so could
                # download the entire multi-GB source file for an 8-byte request.
                if status!=206 or not cr:raise RangeUnsupportedError(f'{self.location} did not honor byte Range (status={status}, Content-Range={cr!r})')
                data=r.read(n+1)
            if len(data)!=n:raise EOFError(f'range read returned {len(data)} bytes, expected {n}')
        self.bytes_fetched+=len(data);return data
    def read_json(self)->dict:
        if not self.remote:return json.loads(Path(self.location).read_text())
        with urlopen(Request(self.location,headers={'Accept-Encoding':'identity'})) as r:data=r.read()
        self.bytes_fetched+=len(data);return json.loads(data)

class SafeTensorFile:
    """Tensor-addressable SafeTensors reader using exact byte ranges.

    Remote sources must return HTTP 206. If the endpoint ignores Range, reading is
    aborted before the response body is consumed, preventing accidental full-file
    downloads.
    """
    def __init__(self,location:str|Path):
        self.location=str(location);self.reader=_Reader(location);h8=self.reader.read_range(0,8);hlen=_HEADER_LEN.unpack(h8)[0]
        if hlen<=0 or hlen>256*1024*1024:raise ValueError(f'implausible SafeTensors header length {hlen}')
        header=json.loads(self.reader.read_range(8,8+hlen));self.data_offset=8+hlen;self.tensors={}
        for name,v in header.items():
            if name=='__metadata__':continue
            a,b=v['data_offsets'];self.tensors[name]=TensorInfo(name,v['dtype'],tuple(v['shape']),self.data_offset+a,self.data_offset+b)
    @property
    def bytes_fetched(self):return self.reader.bytes_fetched
    def read_tensor(self,name:str,device='cpu')->torch.Tensor:
        info=self.tensors[name]
        if info.dtype not in _DTYPE:raise NotImplementedError(f'SafeTensors dtype {info.dtype}')
        raw=bytearray(self.reader.read_range(info.start,info.end));t=torch.frombuffer(raw,dtype=_DTYPE[info.dtype]).clone().reshape(info.shape)
        return t.to(device) if device!='cpu' else t

class ShardedSafeTensorSource:
    """Lazy SafeTensors shard set from model.safetensors.index.json."""
    def __init__(self,index_location:str|Path):
        self.index_location=str(index_location);self.index_reader=_Reader(index_location);idx=self.index_reader.read_json();self.weight_map=dict(idx['weight_map']);self._files={}
        if self.index_location.startswith(('http://','https://')):
            self.base=self.index_location.rsplit('/',1)[0]+'/'
        else:self.base=str(Path(self.index_location).parent)
    def _loc(self,shard:str):return urljoin(self.base,shard) if self.index_location.startswith(('http://','https://')) else str(Path(self.base)/shard)
    def _file(self,shard:str):
        if shard not in self._files:self._files[shard]=SafeTensorFile(self._loc(shard))
        return self._files[shard]
    def names(self):return self.weight_map.keys()
    def info(self,name:str)->TensorInfo:
        shard=self.weight_map[name];return self._file(shard).tensors[name]
    def read_tensor(self,name:str,device='cpu')->torch.Tensor:return self._file(self.weight_map[name]).read_tensor(name,device)
    @property
    def bytes_fetched(self):return self.index_reader.bytes_fetched+sum(f.bytes_fetched for f in self._files.values())
