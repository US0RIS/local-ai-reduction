from __future__ import annotations
from dataclasses import dataclass
import json,mmap,struct,zlib
from pathlib import Path

MAGIC=b'LARCv2\x00\x00'
HEADER=struct.Struct('<8sHHIQQQQQ8x') # fixed 64-byte header
PAGE=struct.Struct('<IHHQQQII24x')    # fixed 64-byte page record
MAJOR=2;MINOR=0
FLAG_REQUIRED=1<<0;FLAG_SHARED=1<<1;FLAG_REFINEMENT=1<<2;FLAG_STREAMABLE=1<<3;FLAG_KV_BASIS=1<<4
CODEC_RAW=0;CODEC_Q4_ROW=1;CODEC_Q8_ROW=2;CODEC_PROJECTION_Q4=3;CODEC_HRVQ64=4;CODEC_LATENT_KV_BASIS_Q4=5;CODEC_SPARSE_RESCUE=6

@dataclass
class PageSpec:
    page_id:int;codec_id:int;flags:int;data:bytes;logical_length:int|None=None;dependency_group:int=0
@dataclass
class PageRecord:
    page_id:int;codec_id:int;flags:int;offset:int;stored_length:int;logical_length:int;crc32:int;dependency_group:int

def align_up(n:int,a:int)->int:return ((n+a-1)//a)*a

def write_larc_v2(path:str|Path,pages:list[PageSpec],manifest:dict,alignment:int=4096)->None:
    if alignment<64 or alignment&(alignment-1):raise ValueError('alignment must be power-of-two >=64')
    if len({p.page_id for p in pages})!=len(pages):raise ValueError('duplicate page id')
    man=dict(manifest);man.setdefault('larc',{});man['larc'].update({'major':MAJOR,'minor':MINOR,'alignment':alignment});mb=json.dumps(man,separators=(',',':'),sort_keys=True).encode()
    table_off=align_up(HEADER.size+len(mb),64);data_off=align_up(table_off+PAGE.size*len(pages),alignment);rec=[];off=data_off
    for p in pages:
        off=align_up(off,alignment);d=bytes(p.data);rec.append(PageRecord(p.page_id,p.codec_id,p.flags,off,len(d),len(d) if p.logical_length is None else p.logical_length,zlib.crc32(d)&0xffffffff,p.dependency_group));off+=len(d)
    file_len=off
    with open(path,'wb') as f:
        f.write(HEADER.pack(MAGIC,MAJOR,MINOR,0,len(mb),len(pages),table_off,data_off,file_len));f.write(mb);f.write(b'\0'*(table_off-f.tell()))
        for r in rec:f.write(PAGE.pack(r.page_id,r.codec_id,r.flags,r.offset,r.stored_length,r.logical_length,r.crc32,r.dependency_group))
        f.write(b'\0'*(data_off-f.tell()))
        for p,r in zip(pages,rec):
            if f.tell()<r.offset:f.write(b'\0'*(r.offset-f.tell()))
            f.write(p.data)

class LARCv2File:
    def __init__(self,path:str|Path):
        self.path=Path(path);self._fh=open(self.path,'rb');self._mm=mmap.mmap(self._fh.fileno(),0,access=mmap.ACCESS_READ)
        magic,maj,minr,flags,mlen,npages,toff,doff,flen=HEADER.unpack_from(self._mm,0)
        if magic!=MAGIC or maj!=MAJOR:raise ValueError('unsupported LARC file')
        if flen!=len(self._mm):raise ValueError('truncated/extended LARC file')
        self.major,self.minor,self.flags,self.data_offset=maj,minr,flags,doff;self.manifest=json.loads(self._mm[HEADER.size:HEADER.size+mlen]);self.pages={}
        for i in range(npages):
            r=PageRecord(*PAGE.unpack_from(self._mm,toff+i*PAGE.size))
            if r.offset+r.stored_length>len(self._mm):raise ValueError('page outside file')
            self.pages[r.page_id]=r
    def page_view(self,page_id:int,verify:bool=False)->memoryview:
        r=self.pages[page_id];v=memoryview(self._mm)[r.offset:r.offset+r.stored_length]
        if verify and (zlib.crc32(v)&0xffffffff)!=r.crc32:raise ValueError(f'CRC failure page {page_id}')
        return v
    def resident_payload_bytes(self,page_ids)->int:return sum(self.pages[i].stored_length for i in set(page_ids))
    def close(self):self._mm.close();self._fh.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
