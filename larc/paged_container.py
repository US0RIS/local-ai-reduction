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

def _manifest_bytes(manifest:dict,alignment:int)->bytes:
    man=dict(manifest);man.setdefault('larc',{});man['larc'].update({'major':MAJOR,'minor':MINOR,'alignment':alignment})
    return json.dumps(man,separators=(',',':'),sort_keys=True).encode()

def write_larc_v2(path:str|Path,pages:list[PageSpec],manifest:dict,alignment:int=4096)->None:
    """Compatibility writer for small artifacts. For converters use LARCv2StreamWriter."""
    if alignment<64 or alignment&(alignment-1):raise ValueError('alignment must be power-of-two >=64')
    if len({p.page_id for p in pages})!=len(pages):raise ValueError('duplicate page id')
    mb=_manifest_bytes(manifest,alignment);table_off=align_up(HEADER.size+len(mb),64);data_off=align_up(table_off+PAGE.size*len(pages),alignment);rec=[];off=data_off
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

class LARCv2StreamWriter:
    """Write a fixed-page-count LARC file without retaining payload pages in RAM.

    The converter declares page_count and manifest up front. The writer reserves the
    table, appends each aligned page immediately, retains only PageRecord objects,
    then patches the table/header on finalize. Peak writer memory is therefore O(page
    count metadata + largest payload passed to add_page), not O(file size).
    """
    def __init__(self,path:str|Path,page_count:int,manifest:dict,alignment:int=4096):
        if page_count<0:raise ValueError('page_count must be >= 0')
        if alignment<64 or alignment&(alignment-1):raise ValueError('alignment must be power-of-two >=64')
        self.path=Path(path);self.page_count=page_count;self.alignment=alignment;self.mb=_manifest_bytes(manifest,alignment)
        self.table_off=align_up(HEADER.size+len(self.mb),64);self.data_off=align_up(self.table_off+PAGE.size*page_count,alignment)
        self.records=[];self.ids=set();self._closed=False;self._fh=open(self.path,'w+b')
        # Header is provisional until the exact file length is known.
        self._fh.write(HEADER.pack(MAGIC,MAJOR,MINOR,0,len(self.mb),page_count,self.table_off,self.data_off,0));self._fh.write(self.mb)
        if self._fh.tell()<self.table_off:self._fh.write(b'\0'*(self.table_off-self._fh.tell()))
        self._fh.write(b'\0'*(PAGE.size*page_count))
        if self._fh.tell()<self.data_off:self._fh.write(b'\0'*(self.data_off-self._fh.tell()))
    def add_page(self,page_id:int,codec_id:int,flags:int,data:bytes|bytearray|memoryview,logical_length:int|None=None,dependency_group:int=0)->PageRecord:
        if self._closed:raise ValueError('writer is closed')
        if len(self.records)>=self.page_count:raise ValueError('more pages than declared')
        if page_id in self.ids:raise ValueError(f'duplicate page id {page_id}')
        off=align_up(self._fh.tell(),self.alignment)
        if self._fh.tell()<off:self._fh.write(b'\0'*(off-self._fh.tell()))
        v=memoryview(data);crc=zlib.crc32(v)&0xffffffff;self._fh.write(v)
        r=PageRecord(page_id,codec_id,flags,off,len(v),len(v) if logical_length is None else logical_length,crc,dependency_group);self.records.append(r);self.ids.add(page_id);return r
    def finalize(self)->None:
        if self._closed:return
        if len(self.records)!=self.page_count:raise ValueError(f'expected {self.page_count} pages, wrote {len(self.records)}')
        file_len=self._fh.tell();self._fh.seek(self.table_off)
        for r in self.records:self._fh.write(PAGE.pack(r.page_id,r.codec_id,r.flags,r.offset,r.stored_length,r.logical_length,r.crc32,r.dependency_group))
        self._fh.seek(0);self._fh.write(HEADER.pack(MAGIC,MAJOR,MINOR,0,len(self.mb),self.page_count,self.table_off,self.data_off,file_len));self._fh.flush();self._fh.close();self._closed=True
    def close(self):
        if not self._closed:self._fh.close();self._closed=True
    def __enter__(self):return self
    def __exit__(self,typ,val,tb):
        if typ is None:self.finalize()
        else:self.close()

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
