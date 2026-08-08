"""Hierarchical residual vector quantization prototype for LARC."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from sklearn.cluster import MiniBatchKMeans

@dataclass
class HRVQConfig:
    vector_dim:int=64; vectors_per_scale:int=4; codebook_size:int=256; stages:int=2; max_train_vectors:int=65536; random_state:int=0
    @property
    def weights_per_scale(self): return self.vector_dim*self.vectors_per_scale
    @property
    def nominal_bits_per_weight(self): return 16.0/self.weights_per_scale + self.stages*8.0/self.vector_dim
@dataclass
class HRVQModel:
    config:HRVQConfig; codebooks:list[np.ndarray]
    @property
    def codebook_bytes(self): return int(sum(c.nbytes for c in self.codebooks))
@dataclass
class HRVQEncoded:
    original_shape:tuple[int,...]; original_numel:int; padded_numel:int; scales:np.ndarray; indices:np.ndarray
    @property
    def payload_bytes(self): return int(self.scales.nbytes+self.indices.nbytes)

def _prepare(weight,cfg):
    flat=np.asarray(weight,dtype=np.float32).reshape(-1); group=cfg.weights_per_scale; padded_numel=((flat.size+group-1)//group)*group
    padded=np.zeros(padded_numel,dtype=np.float32); padded[:flat.size]=flat; blocks=padded.reshape(-1,group)
    scales=np.sqrt(np.mean(blocks*blocks,axis=1)+1e-12).astype(np.float32); vectors=(blocks/scales[:,None]).reshape(-1,cfg.vector_dim)
    return vectors,scales,padded_numel

def train_codebooks(weights:Iterable[np.ndarray],cfg:HRVQConfig)->HRVQModel:
    rng=np.random.default_rng(cfg.random_state); allv=[]
    for w in weights:
        v,_,_=_prepare(w,cfg)
        if len(v)>cfg.max_train_vectors: v=v[rng.choice(len(v),cfg.max_train_vectors,replace=False)]
        allv.append(v)
    train=np.concatenate(allv)
    if len(train)>cfg.max_train_vectors: train=train[rng.choice(len(train),cfg.max_train_vectors,replace=False)]
    residual=train.astype(np.float32,copy=True); cbs=[]
    for stage in range(cfg.stages):
        km=MiniBatchKMeans(n_clusters=cfg.codebook_size,batch_size=min(4096,max(cfg.codebook_size*4,1024)),n_init=1,max_iter=100,random_state=cfg.random_state+stage,reassignment_ratio=0.01)
        labels=km.fit_predict(residual); cb=km.cluster_centers_.astype(np.float32); residual-=cb[labels]; cbs.append(cb.astype(np.float16))
    return HRVQModel(cfg,cbs)

def _nearest(v,cb,batch=4096):
    cb=np.asarray(cb,dtype=np.float32); norm=np.sum(cb*cb,axis=1); out=np.empty(len(v),dtype=np.uint8)
    for s in range(0,len(v),batch):
        x=v[s:s+batch]; out[s:s+len(x)]=np.argmin(norm[None,:]-2.0*(x@cb.T),axis=1).astype(np.uint8)
    return out

def encode(weight,model):
    cfg=model.config; vectors,scales,padded=_prepare(weight,cfg); residual=vectors.astype(np.float32,copy=True); idx=np.empty((len(vectors),cfg.stages),dtype=np.uint8)
    for stage,cb16 in enumerate(model.codebooks):
        cb=cb16.astype(np.float32); ii=_nearest(residual,cb); idx[:,stage]=ii; residual-=cb[ii]
    return HRVQEncoded(tuple(weight.shape),int(weight.size),padded,scales.astype(np.float16),idx)

def decode(encoded,model):
    cfg=model.config; vectors=np.zeros((len(encoded.indices),cfg.vector_dim),dtype=np.float32)
    for stage,cb16 in enumerate(model.codebooks): vectors+=cb16.astype(np.float32)[encoded.indices[:,stage]]
    flat=(vectors.reshape(-1,cfg.weights_per_scale)*encoded.scales.astype(np.float32)[:,None]).reshape(-1)[:encoded.original_numel]
    return flat.reshape(encoded.original_shape)

def storage_bytes(encoded,model,include_codebook=True): return encoded.payload_bytes+(model.codebook_bytes if include_codebook else 0)
