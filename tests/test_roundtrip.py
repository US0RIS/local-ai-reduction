import tempfile
from pathlib import Path
import numpy as np
from larc.container import Chunk,read_manifest,write_larc
from larc.hrvq import HRVQConfig,decode,encode,train_codebooks
from larc.q4 import quantize_q4,dequantize_q4
from larc.projection import fit_projection_bundle,run_bundle

def test_hrvq_shapes_and_finite():
    rng=np.random.default_rng(0); w=rng.standard_normal((130,70),dtype=np.float32); cfg=HRVQConfig(stages=2,codebook_size=64,max_train_vectors=4096); model=train_codebooks([w],cfg); enc=encode(w,model); dec=decode(enc,model); assert dec.shape==w.shape; assert np.isfinite(dec).all(); assert enc.indices.dtype==np.uint8; assert enc.scales.dtype==np.float16

def test_container_manifest():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'x.larc'; write_larc(p,[Chunk('a','raw',b'abc',{'x':1})],{'model':'test'}); m=read_manifest(p); assert m['metadata']['model']=='test'; assert m['chunks'][0]['length']==3

def test_q4_and_projection_bundle():
    rng=np.random.default_rng(2); w=rng.standard_normal((32,32),dtype=np.float32); q=quantize_q4(w); wh=dequantize_q4(q); assert wh.shape==w.shape; x=rng.standard_normal((32,128),dtype=np.float32); b=fit_projection_bundle([w,w*.5],x,8); ys=run_bundle(b,x[:,:4]); assert len(ys)==2 and ys[0].shape==(32,4)
