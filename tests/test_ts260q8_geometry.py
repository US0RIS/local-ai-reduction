import random, struct
from pathlib import Path
import torch
from tools.run6_ts260q8_geometry import TS260Q8Model, MAGIC


def write_fixture(path: Path):
    rng=random.Random(1);D,F,L,H,K,V,T=64,172,5,8,4,512,512;kd=D*K//H;hs=D//H
    plan=[('tok_embeddings',V,D),('rms_att_weight',L,D),('wq',L*D,D),('wk',L*kd,D),('wv',L*kd,D),('wo',L*D,D),('rms_ffn_weight',L,D),('w1',L*F,D),('w2',L*D,F),('w3',L*F,D),('rms_final_weight',1,D),('freq_cis_real',T,hs//2),('freq_cis_imag',T,hs//2)]
    out=bytearray(MAGIC+struct.pack('<7i',D,F,L,H,K,V,T)+struct.pack('<I',len(plan)))
    for name,rows,cols in plan:
        b=name.encode();out+=struct.pack('<H',len(b))+b+struct.pack('<II',rows,cols)
        if name.startswith('rms_'):scale=512
        elif name=='freq_cis_real':scale=65536
        elif name=='freq_cis_imag':scale=1
        else:scale=64
        out+=struct.pack(f'<{rows}I',*([scale]*rows));vals=bytearray()
        for _ in range(rows*cols):
            if name=='freq_cis_real':q=127
            elif name=='freq_cis_imag':q=0
            elif name.startswith('rms_'):q=127
            else:q=rng.randint(-12,12)
            vals.append(q&255)
        out+=vals
    path.write_bytes(out)
    assert len(out)==282584


def test_ts260q8_parser_and_forward(tmp_path):
    p=tmp_path/'fixture.q8';write_fixture(p);m=TS260Q8Model(p)
    assert m.cfg.dim==64 and m.cfg.hidden_dim==172 and m.cfg.n_layers==5
    assert m.cfg.n_heads==8 and m.cfg.n_kv_heads==4 and m.cfg.vocab_size==512
    assert m.tensor_shapes['w1']==[860,64]
    logits=m.forward_sequence([1,2,3,4])
    assert logits.shape==(3,512)
    assert torch.isfinite(logits).all()
    reused=m.forward_sequence([1,2,3,4],override={1:0})
    assert reused.shape==logits.shape and torch.isfinite(reused).all()
