import pytest
import torch
from larc.latent_kv import pack_q2_rows_fp8meta,unpack_q2_rows_fp8meta,FP8_E4M3

@pytest.mark.skipif(FP8_E4M3 is None,reason='torch float8_e4m3fn unavailable')
def test_fp8_metadata_storage_is_one_byte_each():
    torch.manual_seed(17);x=torch.randn(13,16)*0.25
    p,mn,sc,n=pack_q2_rows_fp8meta(x)
    assert p.dtype==torch.uint8 and mn.dtype==torch.uint8 and sc.dtype==torch.uint8
    assert p.numel()==13*4
    assert mn.numel()==13 and sc.numel()==13
    y=unpack_q2_rows_fp8meta(p,mn,sc,n)
    assert y.shape==x.shape
    assert torch.isfinite(y).all()

@pytest.mark.skipif(FP8_E4M3 is None,reason='torch float8_e4m3fn unavailable')
def test_fp8_metadata_has_expected_e4m3_bits():
    x=torch.tensor([[-1.5,-0.5,0.5,1.5]],dtype=torch.float32)
    _,mn,sc,_=pack_q2_rows_fp8meta(x)
    # E4M3-FN(-1.5)=0xbc; scale=(3/3)=1 -> 0x38.
    assert int(mn[0])==0xBC
    assert int(sc[0])==0x38
