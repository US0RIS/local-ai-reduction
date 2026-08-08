import torch
from larc.latent_kv import quantize_head_basis_q4,kivi_latent_cache_bytes,fp16_cache_bytes


def test_quantized_basis_metric_matches_row_space_projection_for_values():
    torch.manual_seed(7)
    heads,rank,dim=2,8,32
    raw=torch.randn(heads,rank,dim)
    q,_=torch.linalg.qr(raw.transpose(-1,-2),mode='reduced')
    basis=q.transpose(-1,-2).contiguous()
    qb=quantize_head_basis_q4(basis,store_metric=True)
    x=torch.randn(heads,11,dim)
    latent=torch.einsum('htd,hrd->htr',x,qb.dequantized)
    corrected=torch.einsum('htr,hrs,hsd->htd',latent,qb.metric_inv,qb.dequantized)
    explicit=torch.einsum('htd,hrd,hrs,hsf->htf',x,qb.dequantized,qb.metric_inv,qb.dequantized)
    naive=torch.einsum('htr,hrd->htd',latent,qb.dequantized)
    assert torch.allclose(corrected,explicit,atol=3e-3,rtol=3e-3)
    assert (corrected-explicit).pow(2).mean() <= (naive-explicit).pow(2).mean()+1e-9


def test_both_metrics_are_charged_in_storage():
    torch.manual_seed(9)
    basis=torch.randn(4,16,32)
    k=quantize_head_basis_q4(basis,store_metric=True)
    v=quantize_head_basis_q4(basis,store_metric=True)
    expected_one=4*16*(16+2)+4*16*16*2
    assert k.storage_bytes==expected_one
    assert k.storage_bytes+v.storage_bytes==6400


def test_smollm_shaped_kivi_accounting_includes_q4_scales_and_both_metrics():
    layers,seq,heads,hd,rank=30,2048,3,64,16
    got=kivi_latent_cache_bytes(layers=layers,seq=seq,kv_heads=heads,head_dim=hd,rank=rank)
    vectors=layers*seq*heads
    payload=2*(vectors*rank*2//8)
    key_meta=layers*heads*(seq//64)*rank*4
    value_meta=vectors*4
    rows=layers*heads*2*rank
    q4_bases=rows*(hd//2)+rows*2
    both_metrics=2*layers*heads*rank*rank*2
    assert got==payload+key_meta+value_meta+q4_bases+both_metrics
    assert fp16_cache_bytes(layers=layers,seq=seq,kv_heads=heads,head_dim=hd)/got > 18.2
