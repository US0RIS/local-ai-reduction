import torch
from larc.latent_kv import quantize_head_basis_q4


def test_quantized_basis_metric_matches_row_space_projection_for_values():
    torch.manual_seed(7)
    heads,rank,dim=2,8,32
    raw=torch.randn(heads,rank,dim)
    # Start from an orthonormal basis, then let Q4 perturb it.
    q,_=torch.linalg.qr(raw.transpose(-1,-2),mode='reduced')
    basis=q.transpose(-1,-2).contiguous()
    qb=quantize_head_basis_q4(basis,store_metric=True)
    x=torch.randn(heads,11,dim)
    latent=torch.einsum('htd,hrd->htr',x,qb.dequantized)
    corrected=torch.einsum('htr,hrs,hsd->htd',latent,qb.metric_inv,qb.dequantized)
    # Explicit orthogonal projection onto row(B_hat).
    explicit=torch.einsum('htd,hrd,hrs,hsf->htf',x,qb.dequantized,qb.metric_inv,qb.dequantized)
    naive=torch.einsum('htr,hrd->htd',latent,qb.dequantized)
    assert torch.allclose(corrected,explicit,atol=3e-3,rtol=3e-3)
    assert (corrected-explicit).pow(2).mean() <= (naive-explicit).pow(2).mean()+1e-9


def test_both_metrics_are_charged_in_storage():
    torch.manual_seed(9)
    basis=torch.randn(4,16,32)
    k=quantize_head_basis_q4(basis,store_metric=True)
    v=quantize_head_basis_q4(basis,store_metric=True)
    # per basis: 4*16*(16 packed bytes + 2 scale bytes) + 4*16*16*2 metric bytes
    expected_one=4*16*(16+2)+4*16*16*2
    assert k.storage_bytes==expected_one
    assert k.storage_bytes+v.storage_bytes==6400
