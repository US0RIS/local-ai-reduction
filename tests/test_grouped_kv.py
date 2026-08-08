import torch
from larc.grouped_kv import pack_q2_token_group_scalar,unpack_q2_token_group_scalar,grouped_latent_cache_bytes,reference_workspace_bytes


def test_grouped_q2_roundtrip_shape_and_storage():
    x=torch.tensor([[0.0,1.0,2.0,3.0],[1.0,2.0,3.0,4.0],[2.0,3.0,4.0,5.0]],dtype=torch.float32)
    q=pack_q2_token_group_scalar(x)
    y=unpack_q2_token_group_scalar(q)
    assert y.shape==x.shape
    assert q.storage_bytes==q.packed.numel()+4
    # Scalar-group Q2 is lossy, but endpoints must survive to FP16 metadata precision.
    assert abs(float(y.min())-0.0)<1e-3
    assert abs(float(y.max())-5.0)<5e-3


def test_recurrent_basis_sets_are_not_multiplied_by_logical_depth():
    shared=grouped_latent_cache_bytes(layers=16,seq=64,kv_heads=4,head_dim=32,rank=16,group_tokens=3,basis_sets=1)
    independent=grouped_latent_cache_bytes(layers=16,seq=64,kv_heads=4,head_dim=32,rank=16,group_tokens=3,basis_sets=16)
    assert shared==58624
    assert independent>shared


def test_run5_workspace_scales_with_context():
    assert reference_workspace_bytes(context=64,hidden=128,heads=4,rank=16,intermediate=256)==8704
    assert reference_workspace_bytes(context=8192,hidden=128,heads=4,rank=16,intermediate=256)==658944
