import torch
from larc.grouped_kv import pack_q2_token_group_scalar,unpack_q2_token_group_scalar,grouped_latent_cache_bytes,reference_workspace_bytes

def test_grouped_q2_roundtrip_shape_and_storage():
    x=torch.tensor([[0.,1.,2.,3.],[1.,2.,3.,4.],[2.,3.,4.,5.]])
    q=pack_q2_token_group_scalar(x);y=unpack_q2_token_group_scalar(q)
    assert y.shape==x.shape
    assert q.storage_bytes==q.packed.numel()+4
    assert abs(float(y.min()))<1e-3 and abs(float(y.max())-5.)<5e-3

def test_shared_basis_set_accounting():
    shared=grouped_latent_cache_bytes(layers=16,seq=64,kv_heads=4,head_dim=32,rank=16,group_tokens=3,basis_sets=1)
    independent=grouped_latent_cache_bytes(layers=16,seq=64,kv_heads=4,head_dim=32,rank=16,group_tokens=3,basis_sets=16)
    assert shared==58624
    assert independent>shared

def test_context_workspace():
    assert reference_workspace_bytes(context=64,hidden=128,heads=4,rank=16,intermediate=256)==8704
    assert reference_workspace_bytes(context=8192,hidden=128,heads=4,rank=16,intermediate=256)==658944
