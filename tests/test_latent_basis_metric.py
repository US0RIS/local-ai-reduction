import torch


def test_inverse_gram_recovers_orthogonal_projection_for_nonorthogonal_rows():
    torch.manual_seed(17)
    r,d=5,11
    b=torch.randn(r,d)
    # Make row geometry intentionally far from orthonormal.
    b[1]*=3.0
    b[2]+=0.7*b[0]
    ginv=torch.linalg.inv(b@b.T)
    x=torch.randn(19,d)

    # Latent encode + corrected reconstruction used by the value path.
    latent=x@b.T
    corrected=latent@ginv@b

    # Canonical orthogonal projector onto row(B).
    projector=b.T@ginv@b
    reference=x@projector
    assert torch.allclose(corrected,reference,atol=1e-5,rtol=1e-5)

    # Key/query latent score uses the same metric and must equal the projected
    # full-space dot product.
    q=torch.randn(d)
    k=torch.randn(d)
    ql=q@b.T
    kl=k@b.T
    corrected_score=ql@ginv@kl
    reference_score=(q@projector)@(k@projector)
    assert torch.allclose(corrected_score,reference_score,atol=1e-5,rtol=1e-5)
