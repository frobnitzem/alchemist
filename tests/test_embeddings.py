import pytest
import torch

from alchemist.embeddings import HermiteSpline

@pytest.mark.parametrize("dim", [
        (2),
        (3),
        (10),
    ])
def test_hermite_spline(dim):
    H = HermiteSpline(dim, 5.0)
    b = H.basis_fns( torch.Tensor([0.4]) )
    assert len(b) == 3
    r = torch.Tensor([[0.1, 4.0], [4.9, 2.3]])
    f = H(r)
    assert f.shape == (2,2,dim)

def test_hermite():
    H = HermiteSpline(2, 4.0)

    for r in [torch.arange(10).reshape((2,5))*0.1+1.05,
              torch.arange(4)*0.2+0.8]:
        print(r/H.dr)
        x = H.interpolate(r)
        print(x)
        print(x.to_dense())
        print()
