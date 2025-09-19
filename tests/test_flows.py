import pytest
import torch

from alchemist.flows import LeapFrog, Q
from alchemist.modules import FNN

@pytest.mark.parametrize("dim", [
        (3),
        (8),
        (10),
    ])
def test_diff(dim, t=0.0):
    r = torch.rand((4, dim), requires_grad=True)
    N = FNN(dim)
    E = N(r, t)
    dE = N.diff(r, t)
    print(dE)
    E.sum().backward()
    print(r.grad)
    err = torch.abs(dE - r.grad).max().item()
    print(err)
    assert err < 1e-7

@pytest.mark.parametrize("dim, batch_size", [
        (3, 8),
        (8, 7),
        (10, 128),
    ])
def test_reverse(dim, batch_size):
    leap = LeapFrog( FNN(dim) )

    normal = torch.distributions.normal.Normal(0, 1)
    def gen():
        return normal.sample((batch_size, dim))
    x = {'r': Q(gen()),
         'p': Q(gen()),
         't': 0.0,
        }

    x0 = x['r']
    print(x0)
    logJ = 0.0
    for i in range(10):
        x, lJ = leap(x)
        logJ += lJ
    for i in range(10):
        x, lJ = leap(x, inverse=True)
        logJ += lJ
    print(x['r'])
    print(x['t'])
    print(logJ)
    err = torch.abs(x0 - x['r']).max().item()
    assert err == 0.0
    assert torch.abs(logJ).max() < 1e-8
    assert x['t'] == 0.0

def run_tests():
    test_diff(2)
    test_reverse()
