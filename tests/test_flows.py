import pytest
import torch
from torch import nn

from alchemist.flows import LeapFrog, Q, fix_kT
from alchemist.modules import FNN

class ReshapeOut(nn.Module):
    def __init__(self, fn, sh):
        super().__init__()
        self.fn = fn
        self.sh = sh
    def forward(self, x, *args):
        return self.fn(x, *args).reshape(self.sh)
    def diff(self, x, *args):
        return self.fn.diff(x, *args)

@pytest.mark.parametrize("dim", [
        (3),
        (8),
        (10),
    ])
def test_diff(dim, t=0.0):
    # TODO: use torch.gradcheck
    # 
    r = torch.rand((4, dim), requires_grad=True)
    N = ReshapeOut(FNN(dim), (-1,))
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
    leap = LeapFrog( ReshapeOut(FNN(dim), (-1,)) )

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
    info = {}
    for i in range(10):
        x, lJ, info = leap(x, info=info)
        logJ += lJ
    info = {}
    for i in range(10):
        x, lJ, info = leap(x, inverse=True, info=info)
        logJ += lJ
    print(x['r'])
    print(x['t'])
    print(logJ)
    err = torch.abs(x0 - x['r']).max().item()
    assert err == 0.0
    assert torch.abs(logJ).max().item() < 1e-8
    assert x['t'] == 0.0

def test_const_ke(batch_size=2, Na=4):
    def U(x, t):
        return (1+t)*(x*x).sum((-2,-1))

    leap = LeapFrog( U, const_kT = 1.0 )

    normal = torch.distributions.normal.Normal(0, 1)
    def gen():
        return normal.sample((batch_size, Na, 3))
    x = {'r': Q(gen()),
         'p': Q(fix_kT(gen(), 1.0)),
         't': 0.0,
        }

    x0 = x['r']
    print(x0)
    logJ = 0.0
    info = {}
    for i in range(10):
        x, lJ, info = leap(x, info=info)
        logJ += lJ
    info = {}
    for i in range(10):
        x, lJ, info = leap(x, inverse=True, info=info)
        logJ += lJ
    print(x['r'])
    print(x['t'])
    print(logJ)
    err = torch.abs(x0 - x['r']).max().item()
    assert err == 0.0
    assert torch.abs(logJ).max().item() < 1e-4
    assert x['t'] == 0.0

def run_tests():
    test_diff(2)
    test_reverse()
