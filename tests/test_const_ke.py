import pytest
import torch

from alchemist.flows import LeapFrog, MultiStep, Q, fix_kT
from alchemist.modules import FNN

def test_ham_change(batch_size=2, Na=1000, dim=3, slope=3.0,
                       const_kT = 1.0):
    # With const_kT, the r*r value above should go from 1 (initial)
    # to 1/[2*(1+slope)]. The velocity will be scaled down to remove
    # heat, so logJ should be negative and increasing

    # With no thermostat, the sample should heat up, so the
    # r*r value above should go from 1 (initial)
    # to kT_final/[2*(1+slope)]
    def U(r, t):
        return (1+slope*t)*(r*r).sum((-2,-1))

    kT = const_kT or 1.0
    leap = MultiStep(LeapFrog( U, const_kT = const_kT ), 10)

    normal = torch.distributions.normal.Normal(0, 1)
    def gen():
        return normal.sample((batch_size, Na, dim))
    x = {'r': Q(gen()),
         'p': Q(fix_kT(gen(), kT)),
         't': 0.0,
        }

    logJ = 0.0

    Ndof = Na*dim
    r = x['r']
    p = x['p']
    r2 = (r*r).sum((-2,-1))/Ndof
    p2 = (p*p).sum((-2,-1))/Ndof
    print( r2, p2, logJ )
    info = {}
    for i in range(100):
        x, lJ, info = leap(x, info=info)
        logJ += lJ
        r = x['r']
        p = x['p']
        r2 = (r*r).sum((-2,-1))/Ndof
        p2 = (p*p).sum((-2,-1))/Ndof
        print( r2, p2, logJ )

test_ham_change(const_kT=1.0, slope=3)
# ends @ tensor([0.1051, 0.1227], grad_fn=<DivBackward0>) tensor([1.0000, 1.0000], grad_fn=<DivBackward0>) tensor([-6815.7861, -7353.7925], grad_fn=<AddBackward0>)

#test_ham_change(const_kT=None, slope=0)
# ends @ tensor([0.5056, 0.5103], grad_fn=<DivBackward0>) tensor([2.0028, 1.9605], grad_fn=<DivBackward0>) tensor([0., 0.])

#test_ham_change(const_kT=None, slope=3)
# ends @ tensor([0.2415, 0.2281], grad_fn=<DivBackward0>) tensor([4.6418, 4.7776], grad_fn=<DivBackward0>) tensor([0., 0.])
# This is out of equilibrium, since the kinetic temp.
# of 4.7 would suggest p**2 = 4.7 ~ 8*r**2 ~ 1.8

#test_ham_change(const_kT=None, slope=1)
# ends @ tensor([0.3257, 0.3264], grad_fn=<DivBackward0>) tensor([3.1172, 3.1966], grad_fn=<DivBackward0>) tensor([0., 0.])
