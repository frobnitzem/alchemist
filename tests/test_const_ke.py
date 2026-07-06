from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import pytest
import torch
import matplotlib.pyplot as plt

from alchemistlib.flows import LeapFrog, MultiStep, Q, fix_kT
from alchemistlib.modules import FNN

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
    x0 =  x.copy()
    for i in range(100):
        x, lJ, info = leap(x)
        logJ += lJ
        r = x['r']
        p = x['p']
        r2 = (r*r).sum((-2,-1))/Ndof
        p2 = (p*p).sum((-2,-1))/Ndof
        # print( r2, p2, logJ )
    loss = 1/const_kT*(U(r,x['t'])-U(x0['r'],0)) - logJ
    print(loss)
    return loss[0].item()

test_ham_change(const_kT=1.0, slope=3)
# ends @ tensor([0.1051, 0.1227], grad_fn=<DivBackward0>) tensor([1.0000, 1.0000], grad_fn=<DivBackward0>) tensor([-6815.7861, -7353.7925], grad_fn=<AddBackward0>)

kT_range = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
kT_losses = []
kT_vars = []

slope_range = [1.5, 2, 2.5, 3.0, 3.5, 4.0, 5.5, 6.0]
slope_losses = []
slope_vars = []

for const_kT in kT_range:
    sub_losses = []
    for i in range(5):
        loss = test_ham_change(const_kT=const_kT, slope=3)
        sub_losses.append(loss)
    mean_loss = sum(sub_losses) / len(sub_losses)
    kT_losses.append(mean_loss)
    var_loss = (sum((l - mean_loss) ** 2 for l in sub_losses) / len(sub_losses))**0.5
    kT_vars.append(var_loss)

for slope in slope_range:
    sub_losses = []
    for i in range(5):
        loss = test_ham_change(const_kT=1.0, slope=slope)
        sub_losses.append(loss)
    mean_loss = sum(sub_losses) / len(sub_losses)
    slope_losses.append(mean_loss)
    var_loss = (sum((l - mean_loss) ** 2 for l in sub_losses) / len(sub_losses))**0.5
    slope_vars.append(var_loss)

plt.errorbar(kT_range, kT_losses, yerr=kT_vars, fmt='-o')
plt.xlabel("const_kT value")
plt.ylabel("Loss")
plt.title("Effect of const_kT on Loss")
plt.savefig("const_kT_loss.png")
plt.close()

plt.errorbar(slope_range, slope_losses, yerr=slope_vars, fmt='-o')
plt.xlabel("slope value")
plt.ylabel("Loss")
plt.title("Effect of slope on Loss")
plt.savefig("slope_loss.png")
plt.close()

# test_ham_change(const_kT=None, slope=0)
# ends @ tensor([0.5056, 0.5103], grad_fn=<DivBackward0>) tensor([2.0028, 1.9605], grad_fn=<DivBackward0>) tensor([0., 0.])

#test_ham_change(const_kT=None, slope=3)
# ends @ tensor([0.2415, 0.2281], grad_fn=<DivBackward0>) tensor([4.6418, 4.7776], grad_fn=<DivBackward0>) tensor([0., 0.])
# This is out of equilibrium, since the kinetic temp.
# of 4.7 would suggest p**2 = 4.7 ~ 8*r**2 ~ 1.8

#test_ham_change(const_kT=None, slope=1)
# ends @ tensor([0.3257, 0.3264], grad_fn=<DivBackward0>) tensor([3.1172, 3.1966], grad_fn=<DivBackward0>) tensor([0., 0.])
