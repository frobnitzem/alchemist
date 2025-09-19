import torch

# First define number formats used in forward and backward quantization
from qtorch import FixedPoint, FloatingPoint
forward_num = FixedPoint(wl=32, fl=18)
#backward_num = FloatingPoint(exp=32, man=18)

# Create a quantizer
from qtorch.quant import Quantizer
Q = Quantizer(forward_number=forward_num, #backward_number=backward_num,
              forward_rounding="nearest", #backward_rounding="stochastic"
             )
from torch import nn
from torch.functional import F

from .mixture import MixtureNormal
from .modules import FNN

class LeapFrog(nn.Module):
    def __init__(self, en, dt=0.001):
        super().__init__()
        self.en = en
        self.dt = dt

    def force(self, r, t):
        return -1*self.en.diff(r, t)

    def forward(self, x, inverse=False):
        if inverse:
            x['r'] = x['r'] - Q(self.dt*x['p'])
            x['t'] = x['t'] - self.dt
            frc = self.force(x['r'], x['t'])
            x['p'] = x['p'] - Q(self.dt*frc)
        else:
            frc = self.force(x['r'], x['t'])
            x['p'] = x['p'] + Q(self.dt*frc)
            x['r'] = x['r'] + Q(self.dt*x['p'])
            x['t'] = x['t'] + self.dt
        if len(x['r'].shape) == 2:
            batches = 1
        else:
            batches = len(x['r'])
        return x, torch.zeros(batches)

class MultiStep(nn.Module):
    def __init__(self, step, n):
        super().__init__()
        self.step = step
        self.n = n

    def forward(self, x, inverse=False):
        logJ = 0.0
        for i in range(self.n):
            x, lJ = self.step(x, inverse=inverse)
            logJ += lJ
        return x, logJ

def train(dataset, H0, process, loss_prior, beta=1.0):
    """ Learn on a stream of data and yield
        the loss after each data element.
        Each data element should contain a data batch of samples.
    """
    import torch.optim as optim

    #for p in process.parameters():
    #    break
    optimizer = optim.Adam(process.parameters(), lr=0.01)

    for x in dataset:
        batch_size = len(x['r'])
        process.zero_grad()
        x0, logJ = process(x)
        loss = loss_prior() + (beta*H0(x0) - logJ)/batch_size
        yield loss.item()

        loss.backward()
        #print(p.grad) # verified is non-zero, O(1e-5 though)
        optimizer.step()

def gen_points(elems, mixt, beta):
    # for each batch of elems, yield a batch of coordinates
    sigma = beta**-0.5
    normal = torch.distributions.normal.Normal(0, sigma)
    for z in elems:
        #print(z)
        r = mixt(z)
        #print(r)
        x = {'r': Q(r),
             'p': Q(normal.sample(r.shape)),
             't': 0.0,
            }
        yield x

