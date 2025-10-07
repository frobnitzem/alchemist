""" Energy Computation NN Modules
"""
import torch
from torch import nn
from torch.functional import F

def auto_diff(fn, x, *args):
    x = x.detach().clone().requires_grad_(True)
    #x.requires_grad_(True)
    E = fn(x, *args)
    return torch.autograd.grad(E, x,
                grad_outputs=torch.ones_like(E), create_graph=True)[0]

class FNN(nn.Module):
    """ The FNN computes the energy (scalar) of an input, x.
        The diff() method computes dE(x)/dx, but don't bother
        implementing diff again.  The LeapFrog
        and other integrators are just going to use auto_diff
        from this module instead.
    """
    def __init__(self, dim):
        super().__init__()
        #self.conv1 = nn.Conv2d(1, 20, 5, 1)
        #self.conv2 = nn.Conv2d(20, 50, 5, 1)
        d1 = int(dim*1.3)
        d2 = int(dim*0.5)
        self.fc1 = nn.Linear(dim, d1)
        self.fc2 = nn.Linear(d1, d2)
        self.fc3 = nn.Linear(d2, 1)

    def forward(self, x, t):
        #x = F.relu(self.conv1(x))
        #x = F.max_pool2d(x, 2, 2)
        #x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2, 2)
        #x = x.view(-1, 4*4*50)
        x0 = self.fc1(x)
        x1 = F.softplus(x0)
        x2 = self.fc2(x1)
        x3 = F.softplus(x2)
        return self.fc3(x3)

    def diff(self, x, t):
        x.requires_grad_(True)
        E = self.forward(x, t)
        return torch.autograd.grad(E, x,
                grad_outputs=torch.ones_like(E), create_graph=True)[0]

    def diff1(self, x, t):
        x0 = self.fc1(x)      # B,d1
        x1 = F.softplus(x0)   # B,d1
        x2 = self.fc2(x1)     # B,d2
        # x3 = F.softplus(x2) # B,d2
        # E = self.fc3(x3)    # B,1

        # Essentially run backprop manually.
        # note: fc3.weight is (1,d2)
        u3 = self.fc3.weight*F.sigmoid(x2) # B,d2
        u2 = torch.mm(u3, self.fc2.weight) # B,d1
        u2 *= F.sigmoid(x0)
        dE = torch.mm(u2, self.fc1.weight) # B,dim
        return dE

    def diff_dot(self, x, t, f):
        # compute dot(dE/dx, f)
        _, vjpfunc = torch.func.vjp(self.forward, x, t)
        vjps = vjpfunc(f)
        return vjps[0]
