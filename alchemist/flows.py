from typing import Optional

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
from .modules import FNN, auto_diff
from .neighbors import assemble_neighbor_features

def dot_x(x,y):
    return (x*y).sum((-2,-1))

def fix_kT(p, kT):
    Ndof = p.size(-2)*p.size(-1)
    p2 = dot_x(p, p)
    return p*torch.sqrt(Ndof / p2)[...,None,None]

class LeapFrog(nn.Module):
    def __init__(self, en, dt=0.001, const_kT: Optional[float] = None):
        super().__init__()
        self.en = en
        self.dt = dt
        if const_kT is not None:
            assert const_kT > 0.0, "const_kT value must be a temp."
        self.const_kT = const_kT

    def force(self, r, t):
        return -1*auto_diff(self.en, r, t)

    def forward(self, x, inverse=False, info={}):
        shape = x['p'].shape
        if inverse:
            x['r'] = x['r'] - Q(self.dt*x['p'])
            x['t'] = x['t'] - self.dt
            frc = self.force(x['r'], x['t'])

            if self.const_kT:
                Ndof = shape[-1]*shape[-2]
                # FIXME: parameterize whole integrator by |p|**2 value
                # (f+p)/a = mom
                # p = a*mom-f
                # p**2 = a**2 mom**2 - 2a mom*f + f**2
                # 0 = a**2 nrm/2 - a mom*f + (f**2-p**2)/2
                nrm = dot_x(x['p'], x['p'])
                f2 = self.dt**2 * dot_x(frc, frc)
                b = self.dt*dot_x(frc, x['p'])
                disc = b*b + nrm*(self.const_kT*Ndof-f2)
                a = (b + torch.sqrt(disc)) / nrm
                x['p'] = Q(a[...,None,None]*x['p']) - Q(self.dt*frc)
                #print(a, b, dot_x(x['p'], x['p']))
                lJ = Ndof*torch.log(a)
            else:
                x['p'] = x['p'] - Q(self.dt*frc)
                lJ = torch.zeros(shape[:-2])
        else:
            frc = self.force(x['r'], x['t'])
            momentum = x['p'] + Q(self.dt*frc)
            if self.const_kT:
                Ndof = shape[-1]*shape[-2]
                #ke = dot_x(x['p'], x['p'])
                ke  = self.const_kT*Ndof
                ke2 = dot_x(momentum, momentum)
                fac = torch.sqrt(ke/ke2)
                #print(1/fac)
                x['p'] = momentum * fac[...,None,None]
                lJ = Ndof*torch.log(fac)
            else:
                x['p'] = momentum
                lJ = torch.zeros(shape[:-2])
            x['r'] = x['r'] + Q(self.dt*x['p'])
            x['t'] = x['t'] + self.dt

        return x, lJ, {}

class GlowBlock(nn.Module):
    def __init__(self, neighborlists, dim, dt=0.001, network_dims = [16]):
        super().__init__()
        self.dt = dt
        self.neighborlists = neighborlists

        # Total neighbors = 1 (self) + sum of neighbors in each shell
        # For GaAs: 1 + 4 + 12 = 17
        num_neighbors = 1 + sum(nl.shape[1] for nl in neighborlists)

        self.step1 = self.make_net(num_neighbors * dim, network_dims, dim)
        self.step2 = self.make_net(num_neighbors * dim, network_dims, dim)
        self.step3 = self.make_net(num_neighbors * dim, network_dims, dim)
        self.step4 = self.make_net(num_neighbors * dim, network_dims, dim)

        self.step1.apply(self.zero_init)
        self.step2.apply(self.zero_init)
        self.step3.apply(self.zero_init)
        self.step4.apply(self.zero_init)

    def make_net(self, in_dim, hidden_dims, out_dim):
        layers = []
        dims = [in_dim] + hidden_dims + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:   # no ReLU after final layer
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def zero_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def neighbor_expansion(self, r, nbr=None):
        # Uses the shared utility to gather features and flattens to (B,N,-1)
        return assemble_neighbor_features(r, self.neighborlists).reshape(r.shape[0], r.shape[1], -1)

    def st1(self, r, t):
        s = self.step1(r).clamp(-5,5)
        t = self.step2(r)
        return (s,t)

    def st2(self, r, t):
        s = self.step3(r).clamp(-5,5)
        t = self.step4(r)
        return (s,t)


    def forward(self, x, inverse=False, info={}):
        if inverse:
            s, t = self.st2(self.neighbor_expansion(x['r']), x['t'])
            x['p'] = (x['p'] - t)*torch.exp(-s)

            lJ = -s.sum(dim=(1,2))

            s, t = self.st1(self.neighbor_expansion(x['p']), x['t'])
            x['r'] = (x['r'] - t)*torch.exp(-s)

            x['t'] = x['t'] - self.dt
            lJ += -s.sum(dim=(1,2))
        else:
            s, t = self.st1(self.neighbor_expansion(x['p']), x['t'])
            x['r'] = x['r']*torch.exp(s) + t

            lJ = s.sum(dim=(1,2))

            s, t = self.st2(self.neighbor_expansion(x['r']), x['t'])
            x['p'] = x['p']*torch.exp(s) + t

            x['t'] = x['t'] + self.dt
            lJ += s.sum(dim=(1,2))

        return x, lJ, {}

class simpleGlowBlock(nn.Module):
    def __init__(self, dim, dt=0.001, network_dims = [16]):
        super().__init__()
        self.dt = dt

        self.step1 = self.make_net(dim, network_dims, dim)
        self.step2 = self.make_net(dim, network_dims, dim)
        self.step3 = self.make_net(dim, network_dims, dim)
        self.step4 = self.make_net(dim, network_dims, dim)

        self.initialize_coupling_net(self.step1)
        self.initialize_coupling_net(self.step2)
        self.initialize_coupling_net(self.step3)
        self.initialize_coupling_net(self.step4)

    def make_net(self, in_dim, hidden_dims, out_dim):
        layers = []
        dims = [in_dim] + hidden_dims + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:   # no ReLU after final layer
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    # def zero_init(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.zeros_(m.weight)
    #         nn.init.zeros_(m.bias)

    def initialize_coupling_net(self, net: nn.Sequential) -> None:
        linear_layers = [
            module for module in net.modules()
            if isinstance(module, nn.Linear)
        ]

        for layer in linear_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        # Zero only the final output layer.
        nn.init.zeros_(linear_layers[-1].weight)
        nn.init.zeros_(linear_layers[-1].bias)

    def st1(self, r, t):
        #implement NN to calculate s,t fom r,t
        s = self.step1(r).clamp(-5,5)
        t = self.step2(r)
        return (s,t)

    def st2(self, r, t):
        #implement NN to calculate s,t fom r,t
        s = self.step3(r).clamp(-5,5)
        t = self.step4(r)
        return (s,t)


    def forward(self, x, inverse=False, info={}):
        if inverse:
            s, t = self.st2(x['r'], x['t'])
            x['p'] = (x['p'] - t)*torch.exp(-s)

            lJ = -s.sum(dim=(1,2))

            s, t = self.st1(x['p'], x['t'])
            x['r'] = (x['r'] - t)*torch.exp(-s)

            x['t'] = x['t'] - self.dt
            lJ += -s.sum(dim=(1,2))
        else:
            s, t = self.st1(x['p'], x['t'])
            x['r'] = x['r']*torch.exp(s) + t
            # print(x['r'])

            lJ = s.sum(dim=(1,2))

            s, t = self.st2(x['r'], x['t'])
            x['p'] = x['p']*torch.exp(s) + t

            x['t'] = x['t'] + self.dt
            lJ += s.sum(dim=(1,2))

        return x, lJ, {}

class Brownian(nn.Module):
    def __init__(self, en, sigma, beta):
        super().__init__()
        self.en = en
        self.beta = beta
        self.sigma = sigma
        self.gamma = 0.5*beta*sigma**2

    def forward(self, x, inverse=False, info={}):
        shape = x['r'].shape

        if 'dE' not in info:
            E, dE = auto_diff(self.en, x['r'], x['t'],
                              return_E=True)
            info['E'] = E
            info['dE'] = dE

        Z = self.sigma * torch.randn_like(x['r'])
        #dx = Q(-self.gamma*info['dE'] + Z)
        dx = -self.gamma*info['dE'] + Z
        x['r'] = x['r'] + dx
        x['t'] = x['t'] + self.dt

        E, dE = auto_diff(self.en, x['r'], x['t'], return_E=True)
        #lJ = ( dot_x(dx - self.gamma*dE, dx - self.gamma*dE)
        #     - dot_x(dx + self.gamma*info['dE'],
        #             dx + self.gamma*info['dE'])
        #     ) / (2*self.sigma**2)
        dE2 = 0.5*(dE + info['dE'])
        lJ = self.beta*dot_x(self.gamma*dE2 - Z, dE2)

        #c = 0.5*self.gamma*(dE - info['dE'])
        #lJ = self.beta*dot_x(dE2, self.gamma*dE2)

        # store next info cache
        info['E']  = E
        info['dE'] = dE

        return x, lJ, cache


class MultiStep(nn.Module):
    def __init__(self, step, n):
        super().__init__()
        self.step = step
        self.n = n

    def forward(self, x, inverse=False, info={}):
        logJ = 0.0
        for i in range(self.n):
            x, lJ, info = self.step(x, inverse=inverse, info=info)
            logJ += lJ
        return x, logJ, info
