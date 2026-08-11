from typing import Optional

import torch, copy

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
    return p*torch.sqrt(Ndof * kT / p2)[...,None,None]

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
    def __init__(self, dim, data_expansion = lambda r: r, data_size=1, dt=0.001, hidden_dims = [16]):
        super().__init__()
        self.dt = dt
        self.data_expansion = data_expansion

        self.step1 = self.make_net(data_size * dim+1, hidden_dims, dim)
        self.step2 = self.make_net(data_size * dim+1, hidden_dims, dim)
        self.step3 = self.make_net(data_size * dim+1, hidden_dims, dim)
        self.step4 = self.make_net(data_size * dim+1, hidden_dims, dim)

        self.initialize_coupling_net(self.step1)
        self.initialize_coupling_net(self.step2)
        self.initialize_coupling_net(self.step3)
        self.initialize_coupling_net(self.step4)

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

    def make_net(self, in_dim, hidden_dims, out_dim):
        layers = []
        dims = [in_dim] + hidden_dims + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:   # no ReLU after final layer
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def st1(self, r, tau):
        #add time as an input to the network
        tau = tau.expand(r.size(0))  # now shape (B,)
        r_time = torch.cat([r, tau[:, None, None].expand(-1, r.size(1), 1)], dim=-1)
        s = self.step1(r_time).clamp(-4,4)
        t = self.step2(r_time)
        return (s,t)

    def st2(self, r, tau):
        tau = tau.expand(r.size(0))  # now shape (B,)
        r_time = torch.cat([r, tau[:, None, None].expand(-1, r.size(1), 1)], dim=-1)
        s = self.step3(r_time).clamp(-4,4)
        t = self.step4(r_time)
        return (s,t)

    def forward(self, x, inverse=False, info={}):
        if inverse:
            x['t'] = x['t'] - self.dt

            s, t = self.st2(self.data_expansion(x['r']), x['t'])
            x['p'] = (x['p'] - t)*torch.exp(-s)

            lJ = -s.sum(dim=(1,2))

            s, t = self.st1(self.data_expansion(x['p']), x['t'])
            x['r'] = (x['r'] - t)*torch.exp(-s)

            lJ += -s.sum(dim=(1,2))
        else:
            s, t = self.st1(self.data_expansion(x['p']), x['t'])
            x['r'] = x['r']*torch.exp(s) + t

            lJ = s.sum(dim=(1,2))

            s, t = self.st2(self.data_expansion(x['r']), x['t'])
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

class MultiIndependent(nn.Module):
    """
    Apply n independently trained copies of a given step module.
    
    Each layer is a deep copy of `step`, so all parameters are independent.
    """
    def __init__(self, step, n):
        super().__init__()
        self.n = n

        # Make n independent copies of the step module
        self.layers = nn.ModuleList([copy.deepcopy(step) for _ in range(n)])

    def forward(self, x, inverse=False, info={}):
        logJ = 0.0

        for layer in self.layers:
            x, lJ, info = layer(x, inverse=inverse, info=info)
            logJ += lJ

        return x, logJ, info

class glow_and_verlet_block(nn.Module):
    def __init__(self, dim, data_expansion = lambda r: r, data_size=1, dt=0.001, hidden_dims = [16]):
        super().__init__()
        self.glow = GlowBlock(dim, data_expansion=data_expansion, data_size=data_size, dt=0, hidden_dims=hidden_dims)
        self.verlet = LeapFrog(self.glow, dt=dt)

    def forward(self, x, inverse=False, info={}):
        #assume x['r'] is structured as dim chemical identities then +3 for coordinates
        #only update the coordinates in verlet, only update the chemical identities in glow
        x_coord  = {
            'r': x['r'][..., -3:],
            'p': x['p'][..., -3:],
            't': x['t']
        }
        x_chem = {
            'r': x['r'][..., :-3],
            'p': x['p'][..., :-3],
            't': x['t']
        }
        if inverse:
            x_coord, lJ1, info = self.verlet(x_coord, inverse=inverse, info=info)
            x_chem, lJ2, info = self.glow(x_chem, inverse=inverse, info=info)
            lJ = lJ1 + lJ2
        else:
            x_chem, lJ1, info = self.glow(x_chem, inverse=inverse, info=info)
            x_coord, lJ2, info = self.verlet(x_coord, inverse=inverse, info=info)
            lJ = lJ1 + lJ2
        x = {
            'r': torch.cat([x_chem['r'], x_coord['r']], dim=-1),
            'p': torch.cat([x_chem['p'], x_coord['p']], dim=-1),
            't': x_coord['t']
        }
        return x, lJ, info

class RealNVP(nn.Module):
    def __init__(self, DIM,NA, hidden_dims=[64,64], n_layers=10, dt = 0.001, do_cartesian=False):
        """
        dim: number of coordinate dimensions (e.g., 3N)
        hidden_dims: list of hidden layer sizes for s,t networks
        n_layers: number of RealNVP coupling layers (default = 10)
        """
        super().__init__()
        self.n_layers = n_layers
        self.NA = NA
        self.DIM = DIM
        self.dt = dt
        self.do_cartesian = do_cartesian

        # Split dimension in half
        self.split = NA // 2

        if do_cartesian:
            self.s_netr = self.make_net((DIM+3)*self.split+1, hidden_dims, (self.split)*(DIM+3))
            self.t_netr = self.make_net((DIM+3)*self.split+1, hidden_dims, (self.split)*(DIM+3))
            self.s_netl = self.make_net((DIM+3)*self.split+1, hidden_dims, (self.split)*(DIM+3))
            self.t_netl = self.make_net((DIM+3)*self.split+1, hidden_dims, (self.split)*(DIM+3))
        else:
            self.s_netr = self.make_net((DIM)*self.split+1, hidden_dims, (self.split)*DIM)
            self.t_netr = self.make_net((DIM)*self.split+1, hidden_dims, (self.split)*DIM)
            self.s_netl = self.make_net((DIM)*self.split+1, hidden_dims, (self.split)*DIM)
            self.t_netl = self.make_net((DIM)*self.split+1, hidden_dims, (self.split)*DIM)

        # Initialize final layer to zero (same as GlowBlock)
        self.initialize_coupling_net(self.s_netr)
        self.initialize_coupling_net(self.t_netr)
        self.initialize_coupling_net(self.s_netl)
        self.initialize_coupling_net(self.t_netl)

    def make_net(self, in_dim, hidden_dims, out_dim):
        layers = []
        dims = [in_dim] + hidden_dims + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def initialize_coupling_net(self, net):
        linear_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
        for layer in linear_layers[:-1]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(linear_layers[-1].weight)
        nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, x, inverse=False, info={}):
        """
        x: dict with keys 'r', 'p', 't'
        Only r is transformed by RealNVP.
        """
        if not self.do_cartesian: #only include chemical identities in RealNVP, not coordinates
            r_chem = x['r']
            r_coord = x['r_coord']
            r = torch.cat([r_chem, r_coord], dim=-1)
            #reshape into per batch dimensions for RealNVP
            logJ = torch.zeros(r_chem.shape[0], device=r_chem.device)
            
            for i in range(self.n_layers):
                if inverse:
                    i = i + 1
                    x['t'] = x['t'] - self.dt
                # Alternating mask
                if i % 2 == 0:
                    r1 = r[:, :self.split, :]
                    r2 = r_chem[:, self.split:, :]
                    r_coordpart = r_coord[:, self.split:, :]
                else:
                    r2 = r_chem[:, :self.split, :]
                    r1 = r[:, self.split:, :]
                    r_coordpart = r_coord[:, :self.split, :]
                r1 = r1.reshape(r1.shape[0], -1)
                r2 = r2.reshape(r2.shape[0], -1)

                # Compute s,t
                t = x['t'].expand(r1.size(0))  # now shape (B,)
                r1_time = torch.cat([r1, t[:, None].expand(-1, 1)], dim=-1)
                if i % 2 == 0:
                    s = self.s_netr(r1_time).clamp(-4, 4)
                    t = self.t_netr(r1_time)
                else:
                    s = self.s_netl(r1_time).clamp(-4, 4)
                    t = self.t_netl(r1_time)

                if inverse:
                    r2 = (r2 - t) * torch.exp(-s)
                    logJ += (-s).sum(dim=-1)
                else:
                    r2 = r2 * torch.exp(s) + t
                    logJ += s.sum(dim=-1)
                    x['t'] = x['t'] + self.dt

                r1 = r1.reshape(r1.shape[0], -1, self.DIM + 3)
                r2 = r2.reshape(r2.shape[0], -1, self.DIM)
                r2 = torch.cat([r2, r_coordpart], dim=-1)  # Reattach coordinates

                # Reassemble
                if i % 2 == 0:
                    r = torch.cat([r1, r2], dim=-2)
                else:
                    r = torch.cat([r2, r1], dim=-2)
                    
                r_chem = r[:,:,:-3]

            x['r'] = r_chem  # Only keep chemical identities
            return x, logJ, info
        else:
            # If do_cartesian is True, apply RealNVP to the entire r tensor
            r = x['r']
            logJ = torch.zeros(r.shape[0], device=r.device)
            
            for i in range(self.n_layers):
                if inverse:
                    i = i + 1
                    x['t'] = x['t'] - self.dt
                # Alternating mask
                if i % 2 == 0:
                    r1 = r[:, :self.split, :]
                    r2 = r[:, self.split:, :]
                else:
                    r2 = r[:, :self.split, :]
                    r1 = r[:, self.split:, :]
                r1 = r1.reshape(r1.shape[0], -1)
                r2 = r2.reshape(r2.shape[0], -1)

                # Compute s,t
                t = x['t'].expand(r1.size(0))  # now shape (B,)
                r1_time = torch.cat([r1, t[:, None].expand(-1, 1)], dim=-1)
                if i % 2 == 0:
                    s = self.s_netr(r1_time).clamp(-4, 4)
                    t = self.t_netr(r1_time)
                else:
                    s = self.s_netl(r1_time).clamp(-4, 4)
                    t = self.t_netl(r1_time)

                if inverse:
                    r2 = (r2 - t) * torch.exp(-s)
                    logJ += (-s).sum(dim=-1)
                else:
                    r2 = r2 * torch.exp(s) + t
                    logJ += s.sum(dim=-1)
                    x['t'] = x['t'] + self.dt

                r1 = r1.reshape(r1.shape[0], -1, self.DIM + 3)
                r2 = r2.reshape(r2.shape[0], -1, self.DIM + 3)

                # Reassemble
                if i % 2 == 0:
                    r = torch.cat([r1, r2], dim=-2)
                else:
                    r = torch.cat([r2, r1], dim=-2)

            x['r'] = r
            return x, logJ, info
