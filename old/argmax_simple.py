import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from alchemistlib.flows import simpleGlowBlock, MultiStep, Q, fix_kT
from alchemistlib.modules import FNN
from utils import _format_pdb_atom, _write_pdb_trajectory, _read_pdb_coords

import matplotlib.pyplot as plt

def generate_sample(batch_size, Na, dim, sigma):
    normal = torch.distributions.Normal(0, sigma)
    categories = (torch.rand(batch_size, Na) > 0.25).int()
    x = {
        'r': torch.stack([categories, 1 - categories], dim=2),  # dim (batch_size, Na, dim) where the seond dim is 1-first dim
        'p': Q(fix_kT(normal.sample((batch_size, Na, dim)), 1.0)),
        't': 0.0
    }

    return x


def calc_loss(x, x0, logJ, z, U = None, kT = 1.0, sigma = 1, losstype = "KL"):
    if losstype == "KL":
        return 1 / kT * (U(x['r'], x['t']) - U(x0['r'], 0.0)) - logJ
    elif losstype == "argmax":
        y = (1/(1+torch.exp(-x['r'][:,:,0]))).sum(dim=1)
        log_pz = -0.5 * ((z['r'] / sigma) ** 2).sum(dim=[1, 2])
        log_pz += -0.5 * z['r'][0].numel() * math.log(2 * math.pi * sigma ** 2)
        return -log_pz - logJ + 1/(y*(1-y))

def sample_v(x):

    u = torch.rand(x.shape[0], x.shape[1], x.shape[2])
    v = -torch.log(1/u - 1)
    k = torch.argmax(v, dim=2)
    cond = (k != x[:, :, 0]).unsqueeze(-1)
    v = torch.where(cond, torch.flip(v, dims=[2]), v)

    return v
    # u = torch.rand(2) 
    # v = -torch.log(1/u-1)
    # k = torch.argmax(v)
    # if not k == x: v = torch.flip(v, dims=[0])
    # return v

# ----------------------------------------------------------------------
# Training GlowBlock on GaAs energy
# ----------------------------------------------------------------------
def train_glowblock_two_part(
    batch_size=1,
    Na=216,
    dim=2,
    sigma=1.0,
    kT=1.0,
    periodic=True,
    n_steps_flow=1,
    epoch_count=25,
    lr=1e-3,
    network_dims=[16],
):
    glow = simpleGlowBlock(dt=0.001, network_dims=network_dims, dim=dim)
    flow = MultiStep(glow, n_steps_flow)

    optimizer = optim.Adam(glow.parameters(), lr=lr)
    losses = []

    for it in range(epoch_count):
        x = generate_sample(batch_size, Na, dim, sigma)
        x0 =  x.copy()

        v = sample_v(x['r'])
        v_flow = x.copy()
        v_flow['r'] = v
        z, logJ, info = flow(v_flow, inverse = True)
        loss = calc_loss(x, x0, logJ, z, sigma=sigma, losstype="argmax")

        losses.append(loss.mean().item())
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        if (it + 1) % 50 == 0:
            print(f"[train] epoch {it+1:4d}  loss = {loss.mean().item():.4f}")

    return glow, losses


# ----------------------------------------------------------------------
# Testing GlowBlock: trajectory + PDB + histogram
# ----------------------------------------------------------------------
def test_glowblock_two_part(glow, batch_size=1, Na=216, dim=2, sigma=1.0, kT=1.0, periodic=True):
    flow = MultiStep(glow, 1)

    x = generate_sample(batch_size, Na, dim, sigma)
    x0 =  x.copy()

    Ga_percents = [x0['r'][:,:,0]]

    v = sample_v(x['r'])
    v_flow = x.copy()
    v_flow['r'] = v
    Ga_percents.append(torch.sigmoid(v_flow['r'][:,:,0]))
    z, logJ, info = flow(v_flow, inverse = True)
    loss = calc_loss(x, x0, logJ, z, sigma=sigma, losstype="argmax")

    Ga_percents.append(torch.sigmoid(z['r'][:,:,0]))

    return torch.stack(Ga_percents), loss.mean().item()


# ----------------------------------------------------------------------
# Main: train, test, write PDB, plot histogram
# ----------------------------------------------------------------------
if __name__ == "__main__":
    coords = _read_pdb_coords()
    # print(coords)
    folder = 'simpleargmax'
    batch_size = 3
    n_steps_flow = 1

    for epoch_count in [1000]:
        for network_dims in [[32,32]]:

            filename = f'{folder}/{len(network_dims)+1}layer_epoch{epoch_count}'

            glow, train_losses = train_glowblock_two_part(n_steps_flow=n_steps_flow, epoch_count=epoch_count, batch_size=batch_size, network_dims=network_dims)
            Ga_percents, test_loss = test_glowblock_two_part(glow,batch_size=batch_size)

            torch.save(glow.state_dict(), f'{filename}_model_weights.pt')

            As_percents = 1 - Ga_percents
            _OUTPUT_PDB = f'{filename}.pdb'
            print(Ga_percents.shape, As_percents.shape)
            _write_pdb_trajectory(_OUTPUT_PDB, coords, Ga_percents[:, 0], As_percents[:, 0])

            plt.plot(train_losses, label='train loss')
            plt.xlabel(f'Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.suptitle(f'GlowBlock Training Loss (epoch_count={epoch_count}, n_steps_flow={n_steps_flow})')
            plt.title(f'Test Loss: {test_loss:.4f}')
            plt.savefig(f'{filename}_loss.png', dpi=150)
            plt.close()