import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from alchemistlib.flows import GlowBlock, MultiStep, Q, fix_kT
from alchemistlib.modules import FNN
from utils import _write_pdb_trajectory, _read_pdb_coords, build_crystal_U

import matplotlib.pyplot as plt

from traintest import train_glowblock_two_part, test_glowblock_two_part

def percentA(r, percenttype = 'chempotential'):
    # Calculate the percentage of particles that are of type A
    if percenttype == 'chempotential':
        dr = r[:, :, 1] - r[:, :, 0]
        return torch.sigmoid(dr)
    elif percenttype == 'direct':
        return r[..., 0:1]  # Return the fraction of class 1 (Ga) directly

def generate_sample(batch_size, Na, dim, sigma):
    normal = torch.distributions.normal.Normal(0, 1)
    x = {
        'r': Q(normal.sample((batch_size, Na, dim))) * sigma,
        'p': Q(fix_kT(normal.sample((batch_size, Na, dim)), 1.0)),
        't': 0.0
    }
    # percent = percentA(x['r'])
    # x['p'] = torch.stack([percent, 1 - percent], dim=2)

    return x


def calc_loss(x, x0, logJ, U, kT, losstype = "KL"):
    if losstype == "KL":
        return 1 / kT * (U(x['r'], x['t']) - U(x0['r'], 0.0)) - logJ

# ----------------------------------------------------------------------
# Training GlowBlock on GaAs energy
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Main: train, test, write PDB, plot histogram
# ----------------------------------------------------------------------
if __name__ == "__main__":
    coords = _read_pdb_coords()
    # print(coords)
    folder = 'glowblockenergyposter'
    batch_size = 10
    n_steps_flow = 1
    U, neighborlists = build_crystal_U(percentA,batch_size, 216, 2, periodic=True)

    for num_batches in [10000]:
        for kT in [1, 0.5, 0.1, 0.05, 0.01]:
            for network_dims in [[32,32]]:

                filename = f'{folder}/{len(network_dims)+1}layer_num_batches{num_batches}_kT{kT}'

                glow = GlowBlock(dt=0.001, neighborlists=neighborlists, network_dims=network_dims, dim=2)
                flow = MultiStep(glow, n_steps_flow)
                loss_func = lambda x, x0, logJ: calc_loss(x, x0, logJ, U, kT)

                glow, train_losses = train_glowblock_two_part(
                    flow=flow,
                    model=glow,
                    generate_sample=generate_sample,
                    loss_func=loss_func,
                    batch_size=batch_size,
                    Na=216,
                    dim=2,
                    sigma=1.0,
                    num_batches=num_batches,
                    lr=1e-3,
                )
                Ga_percents, test_loss = test_glowblock_two_part(
                    flow=flow,
                    generate_sample=generate_sample,
                    percent_func=percentA,
                    loss_func=loss_func,
                    batch_size=batch_size,
                    Na=216,
                    dim=2,
                    sigma=1.0,
                    compute_loss=True,
                )

                torch.save(glow.state_dict(), f'{filename}_model_weights.pt')

                As_percents = 1 - Ga_percents
                _OUTPUT_PDB = f'{filename}.pdb'
                print(Ga_percents.shape, As_percents.shape)
                _write_pdb_trajectory(_OUTPUT_PDB, coords, Ga_percents[:, 0], As_percents[:, 0])

                plt.figure(figsize=(5, 4))
                plt.plot(train_losses, label='train loss')
                plt.xlabel(f'Batches')
                plt.ylabel('Loss')
                plt.legend()
                plt.suptitle(f'GlowBlock Training Loss (num_batches={num_batches}, n_steps_flow={n_steps_flow})')
                plt.title(f'Test Loss: {test_loss:.4f}')
                plt.tight_layout()
                plt.savefig(f'{filename}_loss.png', dpi=150)
                plt.close()