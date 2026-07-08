import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from alchemistlib.flows import GlowBlock, MultiStep, Q, fix_kT
from alchemistlib.modules import FNN
from utils import _format_pdb_atom, _write_pdb_trajectory, neighbor_masks, _read_pdb_coords

import matplotlib.pyplot as plt

def atom_energy_mixed_batched(pA_ga, f1_ga, f2_ga):
    """
    pA_ga : (B, N, 1) tensor of Ga composition for each atom A
    f1_ga : (B, N, 4) tensor of Ga fractions among NN1
    f2_ga : (B, N, 12) tensor of Ga fractions among NN2

    Returns:
        (B, N) tensor of energies for each atom A in each batch
    """

    # Fractions of As
    pA_as = 1 - pA_ga
    f1_as = 1 - f1_ga
    f2_as = 1 - f2_ga

    E1 = (pA_ga * (f1_ga * 0.3 + f1_as * 0.5) +
          pA_as * (f1_ga * 0.5 + f1_as * 0.1))
    # E1 = (pA_ga * (f1_ga * 0.5 + f1_as * 0.1) + #Try to make it favor Ga-As interactions more than As-Ga interactions
    #        pA_as * (f1_ga * 0.1 + f1_as * 0.3))

    E2 = (pA_ga * (f2_ga * 0.15 + f2_as * 0.0) +
          pA_as * (f2_ga * 0.0 + f2_as * 0.05))

    return E1.sum(dim=-1) + E2.sum(dim=-1)

def percentA(r, percenttype = 'chempotential'):
    # Calculate the percentage of particles that are of type A
    if percenttype == 'chempotential':
        dr = r[:, :, 1] - r[:, :, 0]
        return torch.sigmoid(dr)
    elif percenttype == 'direct':
        return r[..., 0:1]  # Return the fraction of class 1 (Ga) directly

def build_U(batch_size, Na, dim, periodic=True, percenttype='chempotential'):
    nbr1, nbr2 = neighbor_masks(periodic=periodic)

    nbr1_tensor = torch.tensor(nbr1, dtype=torch.long).unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, M1)
    nbr2_tensor = torch.tensor(nbr2, dtype=torch.long).unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, M2)

    def U(r, t):
        """
        r: (B, Na, dim)
        t: scalar or (B,)
        """

        dr = percentA(r, percenttype=percenttype).unsqueeze(-1)          # (B, N, 1)
        dr_expanded1 = dr.expand(-1, -1, nbr1_tensor.shape[1])  # (B, N, M1)
        dr_expanded2 = dr.expand(-1, -1, nbr2_tensor.shape[1])  # (B, N, M2)

        # First-shell neighbor values: (B, N, 4)
        nbr1_vals = torch.gather(dr_expanded1, dim=1, index=nbr1_tensor)

        # Second-shell neighbor values: (B, N, 12)
        nbr2_vals = torch.gather(dr_expanded2, dim=1, index=nbr2_tensor)

        return atom_energy_mixed_batched(dr, nbr1_vals, nbr2_vals).sum(1)

    return U, [nbr1, nbr2]

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
def train_glowblock_two_part(
    batch_size=1,
    Na=216,
    dim=2,
    sigma=1.0,
    kT=1.0,
    periodic=True,
    n_steps_flow=1,
    num_batches=25,
    lr=1e-3,
    network_dims=[16],
):
    U, neighborlists = build_U(batch_size, Na, dim, periodic=periodic)
    glow = GlowBlock(dt=0.001, neighborlists=neighborlists, network_dims=network_dims, dim=dim)
    flow = MultiStep(glow, n_steps_flow)

    optimizer = optim.Adam(glow.parameters(), lr=lr)
    losses = []

    for it in range(num_batches):
        x = generate_sample(batch_size, Na, dim, sigma)
        x0 =  x.copy()

        x, logJ, info = flow(x)
        loss = calc_loss(x, x0, logJ, U, kT)

        losses.append(loss.mean().item())
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        if (it + 1) % 50 == 0:
            print(f"[train] batch {it+1:4d}  loss = {loss.mean().item():.4f}")

    return glow, losses


# ----------------------------------------------------------------------
# Testing GlowBlock: trajectory + PDB + histogram
# ----------------------------------------------------------------------
def test_glowblock_two_part(glow, batch_size=1, Na=216, dim=2, sigma=1.0, kT=1.0, periodic=True):
    U, neighborlists = build_U(batch_size, Na, dim, periodic=periodic)
    flow = MultiStep(glow, 1)

    x = generate_sample(batch_size, Na, dim, sigma)
    x0 = x.copy()

    Ga_percents = [percentA(x0['r'])]

    x, logJ, info = flow(x)
    Ga_percents.append(percentA(x['r']))

    loss = 1 / kT * (U(x['r'], x['t']) - U(x0['r'], 0.0)) - logJ

    return torch.stack(Ga_percents), loss.mean().item()


# ----------------------------------------------------------------------
# Main: train, test, write PDB, plot histogram
# ----------------------------------------------------------------------
if __name__ == "__main__":
    coords = _read_pdb_coords()
    # print(coords)
    folder = 'glowblockenergy'
    batch_size = 10
    n_steps_flow = 1

    for num_batches in [10000]:
        for kT in [1, 0.1, 0.01]:
            for network_dims in [[32,32]]:

                filename = f'{folder}/{len(network_dims)+1}layer_num_batches{num_batches}_kT{kT}'

                glow, train_losses = train_glowblock_two_part(n_steps_flow=n_steps_flow, num_batches=num_batches, batch_size=batch_size, network_dims=network_dims, kT=kT)
                Ga_percents, test_loss = test_glowblock_two_part(glow,batch_size=batch_size, kT=kT)

                torch.save(glow.state_dict(), f'{filename}_model_weights.pt')

                As_percents = 1 - Ga_percents
                _OUTPUT_PDB = f'{filename}.pdb'
                print(Ga_percents.shape, As_percents.shape)
                _write_pdb_trajectory(_OUTPUT_PDB, coords, Ga_percents[:, 0], As_percents[:, 0])

                plt.plot(train_losses, label='train loss')
                plt.xlabel(f'Batches')
                plt.ylabel('Loss')
                plt.legend()
                plt.suptitle(f'GlowBlock Training Loss (num_batches={num_batches}, n_steps_flow={n_steps_flow})')
                plt.title(f'Test Loss: {test_loss:.4f}')
                plt.savefig(f'{filename}_loss.png', dpi=150)
                plt.close()