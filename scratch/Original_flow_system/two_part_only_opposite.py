from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
from pathlib import Path
import pytest
import torch

from alchemistlib.flows import LeapFrog, MultiStep, Q, fix_kT
from alchemistlib.modules import FNN

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

    E1 = (pA_ga * (f1_ga * 0.3 + f1_as * 0.1) +
          pA_as * (f1_ga * 0.1 + f1_as * 0.5))

    # E2 = (pA_ga * (f2_ga * -0.1 + f2_as * 0.0) +
    #       pA_as * (f2_ga * 0.0 + f2_as * -0.1))

    return E1.sum(dim=-1) #+ E2.sum(dim=-1)

def percentA(r):
    #Calculate the percentage of particles that are of type A based on their positions, using a sigmoid function to determine the probability of being type A or B.
    dr = r[:,:,1]-r[:,:,0]
    return torch.sigmoid(dr)


def _format_pdb_atom(serial, name, x, y, z, occupancy, bfactor):
    return (
        f"ATOM  {serial:5d}  {name:<3s} XTL     1"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}\n"
    )


def _write_pdb_trajectory(pdb_path, coords, ga_percents, as_percents):
    coords = coords.detach().cpu()
    if coords.ndim != 2:
        raise ValueError(f'Expected a 2D coordinate tensor, got shape {tuple(coords.shape)}')

    if coords.shape[-1] == 2:
        coords = torch.cat([coords, torch.zeros(coords.shape[0], 1, dtype=coords.dtype)], dim=-1)
    elif coords.shape[-1] != 3:
        raise ValueError(f'Expected 2D or 3D coordinates, got shape {tuple(coords.shape)}')

    ga_percents = ga_percents.detach().cpu()
    as_percents = as_percents.detach().cpu()

    if ga_percents.shape != as_percents.shape:
        raise ValueError('Ga and As percentage trajectories must have the same shape')

    with open(pdb_path, 'w') as pdb_file:
        pdb_file.write('REMARK multi-model trajectory with fixed coordinates\n')
        for frame_index, (ga_frame, as_frame) in enumerate(zip(ga_percents, as_percents), start=1):
            pdb_file.write(f'MODEL     {frame_index:4d}\n')
            for atom_index, (coord, ga_value, as_value) in enumerate(zip(coords, ga_frame, as_frame), start=1):
                pdb_file.write(
                    _format_pdb_atom(
                        atom_index,
                        'X',
                        float(coord[0]),
                        float(coord[1]),
                        float(coord[2]),
                        float(ga_value),
                        float(as_value),
                    )
                )
            pdb_file.write('ENDMDL\n')
        pdb_file.write('END\n')

_GAAS_PDB = "../../examples/GaAs/GaAs.pdb"
_GAAS_SCALE = 5.75


def _read_pdb_coords(pdb_path=_GAAS_PDB):
    coords = []
    with open(pdb_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                coords.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])

    if not coords:
        raise ValueError(f'No atom coordinates found in {pdb_path}')

    return torch.tensor(coords, dtype=torch.float64)


def _cluster_shells(distances, tolerance=1e-2):
    shells = []
    for value in torch.sort(distances[distances > 0]).values:
        if not shells or torch.abs(value - shells[-1][-1]) > tolerance:
            shells.append([value])
        else:
            shells[-1].append(value)

    return [torch.stack(shell).mean() for shell in shells]


def neighbor_masks(periodic=True, pdb_path=_GAAS_PDB):
    # Build the two neighbor shells directly from the GaAs PDB coordinates.
    # periodic=True gives 4 first-shell and 12 second-shell neighbors per site.
    crds = _read_pdb_coords(pdb_path)
    delta = crds[:, None, :] - crds[None, :, :]

    if periodic:
        num_atoms = crds.shape[0]
        num_cells = round((num_atoms / 8) ** (1 / 3))
        if num_cells**3 * 8 != num_atoms:
            raise ValueError(f'Unexpected GaAs atom count in {pdb_path}: {num_atoms}')
        box = torch.full((3,), float(num_cells) * _GAAS_SCALE, dtype=crds.dtype, device=crds.device)
        delta = delta - torch.round(delta / box) * box

    dist = torch.linalg.norm(delta, dim=-1)
    dist.fill_diagonal_(float('inf'))

    shells = _cluster_shells(dist[torch.isfinite(dist)])
    if len(shells) < 2:
        raise ValueError('Not enough distance shells to build neighbor masks')

    cut1 = 0.5 * (shells[0] + shells[1])
    cut2 = shells[1] + 0.5 * (shells[1] - shells[0]) if len(shells) < 3 else 0.5 * (shells[1] + shells[2])

    mask1 = dist < cut1
    mask2 = (dist >= cut1) & (dist < cut2)

    nbr1 = [mask1[i].nonzero(as_tuple=True)[0].tolist() for i in range(dist.shape[0])]
    nbr2 = [mask2[i].nonzero(as_tuple=True)[0].tolist() for i in range(dist.shape[0])]

    return nbr1, nbr2

def test_two_part(batch_size=1, Na=216, dim=2, sigma=1.0, periodic=False):
    nbr1, nbr2 = neighbor_masks(periodic=periodic)
    #print row sums of masks to verify that each atom has the expected number of neighbors in each shell

    nbr1_tensor = torch.tensor(nbr1, dtype=torch.long).unsqueeze(0)
    nbr2_tensor = torch.tensor(nbr2, dtype=torch.long).unsqueeze(0)

    def U(r, t):
        '''Calculate the potential energy of the system based on chemical potentials.
        r is a list of particle % compositions, a fraction A and fraction B
        neighbors of r are set
        Nearest neighbors 1 are direct neighbors of which there are 4
        Nearest neighbors 2 are neighbors of neighbors of which there are 4*3=12
        '''

        dr = percentA(r).unsqueeze(-1)   # (B, N, 1)
        dr_expanded = dr.expand(-1, -1, dr.shape[1])   # (B, N, N)
        # print(dr_expanded.shape, nbr1_tensor.shape, nbr2_tensor.shape)

        # First-shell neighbor values: (B, N, 4)
        nbr1_vals = torch.gather(dr_expanded, dim=1, index=nbr1_tensor).squeeze(-1)

        # Second-shell neighbor values: (B, N, 12)
        nbr2_vals = torch.gather(dr_expanded, dim=1, index=nbr2_tensor).squeeze(-1)

        # print(nbr1_vals.shape)
        return (atom_energy_mixed_batched(dr, nbr1_vals, nbr2_vals)).sum(1)

    kT = 1
    leap = MultiStep(LeapFrog(U), 10)

    normal = torch.distributions.normal.Normal(0, 1)
    def gen():
        return normal.sample((batch_size, Na, dim))
    
    def genordered():
        idx = (torch.arange(1, 55).repeat_interleave(4)) % 2
        base = torch.nn.functional.one_hot(idx, num_classes=2).float() #(N, 2)
        base = base.unsqueeze(0) #(1, N, 2)
        return base

    x = {'r': Q(gen())*sigma,
         'p': Q(fix_kT(gen(), kT)),
         't': 0.0,
        }

    logJ = 0.0

    r = x['r']
    x0 =  x.copy()

    Ga_percents = [percentA(x0['r'])]

    for i in range(5000):
        x, lJ, info = leap(x)
        logJ += lJ
        r = x['r']
        Ga_percents.append(percentA(r))
        if i % 50 == 0:
            print(f"Step {i}: r[0,0] = {r[0,0]}, logJ = {logJ}")
    loss = 1/kT*(U(r,x['t'])-U(x0['r'],0)) - logJ
    print(loss)
    return Ga_percents


coords = _read_pdb_coords()
print(coords)
Ga_percents = test_two_part(periodic=True)
Ga_percents = torch.stack(Ga_percents)
As_percents = 1 - Ga_percents

_OUTPUT_PDB = 'leapfrogresults/alternating_newEnergy.pdb'
_write_pdb_trajectory(_OUTPUT_PDB, coords, Ga_percents[:, 0], As_percents[:, 0])

import matplotlib.pyplot as plt

#Save a histogram of the log-weights to visualize their distribution after processing the samples through the simulation.
plt.hist(Ga_percents[-1, 0].flatten().cpu().detach().numpy(), bins=5)
plt.xlabel("Percent Ga")
plt.ylabel("count")
plt.title("Histogram of Percent Ga values")
plt.tight_layout()
plt.savefig("percentGa_histogramalt.png", dpi=150)
plt.close()