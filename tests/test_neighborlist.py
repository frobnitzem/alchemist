from pathlib import Path
import pytest
import torch
import random

from alchemist.flows import LeapFrog, MultiStep, Q, fix_kT
from alchemist.modules import FNN

_GAAS_PDB = Path(__file__).resolve().parents[1] / 'examples' / 'GaAs' / 'GaAs.pdb'
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
    return mask1.to(torch.float32), mask2.to(torch.float32)

mask1, mask2 = neighbor_masks(periodic = False)
#print row sums of masks to verify that each atom has the expected number of neighbors in each shell
print(mask1.sum(dim=-1))
print(mask2.sum(dim=-1))

mask1, mask2 = neighbor_masks(periodic = True)
#print row sums of masks to verify that each atom has the expected number of neighbors in each shell
print(mask1.sum(dim=-1))
print(mask2.sum(dim=-1))

_GAAS_PDB = Path(__file__).resolve().parents[1] / 'examples' / 'GaAs' / 'GaAs.pdb'

def read_pdb_lines(pdb_path=_GAAS_PDB):
    with open(pdb_path, 'r') as f:
        return f.readlines()

def write_pdb_lines(lines, out_path):
    with open(out_path, 'w') as f:
        f.writelines(lines)

def modify_b_factors(pdb_path=_GAAS_PDB, out_path="modified_GaAs.pdb"):
    # Load coordinates and neighbor masks
    coords = _read_pdb_coords(pdb_path)
    mask1, mask2 = neighbor_masks(periodic=True, pdb_path=pdb_path)

    N = coords.shape[0]

    # Pick a random atom index
    center = random.randrange(N)

    # Identify neighbors
    first_neighbors = mask1[center].nonzero(as_tuple=True)[0].tolist()
    second_neighbors = mask2[center].nonzero(as_tuple=True)[0].tolist()

    # Read original PDB
    lines = read_pdb_lines(pdb_path)

    # Prepare B-factor array
    occ = torch.zeros(N)

    # Assign B-factors
    occ[center] = 1.0
    occ[first_neighbors] = 1.0
    occ[second_neighbors] = 0.5

    # Rewrite PDB lines with new B-factors
    new_lines = []
    atom_idx = 0

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # Format B-factor into columns 61–66 (PDB standard)
            b = f"{occ[atom_idx].item():6.2f}"
            newline = line[:54] + b + line[60:]
            new_lines.append(newline)
            atom_idx += 1
        else:
            new_lines.append(line)

    write_pdb_lines(new_lines, out_path)

    print(f"Saved modified PDB to {out_path}")
    print(f"Center atom: {center}")
    print(f"1st neighbors: {first_neighbors}")
    print(f"2nd neighbors: {second_neighbors}")

#modify_b_factors()
