from pathlib import Path
import pytest
import torch
import random

from alchemist.pdbs import read_pdb_coords
from alchemist.neighbors import compute_neighbor_masks, get_neighbor_indices

_GAAS_PDB = Path(__file__).resolve().parents[1] / 'examples' / 'GaAs' / 'GaAs.pdb'
_GAAS_BOX = torch.full((3,), 5.75 * 3)

def test_neighbor_masks_periodic():
    coords = read_pdb_coords(_GAAS_PDB)
    # GaAs shells are roughly 2.45 and 4.2
    cuts = [2.5, 4.5]
    masks = compute_neighbor_masks(coords, _GAAS_BOX, cuts)
    
    assert len(masks) == 2
    # Each atom in GaAs has 4 NN1 and 12 NN2
    assert torch.all(masks[0].sum(dim=-1) == 4)
    assert torch.all(masks[1].sum(dim=-1) == 12)

def test_neighbor_masks_non_periodic():
    coords = read_pdb_coords(_GAAS_PDB)
    # Use a huge box to effectively disable periodicity
    huge_box = torch.full((3,), 1e6)
    cuts = [2.5, 4.5]
    masks = compute_neighbor_masks(coords, huge_box, cuts)
    
    # Non-periodic counts will vary by atom (boundary effects)
    # Just check that they are not all 4/12
    assert not torch.all(masks[0].sum(dim=-1) == 4)

def test_get_neighbor_indices():
    coords = read_pdb_coords(_GAAS_PDB)
    cuts = [2.5, 4.5]
    masks = compute_neighbor_masks(coords, _GAAS_BOX, cuts)
    indices = get_neighbor_indices(masks)
    
    assert len(indices) == 2
    assert indices[0].shape == (coords.shape[0], 4)
    assert indices[1].shape == (coords.shape[0], 12)

def test_modify_b_factors(tmp_path):
    out_path = tmp_path / "modified_GaAs.pdb"
    coords = read_pdb_coords(_GAAS_PDB)
    cuts = [2.5, 4.5]
    masks = compute_neighbor_masks(coords, _GAAS_BOX, cuts)
    indices = get_neighbor_indices(masks)
    
    N = coords.shape[0]
    center = random.randrange(N)
    
    first_neighbors = indices[0][center].tolist()
    second_neighbors = indices[1][center].tolist()
    
    with open(_GAAS_PDB, 'r') as f:
        lines = f.readlines()
    
    occ = torch.zeros(N)
    occ[center] = 1.0
    occ[first_neighbors] = 1.0
    occ[second_neighbors] = 0.5
    
    new_lines = []
    atom_idx = 0
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            b = f"{occ[atom_idx].item():6.2f}"
            newline = line[:54] + b + line[60:]
            new_lines.append(newline)
            atom_idx += 1
        else:
            new_lines.append(line)
            
    with open(out_path, 'w') as f:
        f.writelines(new_lines)
    
    assert out_path.exists()
