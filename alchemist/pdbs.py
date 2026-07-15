import torch
from pathlib import Path
from typing import List, Tuple

def read_pdb_coords(pdb_path: Path) -> torch.Tensor:
    """Reads atom coordinates from a PDB file."""
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

def write_pdb_trajectory(pdb_path: Path, coords: torch.Tensor, ga_percents: torch.Tensor, as_percents: torch.Tensor):
    """Writes a multi-model PDB trajectory with coordinates and percentages in B-factor/Occupancy."""
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
                    f"ATOM  {atom_index:5d}  X   XTL     1"
                    f"    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                    f"{ga_value:6.2f}{as_value:6.2f}\n"
                )
            pdb_file.write('ENDMDL\n')
        pdb_file.write('END\n')
