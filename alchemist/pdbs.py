import torch
from pathlib import Path
from typing import List, Tuple

def _format_pdb_atom(serial, name, x, y, z, occupancy, bfactor):
    return (
        f"ATOM  {serial:5d} {name:<3s} XTL     1"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}\n"
    )
def write_pdb_trajectory(
    pdb_path: Path,
    coords: torch.Tensor,
    ga_percents: torch.Tensor,
    as_percents: torch.Tensor,
) -> None:
    """Writes a multi-model PDB trajectory.
    
    Args:
        pdb_path: Output file path
        coords: Atom coordinates, shape (n_atoms, 2) or (n_atoms, 3)
        ga_percents: Values for occupancy column, shape (n_frames, n_atoms)
        as_percents: Values for B-factor column, shape (n_frames, n_atoms)
    """
    coords = coords.detach().cpu()

    if coords.ndim != 2:
        raise ValueError(f"Expected 2D coords, got {tuple(coords.shape)}")

    if coords.shape[-1] == 2:
        coords = torch.cat(
            [coords, torch.zeros(coords.shape[0], 1, dtype=coords.dtype)],
            dim=-1,
        )
    elif coords.shape[-1] != 3:
        raise ValueError(f"Expected 2D or 3D coords, got {tuple(coords.shape)}")

    ga_percents = ga_percents.detach().cpu()
    as_percents = as_percents.detach().cpu()

    with open(pdb_path, "w") as pdb_file:
        pdb_file.write("REMARK multi-model trajectory with fixed coordinates\n")
        for frame_index, (ga_frame, as_frame) in enumerate(
            zip(ga_percents, as_percents), start=1
        ):
            pdb_file.write(f"MODEL {frame_index:4d}\n")
            for atom_index, (coord, ga_value, as_value) in enumerate(
                zip(coords, ga_frame, as_frame), start=1
            ):
                pdb_file.write(
                    _format_pdb_atom(
                        atom_index,
                        "X",
                        float(coord[0]),
                        float(coord[1]),
                        float(coord[2]),
                        float(ga_value),
                        float(as_value),
                    )
                )
            pdb_file.write("ENDMDL\n")
        pdb_file.write("END\n")

def read_pdb_coords(pdb_path: Path) -> torch.Tensor:
    """Reads atom coordinates from a PDB file."""
    coords = []

    with open(pdb_path, "r") as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                coords.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )

    if not coords:
        raise ValueError(f"No atom coordinates found in {pdb_path}")

    return torch.tensor(coords, dtype=torch.float64)