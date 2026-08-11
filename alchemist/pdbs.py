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
    rs: torch.Tensor,
) -> None:
    """Writes a multi-model PDB trajectory.
    
    Args:
        pdb_path: Output file path
        rs: Atom coordinates, shape (n_frames, n_atoms, DIM+3)
    """
    n_frames, n_atoms, DIM_plus_3 = rs.shape
    if DIM_plus_3 < 3:
        raise ValueError(f"Expected last dimension of rs to be at least 3, got {DIM_plus_3}.")
    XYZs = rs[:,:,-3:]
    chemical_identities = rs[:,:,:-3]
    chemical_percents = torch.softmax(chemical_identities, dim=-1)

    with open(pdb_path, "w") as pdb_file:
        pdb_file.write("REMARK multi-model trajectory with fixed coordinates\n")
        for frame_index in range(n_frames):
            if (frame_index+1) % 200 == 0:
                print(f"Writing frame {frame_index+1}/{n_frames}")
            pdb_file.write(f"MODEL {frame_index:4d}\n")
            for atom_index in range(n_atoms):
                coord = XYZs[frame_index, atom_index]
                p0_value = chemical_percents[frame_index, atom_index, 0]
                pdb_file.write(
                    _format_pdb_atom(
                        atom_index,
                        "X",
                        float(coord[0]),
                        float(coord[1]),
                        float(coord[2]),
                        float(p0_value),
                        float(1.0 - p0_value),
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