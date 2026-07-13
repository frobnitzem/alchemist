import torch

def _format_pdb_atom(serial, name, x, y, z, occupancy, bfactor):
    return (
        f"ATOM  {serial:5d} {name:<3s} XTL     1"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}\n"
    )

def _write_pdb_trajectory(pdb_path, coords, ga_percents, as_percents):
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

_GAAS_SCALE = 5.75

def _read_pdb_coords(pdb_path="../examples/GaAs/GaAs.pdb"):
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

def _cluster_shells(distances, tolerance=1e-2):
    shells = []

    for value in torch.sort(distances[distances > 0]).values:
        if not shells or torch.abs(value - shells[-1][-1]) > tolerance:
            shells.append([value])
        else:
            shells[-1].append(value)

    return [torch.stack(shell).mean() for shell in shells]

def neighbor_masks(periodic=True, pdb_path="../examples/GaAs/GaAs.pdb"):
    # Build the two neighbor shells directly from the GaAs PDB coordinates.
    # periodic=True gives 4 first-shell and 12 second-shell neighbors per site.
    crds = _read_pdb_coords(pdb_path)
    delta = crds[:, None, :] - crds[None, :, :]

    if periodic:
        num_atoms = crds.shape[0]
        num_cells = round((num_atoms / 8) ** (1 / 3))

        if num_cells**3 * 8 != num_atoms:
            raise ValueError(f"Unexpected GaAs atom count: {num_atoms}")

        box = torch.full(
            (3,),
            float(num_cells) * _GAAS_SCALE,
            dtype=crds.dtype,
            device=crds.device,
        )
        delta = delta - torch.round(delta / box) * box

    dist = torch.linalg.norm(delta, dim=-1)
    dist.fill_diagonal_(float("inf"))

    shells = _cluster_shells(dist[torch.isfinite(dist)])

    if len(shells) < 2:
        raise ValueError("Not enough distance shells")

    cut1 = 0.5 * (shells[0] + shells[1])

    if len(shells) < 3:
        cut2 = shells[1] + 0.5 * (shells[1] - shells[0])
    else:
        cut2 = 0.5 * (shells[1] + shells[2])

    mask1 = dist < cut1
    mask2 = (dist >= cut1) & (dist < cut2)

    nbr1 = [mask1[i].nonzero(as_tuple=True)[0].tolist() for i in range(dist.shape[0])]
    nbr2 = [mask2[i].nonzero(as_tuple=True)[0].tolist() for i in range(dist.shape[0])]

    return nbr1, nbr2


def clone_state(x):
    """
    Safe clone for dictionary states.
    """
    return {
        k: v.clone() if torch.is_tensor(v) else v
        for k, v in x.items()
    }

def read_compositions_by_frame(filename, frame_keyword="MODEL"):
    frames = []
    current_frame = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Detect start of a new frame
            if line.startswith(frame_keyword):
                if current_frame:
                    frames.append(torch.tensor(current_frame))
                    current_frame = []
                continue

            # Skip empty or non-numeric lines
            parts = line.split()
            if len(parts) < 2:
                continue

            # Try to parse last two columns as floats
            try:
                compA = float(parts[-2])
                compB = float(parts[-1])
                current_frame.append([compA, compB])
            except ValueError:
                # Not a numeric line
                continue

    # Append last frame if not empty
    if current_frame:
        frames.append(torch.tensor(current_frame))

    return frames