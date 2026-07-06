import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from alchemistlib.flows import GlowBlock, MultiStep, Q, fix_kT
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

    E1 = (pA_ga * (f1_ga * 0.1 + f1_as * -0.1) +
          pA_as * (f1_ga * -0.1 + f1_as * 0.1))

    # E2 = (pA_ga * (f2_ga * 0.15 + f2_as * 0.0) +
    #       pA_as * (f2_ga * 0.0 + f2_as * 0.05))

    return E1.sum(dim=-1) #+ E2.sum(dim=-1)

def percentA(r):
    # Calculate the percentage of particles that are of type A
    dr = r[:, :, 1] - r[:, :, 0]
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


_GAAS_SCALE = 5.75


def _read_pdb_coords(pdb_path='../examples/GaAs/GaAs.pdb'):
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


def neighbor_masks(periodic=True, pdb_path='../examples/GaAs/GaAs.pdb'):
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

def build_U(batch_size, Na, dim, periodic=True):
    nbr1, nbr2 = neighbor_masks(periodic=periodic)

    nbr1_tensor = torch.tensor(nbr1, dtype=torch.long).unsqueeze(0)
    nbr2_tensor = torch.tensor(nbr2, dtype=torch.long).unsqueeze(0)

    def U(r, t):
        """
        r: (B, Na, dim)
        t: scalar or (B,)
        """

        dr = percentA(r).unsqueeze(-1)          # (B, N, 1)
        dr_expanded = dr.expand(-1, -1, dr.shape[1])  # (B, N, N)

        # First-shell neighbor values: (B, N, 4)
        nbr1_vals = torch.gather(dr_expanded, dim=1, index=nbr1_tensor).squeeze(-1)

        # Second-shell neighbor values: (B, N, 12)
        nbr2_vals = torch.gather(dr_expanded, dim=1, index=nbr2_tensor).squeeze(-1)

        return atom_energy_mixed_batched(dr, nbr1_vals, nbr2_vals).sum(1)

    return U, [nbr1, nbr2]

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
    n_steps_flow=10,
    train_iters=25,
    lr=1e-3,
):
    U, neighborlists = build_U(batch_size, Na, dim, periodic=periodic)
    glow = GlowBlock(en=U, dt=0.001, neighborlists=neighborlists)
    flow = MultiStep(glow, n_steps_flow)

    normal = torch.distributions.normal.Normal(0, 1)
    optimizer = optim.Adam(glow.parameters(), lr=lr)

    for it in range(train_iters):
        x = {
            'r': Q(normal.sample((batch_size, Na, dim))) * sigma,
            't': 0.0
        }
        percent = percentA(x['r'])
        x['p'] = torch.stack([percent, 1 - percent], dim=2)
        x0 =  x.copy()
        logJ = 0.0

        for _ in range(2):
            x, lJ, info = flow(x)
            logJ = logJ + lJ
        loss = 1 / kT * (U(x['r'], x['t']) - U(x0['r'], 0.0)) - logJ
        # print(U(x['r'], x['t']), U(x0['r'], 0.0), logJ, loss)

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        if (it + 1) % 10 == 0:
            print(f"[train] iter {it+1:4d}  loss = {loss.mean().item():.4f}")

    return glow


# ----------------------------------------------------------------------
# Testing GlowBlock: trajectory + PDB + histogram
# ----------------------------------------------------------------------
def test_glowblock_two_part(glow, batch_size=1, Na=216, dim=2, sigma=1.0, kT=1.0, periodic=True):
    U, neighborlists = build_U(batch_size, Na, dim, periodic=periodic)
    flow = MultiStep(glow, 10)

    normal = torch.distributions.normal.Normal(0, 1)
    x = {
        'r': Q(normal.sample((batch_size, Na, dim))) * sigma,
        'p': Q(fix_kT(normal.sample((batch_size, Na, dim)), kT)),
        't': 0.0
    }
    percent = percentA(x['r'])
    x['p'] = torch.stack([percent, 1 - percent], dim=2)
    x0 = x.copy()

    logJ = 0.0
    Ga_percents = [percentA(x0['r'])]

    for i in range(10):
        x, lJ, info = flow(x)
        logJ = logJ + lJ
        r = x['r']
        Ga_percents.append(percentA(r))
        if i % 50 == 0:
            print(f"[test] Step {i}: r[0,0] = {r[0,0]}, logJ = {logJ}")

    loss = 1 / kT * (U(r, x['t']) - U(x0['r'], 0.0)) - logJ
    print("[test] final loss:", loss)

    return torch.stack(Ga_percents)


# ----------------------------------------------------------------------
# Main: train, test, write PDB, plot histogram
# ----------------------------------------------------------------------
if __name__ == "__main__":
    coords = _read_pdb_coords()
    # print(coords)

    glow = train_glowblock_two_part(periodic=True)
    Ga_percents = test_glowblock_two_part(glow, periodic=True)

    As_percents = 1 - Ga_percents
    _OUTPUT_PDB = Path(__file__).with_suffix('.pdb')
    _write_pdb_trajectory(_OUTPUT_PDB, coords, Ga_percents[:, 0], As_percents[:, 0])

    # plt.hist(Ga_percents[-1, 0].flatten().cpu().detach().numpy(), bins=5)
    # plt.xlabel("Percent Ga")
    # plt.ylabel("count")
    # plt.title("Histogram of Percent Ga values (GlowBlock)")
    # plt.tight_layout()
    # plt.savefig("percentGa_histogram_glowblock.png", dpi=150)
    # plt.close()
