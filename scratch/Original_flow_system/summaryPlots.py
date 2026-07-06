from pathlib import Path
import torch, glob
import matplotlib.pyplot as plt

_GAAS_SCALE = 5.75
from utils import _format_pdb_atom, _write_pdb_trajectory, neighbor_masks

def _read_pdb_trajectory(pdb_path):
    coords = []
    comps_Ga = []
    curr_coords = []
    curr_comps_Ga = []
    with open(pdb_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('MODEL'):
                if curr_coords:
                    coords.append(torch.tensor(curr_coords, dtype=torch.float64))
                    comps_Ga.append(torch.tensor(curr_comps_Ga, dtype=torch.float64))
                    curr_coords = []
                    curr_comps_Ga = []
            if line.startswith('ATOM') or line.startswith('HETATM'):
                curr_coords.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
                curr_comps_Ga.append(float(line[54:60]))

    if not coords:
        raise ValueError(f'No atom coordinates found in {pdb_path}')

    return torch.stack(coords), torch.stack(comps_Ga)

def _read_pdb_coords(pdb_path):
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


def neighbor_masks(pdb_path = '../examples/GaAs/GaAs.pdb',periodic=True):
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

for file in glob.glob('*.pdb'):
    coords, comps_Ga = _read_pdb_trajectory(file)
    print(f"File: {file}, Coordinates shape: {coords.shape}, Ga content: {comps_Ga}")

    #Calculate 1st neighbor shell and 2nd neighbor shell
    nbr1, nbr2 = neighbor_masks(periodic=True)

    overall_interactions = []
    energys = []
    for frame_idx in range(coords.shape[0]):
        print(f"Frame {frame_idx}:")
        
        #Calculate neighbor interactions for each atom
        interactions = [0,0,0,0,0,0] #First neighbor Ga-Ga, Ga-As, As-As, Second neighbor Ga-Ga, Ga-As, As-As
        for atom_idx in range(coords.shape[1]):
            atom_coord = coords[frame_idx, atom_idx]
            ga_content = comps_Ga[frame_idx, atom_idx]
            
            # First neighbor interactions
            for nbr_idx in nbr1[atom_idx]:
                nbr_coord = comps_Ga[frame_idx, nbr_idx]
                interactions[0] += (ga_content * nbr_coord).sum() # Ga-Ga
                interactions[1] += (ga_content * (1-nbr_coord)+(1-ga_content) * nbr_coord).sum() # Ga-As
                interactions[2] += ((1-ga_content) * (1-nbr_coord)).sum() # As-As

            # Second neighbor interactions
            for nbr_idx in nbr2[atom_idx]:
                nbr_coord = comps_Ga[frame_idx, nbr_idx]
                interactions[3] += (ga_content * nbr_coord).sum() # Ga-Ga
                interactions[4] += (ga_content * (1-nbr_coord)+(1-ga_content) * nbr_coord).sum() # Ga-As
                interactions[5] += ((1-ga_content) * (1-nbr_coord)).sum() # As-As
            
        print(f"Interactions: {interactions}")
        overall_interactions.append(interactions)
        if 'static' in file:
            energy = (interactions[0] * 0.3 + interactions[1] * 0.5 + interactions[2] * 0.1 +
                      interactions[3] * 0.15 + interactions[4] * 0.0 + interactions[5] * 0.05)
        else:
            energy = (interactions[0] * 0.1 + interactions[1] * -0.1 + interactions[2] * 0.1 +
                    interactions[3] * 0.0 + interactions[4] * 0.0 + interactions[5] * 0.0)
        energys.append(energy)
    
    #plot energy vs frame
    plt.plot(energys)
    plt.xlabel('Frame Index')
    plt.ylabel('Potential Energy')
    plt.title(f'Energy vs Frame for {file}')
    plt.savefig(f'{file}_energy_plot.png')
    plt.close()

    #plot interactions vs frame
    overall_interactions = torch.tensor(overall_interactions)
    plt.plot(overall_interactions, label = ['1st Ga-Ga', '1st Ga-As', '1st As-As', '2nd Ga-Ga', '2nd Ga-As', '2nd As-As'])
    plt.xlabel('Frame Index')
    plt.ylabel('Interaction Energy')
    plt.title(f'Interactions vs Frame for {file}')
    plt.legend(title='Interaction Type')
    plt.savefig(f'{file}_interactions_plot.png')
    plt.close()