import torch, glob, json
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

import matplotlib.pyplot as plt
from alchemist.flows import GlowBlock, MultiStep, Q, RealNVP
from alchemist.pdbs import read_pdb_coords, write_pdb_trajectory
from alchemist.neighbors import compute_neighbor_masks, get_neighbor_indices, assemble_neighbor_features
from alchemist.ml_utils import train_and_summarize
from alchemist.analysis import compute_neighbor_histograms, compute_energy_parameterized


if __name__ == "__main__":
    plt.figure(figsize=(6,4))
    #plot energies as a function of Ga composition
    ga_compositions = [1, 0, 0.5, 0.5]
    #strictly ga, as, and perfect alternating ga-as-ga-as
    PDB_PATH = Path('examples/GaAs/GaAs.pdb')
    BOX = torch.full((3,), 5.75 * 3)
    CUTS = [2.5, 4.5]
    N=216
    DIM = 2
    MU = torch.zeros(DIM, dtype=torch.float32)
    E1 = torch.tensor([[-0.3,-0.5],[-0.5,-0.1]], dtype=torch.float32)
    E2 = torch.tensor([[-0.05,-0],[-0,-0.05]], dtype=torch.float32)
    coords = read_pdb_coords(PDB_PATH)
    masks = compute_neighbor_masks(coords, BOX, CUTS)
    neighborlists = get_neighbor_indices(masks)

    ga_energies = []
    for config in range(4):
        if config == 0:
            r = torch.ones(1, N, 1)
        elif config == 1:
            r = torch.zeros(1, N, 1)
        elif config == 2:
            idx = (torch.arange(1, 55).repeat_interleave(4)) % 2
            base = torch.nn.functional.one_hot(idx, num_classes=2).float() #(N, 2)
            r = base.unsqueeze(0)[:,:,0:1]
        else:
            r = torch.rand(1, N, 1)
        r = torch.cat([r, 1 - r], dim=-1) # (1, N, 2)
        r = r * 10
        assembled = assemble_neighbor_features(r, neighborlists)
        ga_energies.append(compute_energy_parameterized(assembled, MU, E1, E2, do_boundary=False).sum().item())
        
    for j in ['Glow','LeapFrog','RealNVP']:
        file = f'Figures/{j}_interactions_vs_temperature.json'
        data = json.load(open(file, 'r'))
        interactions = data['interactions']

        
        energies = []
        for i in range(len(interactions)):
            energy = interactions[i][0] * E1[0,0] * 864 + interactions[i][1] * E1[0,1] * 864 + interactions[i][2] * E1[1,1] * 864 + \
                     interactions[i][3] * E2[0,0] * 2592 + interactions[i][4] * E2[0,1] * 2592 + interactions[i][5] * E2[1,1] * 2592
            energies.append(energy)
        
        comps = []
        for i in range(len(interactions)):
            N_A = interactions[i][0] *864 / 2 + interactions[i][1] * 864 / 4
            N_B = interactions[i][2] * 864 / 2 + interactions[i][1] * 864 / 4
            comp = N_A / (N_A + N_B)
            comps.append(comp)

        plt.plot(comps, energies, label = f'{j}')

    plt.scatter(ga_compositions[0], ga_energies[0], marker='o',label='All Ga', color='r')
    plt.scatter(ga_compositions[1], ga_energies[1], marker='s',label='All As', color='r')
    plt.scatter(ga_compositions[2], ga_energies[2], marker='^',label='Perfect Alternating', color='r')
    plt.scatter(ga_compositions[3], ga_energies[3], marker='*',label='Random Mixture', color='r')

    fontsize = 15
    plt.xlabel('Ga Composition', fontsize=fontsize)
    plt.ylabel('Potential Energy', fontsize=fontsize)
    # plt.title('Phase Diagram', fontsize=fontsize)
    plt.tight_layout()
    plt.legend()
    plt.savefig('Figures/phase_diagram_energy.png')