import torch, glob
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

import matplotlib.pyplot as plt
from alchemist.flows import GlowBlock, MultiStep, Q, MultiIndependent, RealNVP
from alchemist.pdbs import read_pdb_coords, write_pdb_trajectory
from alchemist.neighbors import compute_neighbor_masks, get_neighbor_indices, assemble_neighbor_features
from alchemist.ml_utils import train_and_summarize
from alchemist.analysis import compute_neighbor_histograms, compute_energy_parameterized


if __name__ == "__main__":
    plt.figure(figsize=(5, 4))
    #plot energies as a function of Ga composition
    ga_compositions = [1, 0, 0.5, 0.5]
    #strictly ga, as, and perfect alternating ga-as-ga-as
    PDB_PATH = Path('examples/GaAs/GaAs.pdb')
    BOX = torch.full((3,), 5.75 * 3)
    CUTS = [2.5, 4.5]
    N=216
    DIM = 2
    MU = torch.zeros(DIM, dtype=torch.float32)
    E1 = torch.tensor([[-0.1,-0.5],[-0.5,-0.1]], dtype=torch.float32)
    E2 = torch.tensor([[-0.05,-0],[-0, -0.05]], dtype=torch.float32)
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
        ga_energies.append(compute_energy_parameterized(assembled, MU, E1, E2).sum().item())
        
    comps = []
    energies = []
    for i in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
        glowblock_file = f'outputs/only_train_kT{i}/test_results.pt'
        glowblock_data = torch.load(glowblock_file)
        last_frame_ga_compositions = glowblock_data['last_Ga_composition']
        last_frame_energies = glowblock_data['overall_energies']
        last_frame_avg_comp = last_frame_ga_compositions.mean().item()
        last_frame_avg_energy = last_frame_energies.mean().item()

        comps.append(last_frame_avg_comp)
        energies.append(last_frame_avg_energy)
        kT_num = float(glowblock_file.split('/')[1].split('_')[-1][2:])
        plt.annotate(f'{kT_num-1}', (last_frame_avg_comp, last_frame_avg_energy), xytext=(5, 5), textcoords='offset points')

    plt.plot(comps, energies, color='g', label = 'GlowBlock 3-layer')

    plt.scatter(ga_compositions[0], ga_energies[0], marker='o',label='All Ga', color='r')
    plt.scatter(ga_compositions[1], ga_energies[1], marker='s',label='All As', color='r')
    plt.scatter(ga_compositions[2], ga_energies[2], marker='^',label='Perfect Alternating', color='r')
    plt.scatter(ga_compositions[3], ga_energies[3], marker='*',label='Random Mixture', color='r')

    fontsize = 15
    plt.xlabel('Ga Composition', fontsize=fontsize)
    plt.ylabel('Potential Energy', fontsize=fontsize)
    plt.title('Phase Diagram', fontsize=fontsize)
    plt.tight_layout()
    plt.legend()
    plt.savefig('phase_diagram_energy.png')