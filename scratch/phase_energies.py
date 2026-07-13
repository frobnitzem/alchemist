import torch, glob
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

import matplotlib.pyplot as plt
from utils import read_compositions_by_frame, neighbor_masks
from summary_glowblock import _compute_interactions, _build_neighbor_pairs

def atom_energy_mixed_batched(pA_ga, f1_ga, f2_ga):
    """
    pA_ga : (N, 1) tensor of Ga composition for each atom A
    f1_ga : (N, 4) tensor of Ga fractions among NN1
    f2_ga : (N, 12) tensor of Ga fractions among NN2

    Returns:
        () energy of the system
    """

    # Fractions of As
    pA_as = 1 - pA_ga
    f1_as = 1 - f1_ga
    f2_as = 1 - f2_ga

    E1 = (pA_ga * (f1_ga * 0.3 + f1_as * 0.5) +
          pA_as * (f1_ga * 0.5 + f1_as * 0.1))
    # E1 = (pA_ga * (f1_ga * 0.3 + f1_as * 0.1) + #Try to make it favor Ga-As interactions more than As-Ga interactions
    #        pA_as * (f1_ga * 0.1 + f1_as * 0.5))

    E2 = (pA_ga * (f2_ga * 0.15 + f2_as * 0.0) +
          pA_as * (f2_ga * 0.0 + f2_as * 0.05))

    return (E1.sum(dim=-1)).sum(-1)  + (E2.sum(dim=-1)).sum(dim=-1)

if __name__ == "__main__":
    plt.figure(figsize=(5, 4))
    #plot energies as a function of Ga composition
    ga_compositions = [1, 0, 0.5, 0.5]
    #strictly ga, as, and perfect alternating ga-as-ga-as
    N=216

    ga_energies = [
        atom_energy_mixed_batched(torch.ones(N, 1), torch.ones(N, 4), torch.ones(N, 12)), # all Ga
        atom_energy_mixed_batched(torch.zeros(N, 1), torch.zeros(N, 4), torch.zeros(N, 12)), # all As
        atom_energy_mixed_batched(torch.ones(N // 2, 1), torch.zeros(N // 2, 4), torch.ones(N // 2, 12))+atom_energy_mixed_batched(torch.zeros(N // 2, 1), torch.ones(N // 2, 4), torch.zeros(N // 2, 12)), # perfect alternating
        atom_energy_mixed_batched(torch.ones(N, 1)/2, torch.ones(N, 4)/2, torch.ones(N, 12)/2) # random mixture
    ]

    colors = ['g', 'b', 'c', 'm', 'y']
    kT = [1, 0.5, 0.1, 0.05, 0.01]
    comps = []
    energies = []
    for i in kT:
        for count, glowblock_file in enumerate(glob.glob(f'glowblockenergyposter/3layer*kT{i}*_test_results.pt')):
    # for count, glowblock_file in enumerate(glob.glob('glowblockenergyaltposter/3layer*_test_results.pt')):
            glowblock_data = torch.load(glowblock_file)
            last_frame_ga_compositions = glowblock_data['last_Ga_composition']
            last_frame_energies = glowblock_data['overall_energies']
            last_frame_avg_comp = last_frame_ga_compositions.mean().item()
            last_frame_avg_energy = last_frame_energies.mean().item()

            comps.append(last_frame_avg_comp)
            energies.append(last_frame_avg_energy)

    plt.plot(comps, energies, color='g', label = 'GlowBlock 3-layer')

    folder = 'Original_flow_system/leapfrogresults'
    neighborlists = neighbor_masks(periodic=True)
    first_pairs = _build_neighbor_pairs(neighborlists[0])
    second_pairs = _build_neighbor_pairs(neighborlists[1])
    for pdbfile in glob.glob(f'{folder}/multistep*.pdb'):
        frames = read_compositions_by_frame(pdbfile)
        num_frames = len(frames)
        overall_interactions = torch.empty((num_frames, 6), dtype=torch.float32)
        overall_energies = torch.empty((num_frames,), dtype=torch.float32)
        Ga_comps = torch.empty((num_frames,), dtype=torch.float32)

        for i, frame in enumerate(frames):
            Ga_comps[i] = frame[:,0].mean().item()
            interactions = _compute_interactions(frame[:,0], first_pairs, second_pairs)
            overall_interactions[i] = interactions
            # energy = interactions[0] * 0.3 + interactions[1] * 0.1 + interactions[2] * 0.5
            energy = interactions[0] * 0.3 + interactions[1] * 0.5 + interactions[2] * 0.1 + \
                 interactions[3] * 0.15 + interactions[4] * 0.0 + interactions[5] * 0.05

            overall_energies[i] = energy

    plt.plot(Ga_comps, overall_energies, color='b', label = 'Normalizing Flow')

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
    plt.savefig('phase_diagrams/phase_diagram_energy.png')