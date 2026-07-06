import torch, glob
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

import matplotlib.pyplot as plt

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

    # E1 = (pA_ga * (f1_ga * 0.3 + f1_as * 0.5) +
    #       pA_as * (f1_ga * 0.5 + f1_as * 0.1))
    E1 = (pA_ga * (f1_ga * 0.5 + f1_as * 0.1) + #Try to make it favor Ga-As interactions more than As-Ga interactions
           pA_as * (f1_ga * 0.1 + f1_as * 0.3))

    # E2 = (pA_ga * (f2_ga * 0.15 + f2_as * 0.0) +
    #       pA_as * (f2_ga * 0.0 + f2_as * 0.05))

    return (E1.sum(dim=-1) ).sum(dim=-1) #+ E2.sum(dim=-1)

if __name__ == "__main__":
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

    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for count, glowblock_file in enumerate(glob.glob('glowblockenergyalt/3layer*_test_results.pt')):
        glowblock_data = torch.load(glowblock_file)
        last_frame_ga_compositions = glowblock_data['last_Ga_composition']
        last_frame_energies = glowblock_data['overall_energies']
        plt.scatter(last_frame_ga_compositions, last_frame_energies,alpha=0.01, color=colors[count % len(colors)])

    plt.scatter(ga_compositions, ga_energies, marker='o',label='Energy extremes')

    plt.xlabel('Ga Composition')
    plt.ylabel('Potential Energy')
    plt.title('Phase Diagram')
    plt.legend()
    plt.savefig('phase_diagrams/phase_diagram_energyalt.png')