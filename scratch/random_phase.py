import torch, glob
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

import matplotlib.pyplot as plt
from utils import  neighbor_masks

from energyfunc import atom_energy_mixed_batched

if __name__ == "__main__":
    #plot energies as a function of Ga composition
    ga_compositions = [1, 0, 0.5, 0.5]
    #strictly ga, as, and perfect alternating ga-as-ga-as
    N=216

    ga_energies = [
        atom_energy_mixed_batched(torch.ones(N, 1), torch.ones(N, 4), torch.ones(N, 12)).sum(), # all Ga
        atom_energy_mixed_batched(torch.zeros(N, 1), torch.zeros(N, 4), torch.zeros(N, 12)).sum(), # all As
        atom_energy_mixed_batched(torch.ones(N // 2, 1), torch.zeros(N // 2, 4), torch.ones(N // 2, 12)).sum() + atom_energy_mixed_batched(torch.zeros(N // 2, 1), torch.ones(N // 2, 4), torch.zeros(N // 2, 12)).sum(), # perfect alternating
        atom_energy_mixed_batched(torch.ones(N, 1)/2, torch.ones(N, 4)/2, torch.ones(N, 12)/2).sum() # random mixture
    ]

    color = 'r'
    nbr1, nbr2 = neighbor_masks(periodic=True)
    nbr1 = torch.tensor(nbr1, dtype=torch.long)
    nbr2 = torch.tensor(nbr2, dtype=torch.long)
    for j in range(10):
        if j % 1 == 0:
            print(f"Random sample {j}/10")
        for init_ga in [0.0, 1.0,'special']:
            if init_ga == 'special':
                idx = (torch.arange(1, 55).repeat_interleave(4)) % 2
                ga_content_double = torch.nn.functional.one_hot(idx, num_classes=2).float() #(N, 2)
                ga_content = ga_content_double[:, 0].unsqueeze(-1) #(N, 1)
            else:
                ga_content = torch.ones(N,1)*init_ga
            for i in range(1000):
                rand_idx = torch.randint(0, N, (1,))
                ga_content[rand_idx] = 1-ga_content[rand_idx]

                dr_expanded1 = ga_content.expand(-1, nbr1.shape[1])  # (N, M1)
                dr_expanded2 = ga_content.expand(-1, nbr2.shape[1])  # (N, M2)

                # First-shell neighbor values: (N, 4)
                nbr1_vals = torch.gather(dr_expanded1, dim=0, index=nbr1)

                # Second-shell neighbor values: (N, 12)
                nbr2_vals = torch.gather(dr_expanded2, dim=0, index=nbr2)

                energy = atom_energy_mixed_batched(ga_content, nbr1_vals, nbr2_vals).sum()
                plt.scatter(ga_content.mean().item(), energy.item(), alpha=0.01, color=color,edgecolors='none')


    plt.scatter(ga_compositions, ga_energies, marker='o',label='Energy extremes')

    plt.xlabel('Ga Composition')
    plt.ylabel('Potential Energy')
    plt.title('Phase Diagram')
    plt.legend()
    plt.savefig('phase_diagram_random_sample_long.png')