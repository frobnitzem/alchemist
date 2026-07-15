from pathlib import Path
import pytest
import torch

from alchemist.flows import LeapFrog, MultiStep, Q, fix_kT
from alchemist.modules import FNN

def atom_energy_mixed_batched(pA_ga, f1_ga, f2_ga, params = [0.3, 0.5, 0.1, 0.15, 0, 0.05]):
    """
    pA_ga : (B, N) tensor of Ga composition for each atom A
    f1_ga : (B, N, k1) tensor of Ga fractions among NN1
    f2_ga : (B, N, k2) tensor of Ga fractions among NN2
    params: list of 6 parameters for energy contributions:
        params[0]: E_AA_1 (energy contribution for A-A in shell 1)
        params[1]: E_AB_1 (energy contribution for A-B in shell 1)
        params[2]: E_BB_1 (energy contribution for B-B in shell 1)
        params[3]: E_AA_2 (energy contribution for A-A in shell 2)
        params[4]: E_AB_2 (energy contribution for A-B in shell 2)
        params[5]: E_BB_2 (energy contribution for B-B in shell 2)

    Returns:
        (B, N) tensor of energies for each atom A in each batch
    """

    # Fractions of As
    pA_as = 1 - pA_ga
    f1_as = 1 - f1_ga
    f2_as = 1 - f2_ga

    # Expand pA_ga to match neighbor dims: (B, N, 1)
    pA_ga_exp = pA_ga.unsqueeze(-1)
    pA_as_exp = pA_as.unsqueeze(-1)
    print("shapes: ", pA_ga_exp.shape, f1_ga.shape, f2_ga.shape)

    E1 = (pA_ga_exp * (f1_ga * params[0] + f1_as * params[1]) +
          pA_as_exp * (f1_ga * params[1] + f1_as * params[2]))

    E2 = (pA_ga_exp * (f2_ga * params[3] + f2_as * params[4]) +
          pA_as_exp * (f2_ga * params[4] + f2_as * params[5]))

    return E1.sum(dim=-1) + E2.sum(dim=-1)

#example 1 batch, 1 atoms, 2 neighbors in each shell
print(atom_energy_mixed_batched(torch.tensor([[0.5]]), torch.tensor([[[0.5, 0.5]]]), torch.tensor([[[1.0, 1.0]]])))

#example 2 batch, 2 atoms, 2 neighbors in shell 1 and 4 neighbors in shell 2
print(atom_energy_mixed_batched(torch.tensor([[0.5, 0.5], [0.25, 0.75]]), torch.tensor([[[0.5, 0.5], [0.25, 0.75]], [[0.5, 0.5], [0.25, 0.75]]]), torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])))
