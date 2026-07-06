from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
from pathlib import Path
import pytest
import torch

from alchemistlib.flows import LeapFrog, MultiStep, Q, fix_kT
from alchemistlib.modules import FNN

def atom_energy_mixed_batched(pA_ga, f1_ga, f2_ga):
    """
    pA_ga : (B, N) tensor of Ga composition for each atom A
    f1_ga : (B, N, k1) tensor of Ga fractions among NN1
    f2_ga : (B, N, k2) tensor of Ga fractions among NN2

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

    E1 = (pA_ga_exp * (f1_ga * 0.3 + f1_as * 0.5) +
          pA_as_exp * (f1_ga * 0.5 + f1_as * 0.1))

    E2 = (pA_ga_exp * (f2_ga * 0.15 + f2_as * 0.0) +
          pA_as_exp * (f2_ga * 0.0 + f2_as * 0.05))

    return E1.sum(dim=-1) + E2.sum(dim=-1)

#example 1 batch, 1 atoms, 2 neighbors in each shell
print(atom_energy_mixed_batched(torch.tensor([[0.5]]), torch.tensor([[[0.5, 0.5]]]), torch.tensor([[[1.0, 1.0]]])))

#example 2 batch, 2 atoms, 2 neighbors in shell 1 and 4 neighbors in shell 2
print(atom_energy_mixed_batched(torch.tensor([[0.5, 0.5], [0.25, 0.75]]), torch.tensor([[[0.5, 0.5], [0.25, 0.75]], [[0.5, 0.5], [0.25, 0.75]]]), torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])))
