"""Exact enumeration for small lattices, reusing the project's neighbor and energy code."""
import itertools, math
import torch
from alchemist.neighbors import compute_neighbor_masks, get_neighbor_indices, assemble_neighbor_features
from alchemist.analysis import interaction_energy
from alchemist.pdbs import read_pdb_coords

def write_ring_pdb(path, n=4, spacing=1.0):
    """n atoms on a circle with NN distance `spacing` (n=4: a unit square, so
    the diagonals are NN2 and a cut between 1 and sqrt(2) selects the ring)."""
    R = spacing / (2 * math.sin(math.pi / n))
    with open(path, "w") as f:
        for i in range(n):
            a = 2 * math.pi * i / n + math.pi / n
            f.write(f"ATOM  {i+1:5d} {'C':<3s} XTL     1    {R*math.cos(a):8.3f}{R*math.sin(a):8.3f}{0:8.3f}  1.00  1.00\n")

def neighbor_lists(pdb_path, box, cuts):
    coords = read_pdb_coords(pdb_path)
    masks = compute_neighbor_masks(coords, torch.tensor(box, dtype=torch.float32), cuts)
    return get_neighbor_indices(masks)

def state_index(species):
    """species: (..., N) ints in {0 (A), 1 (B)} -> integer sum_i species_i 2^i."""
    w = 2 ** torch.arange(species.shape[-1], device=species.device)
    return (species * w).sum(-1)

def exact_table(pdb_path, box, cuts, MU, E1, E2=None, beta=1.0):
    """Returns (P, U): Boltzmann probabilities and energies of all 2^N species
    assignments, indexed by state_index.  Energies come from interaction_energy
    on one-hot identities, i.e. the same function the loss uses."""
    nbrs = neighbor_lists(pdb_path, box, cuts)
    N = nbrs[0].shape[0]
    states = torch.tensor(list(itertools.product([0, 1], repeat=N)))[:, torch.arange(N - 1, -1, -1)]
    prob = torch.nn.functional.one_hot(states, 2).float()                    # (2^N, N, 2)
    a = assemble_neighbor_features(prob, nbrs)
    f = lambda t: None if t is None else torch.as_tensor(t, dtype=torch.float32)
    U = interaction_energy(a, f(MU), f(E1), f(E2), tuple(n.shape[1] for n in nbrs)).sum(1)
    order = state_index(states)
    Ut = torch.empty_like(U); Ut[order] = U
    P = torch.softmax(-beta * Ut, 0)
    return P, Ut
