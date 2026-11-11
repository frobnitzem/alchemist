import torch
from torch import nn
from alchemist.analysis import identity_prob, interaction_energy
from alchemist.neighbors import compute_neighbor_masks, get_neighbor_indices, assemble_neighbor_features
from alchemist.pdbs import read_pdb_coords

class NeighborExpansion(nn.Module):
    def __init__(self, pdb_path, box, cuts):
        super().__init__()
        coords = read_pdb_coords(pdb_path)
        box = torch.tensor(box, dtype=torch.float32)
        masks = compute_neighbor_masks(coords, box, cuts)
        neighborlists = get_neighbor_indices(masks)
        
        # Register each list in the neighborlists as a buffer
        for i, nbr in enumerate(neighborlists):
            self.register_buffer(f'nbr_{i}', nbr)
        self.num_shells = len(neighborlists)

    def __call__(self, features):
        # Reconstruct the list of tensors from buffers
        neighborlists = [getattr(self, f'nbr_{i}') for i in range(self.num_shells)]
        res = assemble_neighbor_features(features, neighborlists)
        # res: (B, N, M, D) -> (B, N, M*D)
        B, N, M, D = res.shape
        return res.view(B, N, M * D)


class NeighborEnergy(nn.Module):
    """Potential V(r, t) = |r|^2/(2 sigma_r^2) + U(identity(r)) for use as the
    energy of flows.LeapFrog (force = -dV/dr).  It has no parameters: the
    Verlet flow integrates the physical (H0 + U) dynamics.

    identity="softmax" gives a smooth potential; "argmax" evaluates U on the
    hard identity and the force uses the straight-through (softmax) gradient.
    """
    def __init__(self, pdb_path, box, cuts, MU, E1, E2=None, sigma_r=1.0, identity="softmax"):
        super().__init__()
        coords = read_pdb_coords(pdb_path)
        masks = compute_neighbor_masks(coords, torch.tensor(box, dtype=torch.float32), cuts)
        nbrs = get_neighbor_indices(masks)
        for i, n in enumerate(nbrs):
            self.register_buffer(f'nbr_{i}', n)
        self.num_shells = len(nbrs)
        self.sizes = tuple(n.shape[1] for n in nbrs)
        self.register_buffer('MU', torch.as_tensor(MU, dtype=torch.float32))
        self.register_buffer('E1', torch.as_tensor(E1, dtype=torch.float32))
        self.register_buffer('E2', torch.zeros(2, 2) if E2 is None else torch.as_tensor(E2, dtype=torch.float32))
        self.sigma_r = sigma_r
        self.identity = identity

    def forward(self, r, t=None):
        nbrs = [getattr(self, f'nbr_{i}') for i in range(self.num_shells)]
        prob = assemble_neighbor_features(identity_prob(r, self.identity), nbrs)
        U = interaction_energy(prob, self.MU, self.E1, self.E2, self.sizes).sum(1)
        return (r ** 2).sum(dim=(-2, -1)) / (2 * self.sigma_r ** 2) + U
