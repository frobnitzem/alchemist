import torch
from torch import nn
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
