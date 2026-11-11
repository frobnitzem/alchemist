import torch
from typing import List, Tuple

def compute_neighbor_masks(coords: torch.Tensor, box: torch.Tensor, cuts: List[float]) -> List[torch.Tensor]:
    """
    Computes periodic neighbor masks for a given set of distance cutoffs.
    
    Args:
        coords: (N, 3) tensor of atom coordinates.
        box: (3,) tensor of box dimensions.
        cuts: List of distance cutoffs for each shell.
        
    Returns:
        List of boolean masks, each of shape (N, N).
    """
    # Compute all-to-all distances with periodic boundary conditions
    delta = coords[:, None, :] - coords[None, :, :]
    delta = delta - torch.round(delta / box) * box
    dist = torch.linalg.norm(delta, dim=-1)
    
    # Ignore self-distance for mask calculation
    dist.fill_diagonal_(float('inf'))
    
    masks = []
    prev_cut = 0.0
    for cut in cuts:
        # Each shell is defined as the region between the previous cut and current cut
        mask = (dist >= prev_cut) & (dist < cut)
        masks.append(mask)
        prev_cut = cut
        
    return masks

def get_neighbor_indices(masks: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Converts boolean masks to index tensors.
    
    Args:
        masks: List of boolean masks (N, N).
        
    Returns:
        List of index tensors (N, M_i).
    """
    indices = []
    for mask in masks:
        # For each atom, find indices of True values in the mask
        # This assumes each atom has the same number of neighbors in the shell
        idx = torch.stack([mask[i].nonzero(as_tuple=True)[0] for i in range(mask.shape[0])])
        indices.append(idx)
    return indices

def assemble_neighbor_features(features: torch.Tensor, neighborlists: List[torch.Tensor]) -> torch.Tensor:
    """
    Assembles features for an atom and its neighbors into a single tensor.
    
    Args:
        features: (B, N, D) tensor of atom features.
        neighborlists: List of index tensors [nbr1, nbr2, ...].
        
    Returns:
        (B, N, 1 + sum(M_i), D) tensor.
    """
    # Self slot, then one advanced-indexing gather per shell:
    # features[:, nbr] is (B, N, M, D) with no expanded intermediate.
    shells = [features.unsqueeze(2)]
    shells.extend(features[:, nbr] for nbr in neighborlists)
    return torch.cat(shells, dim=2)

def sum_neighbor_features(assembled_features: torch.Tensor) -> torch.Tensor:
    """
    Sums assembled neighbor features over the atom dimension.
    
    Args:
        assembled_features: (B, N, M_total, D) tensor.
        
    Returns:
        (B, M_total, D) tensor.
    """
    return assembled_features.sum(dim=1)
