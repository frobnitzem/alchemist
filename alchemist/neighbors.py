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
    B, N, D = features.shape
    
    # 1. Self features: (B, N, 1, D)
    # We want to pick the i-th atom for the i-th slot.
    # features: (B, N, D) -> unsqueeze(2) -> (B, N, 1, D)
    # The self-feature for atom i is just features[:, i, :].
    # We can simply use the features tensor as the self-shell.
    self_feat = features.unsqueeze(2) # (B, N, 1, D)
    
    all_shells = [self_feat]
    
    # 2. Neighbor shells
    for nbr in neighborlists:
        # nbr: (N, M)
        # Expand nbr to (B, N, M, D)
        idx = nbr.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, D)
        # Expand features to (B, N, M, D) for gathering along dim=1 (the atom axis)
        feat_exp = features.unsqueeze(2).expand(-1, -1, nbr.shape[1], -1)
        shell_feat = torch.gather(feat_exp, dim=1, index=idx)
        all_shells.append(shell_feat)
        
    # Concatenate along the neighbor dimension: (B, N, 1 + M1 + M2..., D)
    return torch.cat(all_shells, dim=2)

def sum_neighbor_features(assembled_features: torch.Tensor) -> torch.Tensor:
    """
    Sums assembled neighbor features over the atom dimension.
    
    Args:
        assembled_features: (B, N, M_total, D) tensor.
        
    Returns:
        (B, M_total, D) tensor.
    """
    return assembled_features.sum(dim=1)
