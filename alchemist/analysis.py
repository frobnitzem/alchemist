import torch
from typing import Tuple, Optional

def compute_neighbor_histograms(
    assembled_features: torch.Tensor, 
    bins: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates 2D histograms of neighbor compositions.
    
    Args:
        assembled_features: (B, N, 17, D) tensor.
        bins: Number of bins for the 2D histogram.
        
    Returns:
        Tuple of (nn1_hist, nn2_hist), each of shape (bins, bins).
    """
    # 1. Softmax to get probabilities (B, N, 17, D)
    probs = torch.softmax(assembled_features, dim=-1)
    
    # 2. Extract p_a (B, N, 17, 2) - assuming D=2 for composition
    # We only care about the first two components if D > 2
    p = probs[..., :2]
    
    # 3. Extract the central atom's probability (B, N, 2)
    # The first slot (index 0) is the self-feature
    p_self = p[..., 0, :] # (B, N, 2)
    
    # 4. Extract neighbor probabilities
    p_nn1 = p[..., 1:5, :]  # (B, N, 4, 2)
    p_nn2 = p[..., 5:17, :] # (B, N, 12, 2)
    
    def get_pair_hist(p_center, p_neighbors):
        # p_center: (B, N, 2)
        # p_neighbors: (B, N, M, 2)
        B, N, M, _ = p_neighbors.shape
        
        # Expand p_center to match neighbors: (B, N, M, 2)
        p_center_exp = p_center.unsqueeze(2).expand(-1, -1, M, -1)
        
        # Flatten to (B*N*M, 2)
        centers = p_center_exp.reshape(-1, 2)
        neighbors = p_neighbors.reshape(-1, 2)
        
        # Use torch.histogram2d if available, or manual binning
        # Manual binning for better control and compatibility
        # Scale [0, 1] to [0, bins-1]
        c_bins = (centers * bins).long().clamp(0, bins - 1)
        n_bins = (neighbors * bins).long().clamp(0, bins - 1)
        
        # Convert 2D indices to 1D flat indices
        flat_indices = c_bins[:, 0] * bins + n_bins[:, 0]
        
        hist = torch.zeros(bins * bins, device=centers.device)
        hist.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
        
        return hist.reshape(bins, bins)

    nn1_hist = get_pair_hist(p_self, p_nn1)
    nn2_hist = get_pair_hist(p_self, p_nn2)
    
    return nn1_hist, nn2_hist

def compute_energy_parameterized(
    assembled_features: torch.Tensor,
    mu: torch.Tensor,
    E1: torch.Tensor,
    E2: torch.Tensor
) -> torch.Tensor:
    """
    Computes energy per atom using parameterized tensors.
    
    Args:
        assembled_features: (B, N, 17, D)
        mu: (D,) - chemical potential
        E1: (D, D) - NN1 interaction matrix
        E2: (D, D) - NN2 interaction matrix
        
    Returns:
        (B, N) tensor of energies.
    """
    # Softmax to get probabilities (B, N, 17, D)
    p = torch.softmax(assembled_features, dim=-1)
    
    # p_a: (B, N, D)
    p_a = p[..., 0, :]
    
    # p1_ab: (B, N, 4, D, D) - outer product of center and neighbor
    p1 = p[..., 1:5, :] # (B, N, 4, D)
    p1_ab = torch.einsum('bnd, bnm -> bndm', p_a, p1)
    
    # p2_ab: (B, N, 12, D, D)
    p2 = p[..., 5:17, :] # (B, N, 12, D)
    p2_ab = torch.einsum('bnd, bnm -> bndm', p_a, p2)
    
    # Energy = dot(p_a, mu) + sum(dot(p1_ab, E1)) + sum(dot(p2_ab, E2))
    # dot(p_a, mu) -> (B, N)
    term_mu = torch.matmul(p_a, mu)
    
    # dot(p1_ab, E1) -> (B, N, 4)
    # p1_ab is (B, N, 4, D, D), E1 is (D, D)
    # We want the Frobenius inner product for each neighbor: sum_{i,j} p1_ab[i,j] * E1[i,j]
    term_e1 = torch.einsum('bndm, dm -> bn', p1_ab, E1) # This is wrong, E1 is (D,D)
    # Correct: sum over D,D for each neighbor m
    term_e1 = torch.einsum('bndm, dm -> bn', p1_ab, E1) # Still wrong.
    
    # Let's use a simpler contraction:
    # For each neighbor m: sum_{i,j} p_a[i] * p_neighbor[j] * E1[i,j]
    # This is p_a^T @ E1 @ p_neighbor
    # p_a: (B, N, D), E1: (D, D), p1: (B, N, 4, D)
    
    # (B, N, D) @ (D, D) -> (B, N, D)
    p_a_E1 = torch.matmul(p_a, E1) 
    # (B, N, D) * (B, N, 4, D) -> sum over D -> (B, N, 4)
    term_e1 = torch.sum(p_a_E1.unsqueeze(2) * p1, dim=-1)
    
    # Same for E2
    p_a_E2 = torch.matmul(p_a, E2)
    term_e2 = torch.sum(p_a_E2.unsqueeze(2) * p2, dim=-1)
    
    return term_mu + term_e1.sum(dim=-1) + term_e2.sum(dim=-1)
