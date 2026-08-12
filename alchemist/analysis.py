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
    E2: torch.Tensor,
    sigma: Optional[torch.Tensor] = 1,
    do_boundary: Optional[bool] = True
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
    r = assembled_features[..., 0, :]  # (B, N, D)
    r2 = (r ** 2).sum(dim=-1)  # (B, N)
    
    # p_a: (B, N, D)
    p_a = p[..., 0, :]
    
    # Neighbors
    p1 = p[..., 1:5, :] # (B, N, 4, D)
    p2 = p[..., 5:17, :] # (B, N, 12, D)
    
    # Energy = dot(p_a, mu) + sum(p_a^T @ E1 @ p_neighbor) + sum(p_a^T @ E2 @ p_neighbor)
    
    term_bound =  r2 / (2 * sigma ** 2)  # (B, N)

    # term_mu: (B, N)
    # p_a is (B, N, D), mu is (D,)
    # Use einsum to ensure we contract over D regardless of B=1
    term_mu = torch.einsum('bnd, d -> bn', p_a, mu)
    
    # term_e1: (B, N, 4)
    # p_a @ E1 -> (B, N, D)
    p_a_E1 = torch.matmul(p_a, E1) 
    # sum over D: (B, N, D) * (B, N, 4, D) -> (B, N, 4)
    term_e1 = torch.sum(p_a_E1.unsqueeze(2) * p1, dim=-1)
    
    # term_e2: (B, N, 12)
    p_a_E2 = torch.matmul(p_a, E2)
    term_e2 = torch.sum(p_a_E2.unsqueeze(2) * p2, dim=-1)
    
    if do_boundary:
        return term_bound + term_mu + term_e1.sum(dim=-1) + term_e2.sum(dim=-1)
    else:
        return term_mu + term_e1.sum(dim=-1) + term_e2.sum(dim=-1)

from typing import Optional, Union
TensorLike = Union[float, torch.Tensor]

def compute_energy_LennardJones(
    particle_locations: torch.Tensor,
    epsilon_LJ: TensorLike = 1.0,
    sigma_LJ: TensorLike = 1.0,
    box: Optional[TensorLike] = None,
    cutoff: Optional[TensorLike] = None,
    p: float = 12.0,
    q: float = 6.0,
    spatial_dim: int = 3,
) -> torch.Tensor:
    """
    Computes generalized Lennard-Jones energy per particle.

    Args:
        particle_locations:
            (B, N, D), (N, D), (B, N*D), or (N*D,)
        epsilon_LJ:
            Energy scale; scalar or shape (B,)
        sigma_LJ:
            Length scale; scalar or shape (B,)
        box:
            Rectangular periodic box; scalar, (D,), or (B, D)
        cutoff:
            Optional hard cutoff; scalar or shape (B,)
        p, q:
            LJ exponents. Standard LJ uses p=12, q=6.

    Returns:
        Energy per particle with shape (B, N).
        Summing over N gives the physical total potential energy.
    """
    if not p > q > 0:
        raise ValueError("Require p > q > 0.")

    x = particle_locations
    if not x.is_floating_point():
        x = x.float()

    # Convert all supported inputs to (B, N, D).
    if x.ndim == 1:
        x = x.reshape(1, -1, spatial_dim)
    elif x.ndim == 2:
        x = (
            x.unsqueeze(0)
            if x.shape[-1] == spatial_dim
            else x.reshape(x.shape[0], -1, spatial_dim)
        )

    if x.ndim != 3 or x.shape[-1] != spatial_dim:
        raise ValueError(
            "Expected (N,D), (B,N,D), (N*D,), or (B,N*D)."
        )

    B, N, D = x.shape

    def batch_parameter(value: TensorLike) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=x.dtype, device=x.device)

        if value.numel() not in (1, B):
            raise ValueError(
                "Parameters must be scalar or contain one value per batch."
            )

        if value.numel() == B and B > 1:
            return value.reshape(B, 1, 1)

        return value.squeeze()

    epsilon = batch_parameter(epsilon_LJ)
    sigma = batch_parameter(sigma_LJ)

    if torch.any(epsilon <= 0) or torch.any(sigma <= 0):
        raise ValueError("epsilon_LJ and sigma_LJ must be positive.")

    # dr[b, i, j] = r_i - r_j
    delta = x[:, :, None, :] - x[:, None, :, :]
    if box is not None:
        box = torch.as_tensor(box, dtype=x.dtype, device=x.device)
        if box.ndim == 0:
            box = box.repeat(D)
        if box.ndim == 1:
            box = box.reshape(1, 1, 1, D)
        elif box.ndim == 2:
            box = box.reshape(B, 1, 1, D)

        delta = torch.remainder(delta + box / 2, box) - box / 2
    
    r2 = delta.square().sum(dim=-1)

    # Remove self-interactions.
    mask = ~torch.eye(N,dtype=torch.bool,device=x.device).unsqueeze(0)

    if cutoff is not None:
        cutoff = batch_parameter(cutoff)

        if torch.any(cutoff <= 0):
            raise ValueError("cutoff must be positive.")

        mask = mask & (r2 < cutoff.square())

    # Give excluded pairs a safe nonzero distance.
    r2 = torch.where(mask, r2, torch.ones_like(r2))

    sigma_over_r_sq = sigma.square() / r2

    c_pq = p / (
        (p - q)
        * (q / p) ** (q / (p - q))
    )

    # Factorized form avoids inf - inf when particles overlap.
    pair_energy = (
        c_pq
        * epsilon
        * sigma_over_r_sq.pow(q / 2)
        * (
            sigma_over_r_sq.pow((p - q) / 2)
            - 1.0
        )
    )

    pair_energy = torch.where(
        mask,
        pair_energy,
        torch.zeros_like(pair_energy),
    )

    # Split each pair energy equally between its two particles.
    return 0.5 * pair_energy.sum(dim=-1)