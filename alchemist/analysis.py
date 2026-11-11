import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from alchemist.neighbors import compute_neighbor_masks, get_neighbor_indices, assemble_neighbor_features
from alchemist.pdbs import read_pdb_coords

# Notation: (r, p) is the extended phase-space point.  r holds per-atom
# features (the first C entries are element logits), p holds the conjugate
# "momenta".  Probabilities are never called p; they are `prob`, computed
# from r by identity_prob().

def identity_prob(r: torch.Tensor, mode: str = "argmax") -> torch.Tensor:
    """Per-atom element probability/identity from logits r (..., C).

    mode="softmax": prob = softmax(r).
    mode="argmax":  prob = one_hot(argmax(r)) in the forward pass, with the
        gradient of softmax(r) in the backward pass (straight-through).  The
        argmax identity is piecewise constant in r, so without the
        straight-through term the energy would give no training signal.
    """
    prob = torch.softmax(r, dim=-1)
    if mode == "softmax":
        return prob
    if mode == "argmax":
        hard = F.one_hot(r.argmax(dim=-1), r.shape[-1]).to(prob.dtype)
        return hard + (prob - prob.detach())
    raise ValueError(f"unknown identity mode {mode!r}")

def prob_A(r, identity="argmax"):
    """Probability that an atom is species A (index 0).  r: (B, N, C) -> (B, N)."""
    return identity_prob(r, identity)[..., 0]

def _setup_neighbors(pdb_path, box, cuts):
    coords = read_pdb_coords(pdb_path)
    box = torch.tensor(box, dtype=torch.float32)
    return get_neighbor_indices(compute_neighbor_masks(coords, box, cuts))

class CompositionAnalysis:
    r""" Conditional probability that the neighbor in each slot is A given
        that the central atom is A: \sum prob_A[0]*prob_A[nbr] / \sum prob_A[0].
        Slot 0 holds the overall probability of A.
    """
    def __init__(self, pdb_path, box, cuts, identity="argmax"):
        self.neighborlists = _setup_neighbors(pdb_path, box, cuts)
        self.identity = identity

    def reduce(self, res1, res2):
        return {
            'composition': res1['composition'] + res2['composition'],
            'atoms': res1['atoms'] + res2['atoms'],
        }

    def finalize(self, res, total_batches):
        atoms = res['atoms']
        comp = res['composition'].clone()
        comp[1:,:] /= comp[0,None,:]  # divide by \sum prob_A
        comp[0,:]  /= atoms           # divide by \sum 1
        return {'composition': comp, 'atoms': atoms/total_batches}

    def __call__(self, x_tuple, batch):
        x = x_tuple[0]
        device = x['r'].device
        pA = prob_A(x['r'], self.identity).unsqueeze(-1)       # (B, N, 1)
        nbrs = [n.to(device) for n in self.neighborlists]
        assembled = assemble_neighbor_features(pA, nbrs)       # (B, N, 17, 1)
        assembled[:,:,1:,:] *= assembled[:,:,0,None,:].clone()
        return {'composition': assembled.sum(dim=(0, 1)),
                'atoms': assembled.shape[0] * assembled.shape[1]}

class InteractHist:
    def __init__(self, pdb_path, box, cuts, n_bins=100, identity="argmax"):
        self.neighborlists = _setup_neighbors(pdb_path, box, cuts)
        self.identity = identity
        self.n_bins = n_bins
        self.bins = [torch.linspace(0, 4, n_bins + 1),
                     torch.linspace(0, 12, n_bins + 1)]

    def __call__(self, x_tuple, batch):
        x = x_tuple[0]
        device = x['r'].device
        pA = prob_A(x['r'], self.identity)                     # (B, N)
        nbrs = [n.to(device) for n in self.neighborlists]
        assembled = assemble_neighbor_features(pA.unsqueeze(-1), nbrs)
        pA_c = assembled[:, :, 0, 0].unsqueeze(-1)             # (B, N, 1)
        shells = [assembled[:, :, 1:5, 0], assembled[:, :, 5:17, 0]]

        batch_hist = torch.zeros((2, 4, self.n_bins), device=device)
        for s, pA_n in enumerate(shells):
            counts = [(pA_c * pA_n).sum(-1),                   # AA
                      (pA_c * (1 - pA_n)).sum(-1),             # AB
                      ((1 - pA_c) * pA_n).sum(-1),             # BA
                      ((1 - pA_c) * (1 - pA_n)).sum(-1)]       # BB
            for t, vals in enumerate(counts):
                batch_hist[s, t] = torch.histogram(
                    vals.flatten().cpu(), bins=self.bins[s])[0]
        return {'prob_A': pA.sum(),
                'atoms': torch.tensor([pA.numel()], device=device),
                'hist_NN': batch_hist}

    def reduce(self, res1, res2):
        return {k: res1[k] + res2[k] for k in ('prob_A', 'atoms', 'hist_NN')}

    def finalize(self, res, total_batches):
        return {'prob_A': res['prob_A'] / res['atoms'],
                'hist_NN': res['hist_NN'],
                'bin1': self.bins[0], 'bin2': self.bins[1]}

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
    
    # 2. Extract prob_a (B, N, 17, 2) - assuming D=2 for composition
    # We only care about the first two components if D > 2
    prob = probs[..., :2]
    
    # 3. Extract the central atom's probability (B, N, 2)
    # The first slot (index 0) is the self-feature
    prob_self = prob[..., 0, :] # (B, N, 2)
    
    # 4. Extract neighbor probabilities
    prob_nn1 = prob[..., 1:5, :]  # (B, N, 4, 2)
    prob_nn2 = prob[..., 5:17, :] # (B, N, 12, 2)
    
    def get_pair_hist(prob_center, prob_neighbors):
        # prob_center: (B, N, 2)
        # prob_neighbors: (B, N, M, 2)
        B, N, M, _ = prob_neighbors.shape
        
        # Expand prob_center to match neighbors: (B, N, M, 2)
        prob_center_exp = prob_center.unsqueeze(2).expand(-1, -1, M, -1)
        
        # Flatten to (B*N*M, 2)
        centers = prob_center_exp.reshape(-1, 2)
        neighbors = prob_neighbors.reshape(-1, 2)
        
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

    nn1_hist = get_pair_hist(prob_self, prob_nn1)
    nn2_hist = get_pair_hist(prob_self, prob_nn2)
    
    return nn1_hist, nn2_hist

def interaction_energy(
    prob: torch.Tensor,
    mu: torch.Tensor,
    E1: torch.Tensor,
    E2: Optional[torch.Tensor] = None,
    sizes: Tuple[int, ...] = (4, 12),
) -> torch.Tensor:
    """
    Per-atom U(r) = -mu.prob_i + sum_shells sum_{j in shell(i)} prob_i^T E_shell prob_j
    (grand-canonical: E - mu.N, so mu enters with a minus sign).  Each bond is
    seen from both ends, so a bond between i and j contributes twice.

    Args:
        prob: (B, N, 1+sum(sizes), C) element probabilities/identities of each
              atom and its neighbors (slot 0 self, then the shells in order),
              from identity_prob().
        mu: (C,) chemical potentials.
        E1, E2: (C, C) interaction matrices of shells 1 and 2 (E2 may be None
              when there is only one shell).
        sizes: number of neighbors in each shell (default NN1=4, NN2=12).
    Returns:
        (B, N) tensor.
    """
    prob_a = prob[..., 0, :]
    U = -torch.einsum('bnc, c -> bn', prob_a, mu)
    start = 1
    for E, m in zip((E1, E2), sizes):
        U = U + torch.sum(torch.matmul(prob_a, E).unsqueeze(2) * prob[..., start:start + m, :], dim=(-2, -1))
        start += m
    return U

def compute_energy_parameterized(
    assembled_features: torch.Tensor,
    mu: torch.Tensor,
    E1: torch.Tensor,
    E2: torch.Tensor,
    sigma: Optional[torch.Tensor] = 1,
    do_boundary: Optional[bool] = True
) -> torch.Tensor:
    """Legacy wrapper (softmax identities, optional r^2/2sigma^2 bound) used by
    the pre-AIfront scripts.  New code uses identity_prob + interaction_energy
    and the H0 term in losses.KLLoss."""
    U = interaction_energy(torch.softmax(assembled_features, dim=-1), mu, E1, E2)
    if do_boundary:
        r = assembled_features[..., 0, :]
        U = U + (r ** 2).sum(dim=-1) / (2 * sigma ** 2)
    return U


class SiteHist:
    """Per-site neighbor statistics split by the central atom's type.

    For every atom i (type A if prob_A[i] is largest; with soft identities the
    split is weighted by prob_A[i] / 1-prob_A[i]):
      prob1 = mean over NN1 of prob_A[neighbor]      (NN1 A-fraction)
      prob2 = mean over NN2 of prob_A[neighbor]      (NN2 A-fraction)
      e1    = NN1 energy contribution  prob_i^T E1 prob_j summed over NN1
      e2    = NN2 energy contribution
    Accumulates weighted 2D histograms (prob1 vs prob2, prob1 vs e1) and
    weighted sums of prob1, prob2, e1, e2, indexed by center type [A, B].
    """
    def __init__(self, pdb_path, box, cuts, E1, E2, n_bins=21, identity="argmax"):
        self.neighborlists = _setup_neighbors(pdb_path, box, cuts)
        self.E1 = torch.as_tensor(E1, dtype=torch.float32)
        self.E2 = torch.as_tensor(E2, dtype=torch.float32)
        self.identity = identity
        self.n_bins = n_bins
        self.pbins = torch.linspace(-1 / (2 * (n_bins - 1)), 1 + 1 / (2 * (n_bins - 1)), n_bins + 1)
        e_lo = float(min(self.E1.min(), 0)) * 4 - 0.01
        e_hi = float(max(self.E1.max(), 0)) * 4 + 0.01
        self.ebins = torch.linspace(e_lo, e_hi, n_bins + 1)

    def __call__(self, x_tuple, batch):
        x = x_tuple[0]
        dev = x['r'].device
        prob = identity_prob(x['r'], self.identity).detach()[..., :2]   # (B,N,2)
        nbrs = [n.to(dev) for n in self.neighborlists]
        a = assemble_neighbor_features(prob, nbrs)                      # (B,N,17,2)
        pc = a[:, :, 0, :]                                              # (B,N,2) center
        p1 = a[:, :, 1:5, 0].mean(-1)
        p2 = a[:, :, 5:17, 0].mean(-1)
        e1 = (torch.matmul(pc, self.E1.to(dev)).unsqueeze(2) * a[:, :, 1:5, :]).sum((-2, -1))
        e2 = (torch.matmul(pc, self.E2.to(dev)).unsqueeze(2) * a[:, :, 5:17, :]).sum((-2, -1))
        H12 = torch.zeros(2, self.n_bins, self.n_bins)
        H1e = torch.zeros(2, self.n_bins, self.n_bins)
        S = torch.zeros(2, 5)  # weight, sum p1, p2, e1, e2
        for t in range(2):
            w = pc[..., t].flatten().cpu()
            v1, v2, ve1, ve2 = (q.flatten().cpu() for q in (p1, p2, e1, e2))
            H12[t] = torch.histogramdd(torch.stack([v1, v2], -1), bins=[self.pbins, self.pbins], weight=w)[0]
            H1e[t] = torch.histogramdd(torch.stack([v1, ve1], -1), bins=[self.pbins, self.ebins], weight=w)[0]
            S[t] = torch.stack([w.sum(), (w * v1).sum(), (w * v2).sum(), (w * ve1).sum(), (w * ve2).sum()])
        return {'H12': H12, 'H1e': H1e, 'S': S}

    def reduce(self, r1, r2):
        return {k: r1[k] + r2[k] for k in r1}

    def finalize(self, res, total_batches):
        S = res['S']
        mean = S[:, 1:] / S[:, :1].clamp_min(1e-9)   # (2,4): rows A,B; cols p1,p2,e1,e2
        return {'H_prob1_prob2': res['H12'], 'H_prob1_e1': res['H1e'],
                'site_weight': S[:, 0].contiguous(), 'site_mean': mean.contiguous(),
                'pbins': self.pbins, 'ebins': self.ebins}


class StateDist:
    """Empirical distribution over the 2^N species assignments (argmax
    identities, species 0 = A) compared with the exact Boltzmann table."""
    def __init__(self, pdb_path, box, cuts, MU, E1, E2=None, KT=1.0):
        from alchemist.ising import exact_table
        self.P, _ = exact_table(pdb_path, box, cuts, MU, E1, E2, beta=1.0 / KT)

    def __call__(self, x_tuple, batch):
        from alchemist.ising import state_index
        sp = x_tuple[0]['r'].argmax(-1)
        return {'counts': torch.bincount(state_index(sp), minlength=len(self.P)).float()}

    def reduce(self, r1, r2):
        return {'counts': r1['counts'] + r2['counts']}

    def finalize(self, res, total_batches):
        q = res['counts'] / res['counts'].sum()
        P = self.P.to(q.device)
        kl = (P * (P.clamp_min(1e-30).log() - q.clamp_min(1e-12).log())).sum()
        return {'q': q, 'P': P.clone(), 'KL_pi_q': kl.reshape(1),
                'TV': (0.5 * (q - P).abs().sum()).reshape(1)}
