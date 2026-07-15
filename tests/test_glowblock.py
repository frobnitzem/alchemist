import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from alchemist.flows import GlowBlock, MultiStep, Q, fix_kT
from alchemist.pdbs import read_pdb_coords
from alchemist.neighbors import compute_neighbor_masks, get_neighbor_indices, assemble_neighbor_features, sum_neighbor_features
from alchemist.ml_utils import train_and_summarize

_GAAS_PDB = Path(__file__).resolve().parents[1] / 'examples' / 'GaAs' / 'GaAs.pdb'
_GAAS_BOX = torch.full((3,), 5.75 * 3)

def atom_energy_mixed_batched(pA_ga, f1_ga, f2_ga):
    """
    pA_ga : (B, N, 1) tensor of Ga composition for each atom A
    f1_ga : (B, N, 4) tensor of Ga fractions among NN1
    f2_ga : (B, N, 12) tensor of Ga fractions among NN2
    Returns: (B, N) tensor of energies for each atom A in each batch
    """
    pA_as = 1 - pA_ga
    f1_as = 1 - f1_ga
    f2_as = 1 - f2_ga

    E1 = (pA_ga * (f1_ga * 0.3 + f1_as * 0.5) +
          pA_as * (f1_ga * 0.5 + f1_as * 0.1))

    E2 = (pA_ga * (f2_ga * 0.15 + f2_as * 0.0) +
          pA_as * (f2_ga * 0.0 + f2_as * 0.05))

    return E1.sum(dim=-1) + E2.sum(dim=-1)

def percentA(r):
    dr = r[:, :, 1] - r[:, :, 0]
    return torch.sigmoid(dr)

def build_U(batch_size, Na, dim, periodic=True):
    coords = read_pdb_coords(_GAAS_PDB)
    cuts = [2.5, 4.5]
    box = _GAAS_BOX if periodic else torch.full((3,), 1e6)
    
    masks = compute_neighbor_masks(coords, box, cuts)
    neighborlists = get_neighbor_indices(masks)
    
    def U(r, t):
        # r: (B, Na, dim)
        # t: scalar or (B,)
        dr = percentA(r).unsqueeze(-1) # (B, N, 1)
        
        # Use assemble_neighbor_features to get (B, N, 17, 1)
        # 1 self, 4 NN1, 12 NN2
        assembled = assemble_neighbor_features(dr, neighborlists) # (B, N, 17, 1)
        
        # Split into buckets
        # self: [:, :, 0:1, :], nn1: [:, :, 1:5, :], nn2: [:, :, 5:17, :]
        pA_ga = assembled[:, :, 0:1, :].squeeze(-1) # (B, N, 1)
        f1_ga = assembled[:, :, 1:5, :].squeeze(-1) # (B, N, 4)
        f2_ga = assembled[:, :, 5:17, :].squeeze(-1) # (B, N, 12)
        
        return atom_energy_mixed_batched(pA_ga, f1_ga, f2_ga).sum(1)

    return U, neighborlists

def train_glowblock_two_part(
    batch_size=1,
    Na=216,
    dim=2,
    sigma=1.0,
    kT=1.0,
    periodic=True,
    n_steps_flow=10,
    train_iters=25,
    lr=1e-3,
):
    U, neighborlists = build_U(batch_size, Na, dim, periodic=periodic)
    glow = GlowBlock(neighborlists=neighborlists, dim=dim, dt=0.001)
    flow = MultiStep(glow, n_steps_flow)
    
    normal = torch.distributions.normal.Normal(0, 1)
    
    def data_gen():
        while True:
            x = {
                'r': Q(normal.sample((batch_size, Na, dim))) * sigma,
                't': 0.0
            }
            percent = percentA(x['r'])
            x['p'] = torch.stack([percent, 1 - percent], dim=2)
            yield x

    def loss_fn(batch):
        x0 = batch.copy()
        x = batch.copy()
        logJ = 0.0
        # The original loop did 2 steps for training
        for _ in range(2):
            x, lJ, info = flow(x)
            logJ = logJ + lJ
        
        loss = 1 / kT * (U(x['r'], x['t']) - U(x0['r'], 0.0)) - logJ
        return loss.mean()

    def feature_extractor(model, batch):
        # model here is the flow (MultiStep)
        x = batch.copy()
        for _ in range(n_steps_flow):
            x, _, _ = model(x)
        
        # Extract features for sum_neighbor_features
        # The target is the composition (B, N, 1)
        dr = percentA(x['r']).unsqueeze(-1)
        assembled = assemble_neighbor_features(dr, neighborlists)
        return sum_neighbor_features(assembled)

    optimizer = optim.Adam(glow.parameters(), lr=lr)
    
    # We map train_iters to epochs/batches. 
    # Original was 25 iterations. Let's do 24 training iterations and 1 eval epoch.
    # 1 epoch of 24 batches, then 1 epoch of 1 batch for summary.
    glow, mean, var = train_and_summarize(
        model=flow,
        loss_fn=loss_fn,
        data_generator=data_gen,
        optimizer=optimizer,
        epochs=2,
        batches_per_epoch=train_iters - 1,
        feature_extractor=feature_extractor
    )
    
    return glow, mean, var

def test_glowblock_two_part():
    batch_size=1
    Na=216
    dim=2
    sigma=1.0
    kT=1.0
    periodic=True
    
    # Use the new training utility via the helper
    glow, mean, var = train_glowblock_two_part(
        batch_size=batch_size,
        Na=Na,
        dim=dim,
        sigma=sigma,
        kT=kT,
        periodic=periodic
    )
    
    # Assert shapes of the statistical summaries
    # Expected: (17, 1) because dr is (B, N, 1)
    assert mean.shape == (17, 1)
    assert var.shape == (17, 1)
    
    # Final sanity check for NaNs
    assert not torch.isnan(mean).any()
    assert not torch.isnan(var).any()
