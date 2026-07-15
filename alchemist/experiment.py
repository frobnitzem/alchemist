import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from alchemist.flows import GlowBlock, MultiStep, Q
from alchemist.pdbs import read_pdb_coords, write_pdb_trajectory
from alchemist.neighbors import compute_neighbor_masks, get_neighbor_indices, assemble_neighbor_features
from alchemist.ml_utils import train_and_summarize
from alchemist.analysis import compute_neighbor_histograms, compute_energy_parameterized

# --- Configuration ---
PDB_PATH = Path('examples/GaAs/GaAs.pdb')
BOX = torch.full((3,), 5.75 * 3)
CUTS = [2.5, 4.5]
BATCH_SIZE = 1
NA = 216
DIM = 2
SIGMA = 1.0
KT = 1.0
LR = 1e-3
TRAIN_ITERS = 25
N_STEPS_FLOW = 10

# Energy Parameters
MU = torch.zeros(DIM)
E1 = torch.eye(DIM) * 0.1
E2 = torch.eye(DIM) * 0.05

def percentA(r):
    dr = r[:, :, 1] - r[:, :, 0]
    return torch.sigmoid(dr)

def main():
    # 1. Setup Geometry
    coords = read_pdb_coords(PDB_PATH)
    masks = compute_neighbor_masks(coords, BOX, CUTS)
    neighborlists = get_neighbor_indices(masks)
    
    # 2. Model Setup
    glow = GlowBlock(neighborlists=neighborlists, dim=DIM, dt=0.001)
    flow = MultiStep(glow, N_STEPS_FLOW)
    optimizer = optim.Adam(glow.parameters(), lr=LR)
    
    # 3. Training Utilities
    normal = torch.distributions.normal.Normal(0, 1)
    
    def data_gen():
        while True:
            x = {'r': Q(normal.sample((BATCH_SIZE, NA, DIM))) * SIGMA, 't': 0.0}
            p = percentA(x['r'])
            x['p'] = torch.stack([p, 1 - p], dim=2)
            yield x

    def loss_fn(batch):
        x0 = batch.copy()
        x = batch.copy()
        logJ = 0.0
        for _ in range(2):
            x, lJ, info = flow(x)
            logJ = logJ + lJ
        
        # Simple energy for training (using parameterized version for consistency)
        dr = percentA(x['r']).unsqueeze(-1)
        assembled = assemble_neighbor_features(dr, neighborlists)
        energy = compute_energy_parameterized(assembled, MU, E1, E2)
        
        dr0 = percentA(x0['r']).unsqueeze(-1)
        assembled0 = assemble_neighbor_features(dr0, neighborlists)
        energy0 = compute_energy_parameterized(assembled0, MU, E1, E2)
        
        loss = 1 / KT * (energy.sum(1) - energy0.sum(1)) - logJ
        return loss.mean()

    def feature_extractor(model, batch):
        x = batch.copy()
        for _ in range(N_STEPS_FLOW):
            x, _, _ = model(x)
        dr = percentA(x['r']).unsqueeze(-1)
        return assemble_neighbor_features(dr, neighborlists)

    # 4. Train and Summarize
    print("Starting training...")
    flow, mean, var = train_and_summarize(
        model=flow,
        loss_fn=loss_fn,
        data_generator=data_gen,
        optimizer=optimizer,
        epochs=2,
        batches_per_epoch=TRAIN_ITERS - 1,
        feature_extractor=feature_extractor
    )
    print("Training complete.")

    # 5. Final Sampling and Analysis
    num_samples = 100
    samples_features = []
    samples_r = []
    
    flow.eval()
    with torch.no_grad():
        for _ in range(num_samples):
            x = {'r': Q(normal.sample((1, NA, DIM))) * SIGMA, 't': 0.0}
            p = percentA(x['r'])
            x['p'] = torch.stack([p, 1 - p], dim=2)
            
            for _ in range(N_STEPS_FLOW):
                x, _, _ = flow(x)
            
            dr = percentA(x['r']).unsqueeze(-1)
            feat = assemble_neighbor_features(dr, neighborlists)
            samples_features.append(feat)
            samples_r.append(x['r'])

    all_feat = torch.cat(samples_features, dim=0) # (B, N, 17, D)
    
    # Plot 1: Neighbor Histograms
    nn1_hist, nn2_hist = compute_neighbor_histograms(all_feat)
    
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    axes1[0].imshow(nn1_hist.numpy(), extent=[0,1,0,1], origin='lower')
    axes1[0].set_title("NN1 Composition Histogram")
    axes1[0].set_xlabel("Neighbor p_a")
    axes1[0].set_ylabel("Center p_a")
    
    axes1[1].imshow(nn2_hist.numpy(), extent=[0,1,0,1], origin='lower')
    axes1[1].set_title("NN2 Composition Histogram")
    axes1[1].set_xlabel("Neighbor p_a")
    axes1[1].set_ylabel("Center p_a")
    plt.savefig('neighbor_histograms.png')
    
    # Plot 2: Energy Analysis
    per_atom_energy = compute_energy_parameterized(all_feat, MU, E1, E2) # (B, N)
    B, N = per_atom_energy.shape
    flat_energy = per_atom_energy.flatten()
    atom_indices = torch.arange(B * N).float()
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    
    # (0,0) 2D Histogram: Atom Index vs Energy
    axes2[0,0].hist2d(atom_indices.numpy(), flat_energy.numpy(), bins=30)
    axes2[0,0].set_title("Energy vs Atom Index")
    axes2[0,0].set_xlabel("Atom Index")
    axes2[0,0].set_ylabel("Energy")
    
    # (0,1) Marginal: Energy Distribution
    axes2[0,1].hist(flat_energy.numpy(), bins=30)
    axes2[0,1].set_title("Energy Marginal")
    axes2[0,1].set_xlabel("Energy")
    
    # (1,0) Marginal: Atom Index Distribution (should be flat)
    axes2[1,0].hist(atom_indices.numpy(), bins=30)
    axes2[1,0].set_title("Atom Index Marginal")
    axes2[1,0].set_xlabel("Atom Index")
    
    axes2[1,1].axis('off') # Empty panel
    
    plt.tight_layout()
    plt.savefig('energy_analysis.png')

    # 6. Save Trajectory
    # We need Ga percents for the PDB writer
    ga_percents = percentA(torch.cat(samples_r, dim=0)) # (B*1, N)
    as_percents = 1 - ga_percents
    write_pdb_trajectory(Path('generated_samples.pdb'), coords, ga_percents, as_percents)
    print("Results saved to neighbor_histograms.png, energy_analysis.png, and generated_samples.pdb")

if __name__ == "__main__":
    main()
