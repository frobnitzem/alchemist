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
TRAIN_ITERS = 10
N_STEPS_FLOW = 10
EPOCHS = 100
HIDDEN_DIMS = [32]
output_dir = 'outputs/original_GaAs'

# Energy Parameters
MU = torch.zeros(DIM, dtype=torch.float32)
E1 = torch.tensor([[0.3,0.5],[0.5,0.1]], dtype=torch.float32)
E2 = torch.tensor([[0.15,0],[0,0.05]], dtype=torch.float32)

def main():
    # 1. Setup Geometry
    coords = read_pdb_coords(PDB_PATH)
    masks = compute_neighbor_masks(coords, BOX, CUTS)
    neighborlists = get_neighbor_indices(masks)

    def data_expansion(r):
        return assemble_neighbor_features(r, neighborlists).reshape(r.shape[0], r.shape[1], -1)
        
    # 2. Model Setup
    glow = GlowBlock(dim=DIM, dt=0.001, hidden_dims=HIDDEN_DIMS, data_size = 17, data_expansion=data_expansion)
    flow = MultiStep(glow, N_STEPS_FLOW)
    optimizer = optim.Adam(glow.parameters(), lr=LR)
    
    # 3. Training Utilities
    normal = torch.distributions.normal.Normal(0, 1)
    
    def data_gen():
        while True:
            x = {
                'r': Q(normal.sample((BATCH_SIZE, NA, DIM))) * SIGMA, 
                'p': Q(normal.sample((BATCH_SIZE, NA, DIM))),
                't': 0.0
                }
            yield x

    def loss_fn(x0,x,logJ): #KL divergence loss
        
        assembled = assemble_neighbor_features(x['r'], neighborlists)
        energy = compute_energy_parameterized(assembled, MU, E1, E2)
        
        assembled0 = assemble_neighbor_features(x0['r'], neighborlists)
        energy0 = compute_energy_parameterized(assembled0, MU, E1, E2) #compute_energy_parameterized has a softmax in it
        
        # energy is (B, N), energy.sum(1) is (B,)
        # logJ is (B,)
        loss = 1 / KT * (energy.sum(1) - energy0.sum(1)) - logJ
        return loss.mean()

    def feature_extractor(x):
        B, N, D = x['r'].shape
        percents = torch.softmax(x['r'], dim=-1) # (B, N, D)
        neighbor_percents = assemble_neighbor_features(x['r'], neighborlists) # (B, N, 17, D)
        # create semi covariance matrix: (B, N, D, D) of p_a * p_a first neighbors
        p_a = neighbor_percents[..., 0,:] # (B, N, D)
        p_a_neighbors = neighbor_percents[..., 1:5, :] # (B, N, 4, D)
        # Compute covariance: (B, N, D, D)
        cov = torch.einsum('bni,bnxj->ij', p_a, p_a_neighbors)/(B*N)
        return cov

    # 4. Train and Summarize
    print("Starting training...")
    means, vars, losses = train_and_summarize(
        model=flow,
        loss_fn=loss_fn,
        data_generator=data_gen,
        optimizer=optimizer,
        epochs=EPOCHS,
        batches_per_epoch=TRAIN_ITERS - 1,
        feature_extractor=feature_extractor
    )
    print("Training complete.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 5. Final Sampling and Analysis
    num_samples = 100
    samples_features = []
    samples_r = []
    
    flow.eval()
    with torch.no_grad():
        for _ in range(num_samples):
            x = {
                'r': Q(normal.sample((1, NA, DIM))) * SIGMA, 
                'p': Q(normal.sample((1, NA, DIM))),
                't': 0.0
                }
            
            x, _, _ = flow(x)
            
            feat = assemble_neighbor_features(x['r'], neighborlists)
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
    plt.savefig(f'{output_dir}/neighbor_histograms.png')
    
    # Plot 2: Energy Analysis
    per_atom_energy = compute_energy_parameterized(all_feat, MU, E1, E2) # (B, N)
    B, N = per_atom_energy.shape
    
    # Get p_a for all atoms: (B, N, 17, D) -> (B, N, D) -> (B, N)
    # We take the first slot (self) and the first component (p_a)
    p_a_all = torch.softmax(all_feat, dim=-1)[..., 0, 0]
    
    flat_energy = per_atom_energy.flatten()
    flat_p_a = p_a_all.flatten()
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    
    # (0,0) 2D Histogram: p_a vs Energy
    axes2[0,0].hist2d(flat_p_a.numpy(), flat_energy.numpy(), bins=30)
    axes2[0,0].set_title("Energy vs Composition (p_a)")
    axes2[0,0].set_xlabel("p_a")
    axes2[0,0].set_ylabel("Energy")
    
    # (0,1) Marginal: Energy Distribution
    axes2[0,1].hist(flat_energy.numpy(), bins=30)
    axes2[0,1].set_title("Energy Marginal")
    axes2[0,1].set_xlabel("Energy")
    
    # (1,0) Marginal: p_a Distribution
    axes2[1,0].hist(flat_p_a.numpy(), bins=30)
    axes2[1,0].set_title("Composition Marginal")
    axes2[1,0].set_xlabel("p_a")
    
    axes2[1,1].axis('off') # Empty panel
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/energy_analysis.png')

    # Plot 3: Loss Analysis
    fig3, axes3 = plt.subplots(1, 1, figsize=(12, 5))
    axes3.plot(losses)
    axes3.set_title("Training Loss")
    axes3.set_xlabel("Batch")
    axes3.set_ylabel("Loss")
    plt.savefig(f'{output_dir}/ loss_analysis.png')

    # 6. Save Trajectory
    # We need Ga percents for the PDB writer
    percents = torch.softmax(torch.cat(samples_r, dim=0), dim=-1) # (B, N, D)
    ga_percents = percents[..., 0] # (B, N)
    as_percents = percents[..., 1] # (B, N)
    write_pdb_trajectory(Path(f'{output_dir}/generated_samples.pdb'), coords, ga_percents, as_percents)

    # 7. Save model
    torch.save(flow.state_dict(), f'{output_dir}/trained_flow_model.pth')

    # 8. Save configuration
    config = {
        'PDB_PATH': str(PDB_PATH),
        'BOX': BOX.tolist(),
        'CUTS': CUTS,
        'BATCH_SIZE': BATCH_SIZE,
        'NA': NA,
        'DIM': DIM,
        'SIGMA': SIGMA,
        'KT': KT,
        'LR': LR,
        'TRAIN_ITERS': TRAIN_ITERS,
        'N_STEPS_FLOW': N_STEPS_FLOW,
        'EPOCHS': EPOCHS,
        'MU': MU.tolist(),
        'E1': E1.tolist(),
        'E2': E2.tolist()
    }
    with open(f'{output_dir}/config.txt', 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    print("Results saved to output directory:", output_dir)

if __name__ == "__main__":
    main()
