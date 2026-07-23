from tkinter import X
import torch, math
import torch.optim as optim
import torch.nn.functional as F
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
BATCH_SIZE = 10
NA = 216
DIM = 2
SIGMA = 1.0
KT = 0.1
LR = 1e-3
TRAIN_ITERS = 10
N_STEPS_FLOW = 20
EPOCHS = 20
HIDDEN_DIMS = [32,32,32]
output_dir = 'outputs/initialize_and_train_new'

# Energy Parameters
MU = torch.zeros(DIM, dtype=torch.float32)
E1 = torch.tensor([[-0.3,-0.5],[-0.5,-0.1]], dtype=torch.float32)
E2 = torch.tensor([[-0.15,-0],[-0,-0.05]], dtype=torch.float32)

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
    
    def data_gen_ordered(vsample = False):
        """Generate data in a specific order."""
        idx = (torch.arange(1, 55).repeat_interleave(4)) % 2
        base = torch.nn.functional.one_hot(idx, num_classes=2).float() #(N, 2)
        base = base.unsqueeze(0)
        while True:
            r = base.repeat(BATCH_SIZE, 1, 1) #(B, N, 2)
            if normal.sample((1,)).item() < 0:
                r = 1-r
            if vsample:
                r, logq = sample_v(r)
            else:
                r = r * 10
                logq = torch.zeros(BATCH_SIZE, dtype=torch.float32)
            x = {
                'r': r, 
                'p': Q(normal.sample((BATCH_SIZE, NA, DIM))),
                't': 0.0,
                'loss': logq
                }
            yield x

    def generate_ideal_gas_sample(
        ratio= [0.75, 0.25],
        shuffle=True,
    ): 
        ratio = torch.tensor(ratio, dtype=torch.float32)
        assert torch.sum(ratio) == 1.0, "Ratio must sum to 1."
        assert len(ratio) == DIM

        ratio = torch.tensor(ratio, dtype=torch.float32)

        counts = torch.round(ratio * NA).long()
        diff = NA - counts.sum()
        counts[-1] += diff   # adjust last class

        base_labels = torch.cat([
            torch.full((counts[i],), i, dtype=torch.long)
            for i in range(DIM)
        ], dim=0)

        labels = base_labels.unsqueeze(0).expand(BATCH_SIZE, NA).clone()

        while True:
            if shuffle:
                perm = torch.rand(BATCH_SIZE, NA).argsort(dim=1)
                labels = torch.gather(labels, dim=1, index=perm)
            onehot = F.one_hot(labels, num_classes=DIM).float()

            # Sample v ~ q(v|x) satisfying argmax(v) = x.
            v, logq = sample_v(onehot)

            # q(a | v, x) = N(0, I), chosen here to be independent.
            # Keep the original tensor because flow mutates/replaces state entries.
            p_aux = torch.randn_like(v)

            yield {
                "labels": labels,
                "r": v,
                "p": p_aux,
                "t": torch.tensor(0.0),
                "loss": logq
            }
    
    def sample_v(x_onehot):
        assert DIM == 2, "This helper is for binary categories only."

        device = x_onehot.device
        dtype = x_onehot.dtype

        labels = x_onehot.argmax(dim=-1)  # (B, N)

        # Sample u from iid standard logistic.
        # v = logit(eps), u ~ Uniform(0, 1)
        u = torch.rand(BATCH_SIZE, NA, DIM, device=device, dtype=dtype).clamp(1e-6, 1.0 - 1e-6)
        v = torch.log(u) - torch.log1p(-u)

        # if argmax is wrong, flip the two channels
        wrong = v.argmax(dim=-1) != labels          # (B, N)
        v_flipped = v.flip(dims=[-1])               # swap class 0 and 1
        v = torch.where(wrong.unsqueeze(-1), v_flipped, v)

        # log q(v|x)
        # flip/order transform gives factor 2 per site
        logq = (F.logsigmoid(v) + F.logsigmoid(-v)).sum(dim=[1, 2]) + NA * math.log(2.0)

        return v, logq

    def loss_ELBO(x0, x, logJ): #ELBO loss
        """
        Correct ideal-gas Argmax Flow loss.
            ELBO = E_{v ~ q(v|x)} [log p(v) - log q(v|x)]
        """

        # Base density log p(z)
        logpz = -0.5 * math.log(2.0 * math.pi * SIGMA ** 2) - 0.5 * (x["r"] / SIGMA) ** 2
        logpz = logpz.sum(dim=[1, 2])

        logpp = -0.5 * math.log(2.0 * math.pi * SIGMA ** 2) - 0.5 * (x['p'] / SIGMA) ** 2
        logpp = logpp.sum(dim=[1, 2])
        logq_p_aux = -0.5 * math.log(2.0 * math.pi * SIGMA ** 2) - 0.5 * (x0['p'] / SIGMA) ** 2
        logq_p_aux = logq_p_aux.sum(dim=[1, 2]) 

        loss = -(logpz + logJ - x['loss'] - logq_p_aux + logpp)

        loss = -(logpz + logJ - x['loss'])

        return loss.mean()

    def loss_KL(x0,x,logJ):
        
        assembled = assemble_neighbor_features(x['r'], neighborlists)
        energy = compute_energy_parameterized(assembled, MU, E1, E2)
        
        assembled0 = assemble_neighbor_features(x0['r'], neighborlists)
        energy0 = compute_energy_parameterized(assembled0, MU, E1, E2) #compute_energy_parameterized has a softmax in it
        
        # energy is (B, N), energy.sum(1) is (B,)
        # logJ is (B,)
        loss = 1 / KT * (energy.sum(1) - energy0.sum(1)) - logJ
        return loss.mean()

    def loss_MLE(x0, x, logJ):
        """
        Maximum Likelihood Estimation (MLE) loss for reverse flow.
        """
        # Base density log p(z)
        logpz = -0.5 * math.log(2.0 * math.pi * SIGMA ** 2) - 0.5 * (x["r"] / SIGMA) ** 2
        logpz = logpz.sum(dim=[1, 2])

        logpp = -0.5 * math.log(2.0 * math.pi * SIGMA ** 2) - 0.5 * (x['p'] / SIGMA) ** 2
        logpp = logpp.sum(dim=[1, 2])
        
        loss = -(logpz + logJ + logpp)

        return loss.mean()

    def feature_extractor(x):
        B, N, D = x['r'].shape
        percents = torch.softmax(x['r'], dim=-1) # (B, N, D)
        neighbor_percents = assemble_neighbor_features(percents, neighborlists) # (B, N, 17, D)
        # create semi covariance matrix: (B, N, D, D) of p_a * p_a first neighbors
        p_a = neighbor_percents[..., 0,:] # (B, N, D)
        p_a_neighbors = neighbor_percents[..., 1:5, :] # (B, N, 4, D)
        # Compute covariance: (B, N, D, D)
        cov = torch.einsum('bni,bnxj->ij', p_a, p_a_neighbors)/(B*N)
        return cov
        
    # Initialize with initial samples to get a baseline for the features
    print("Starting initialization...")
    means, vars, losses_init = train_and_summarize(
        model=flow,
        loss_fn=loss_MLE,
        data_generator=data_gen_ordered,
        optimizer=optimizer,
        epochs=EPOCHS,
        batches_per_epoch=TRAIN_ITERS - 1,
        feature_extractor=feature_extractor,
        inverse=True
    )
    print("Initialization complete.")
    
    # 4. Train and Summarize
    print("Starting training...")
    means, vars, losses_train = train_and_summarize(
        model=flow,
        loss_fn=loss_KL,
        data_generator=data_gen,
        optimizer=optimizer,
        epochs=EPOCHS,
        batches_per_epoch=TRAIN_ITERS - 1,
        feature_extractor=feature_extractor,
        inverse=False
    )
    print("Training complete.")
    
    losses = losses_init + losses_train
    # losses = [0]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 5. Final Sampling and Analysis
    num_samples = 1000
    samples_features = []
    
    flow.eval()
    with torch.no_grad():
        for _ in range(num_samples):
            if _+1 % 100 == 0:
                print(f"Generating sample {_}/{num_samples}")
            x = {
                'r': Q(normal.sample((1, NA, DIM))) * SIGMA, 
                'p': Q(normal.sample((1, NA, DIM))),
                't': 0.0
                }
            
            x, _, _ = flow(x)
            # for i in range(N_STEPS_FLOW*2):
            #     neighbor = assemble_neighbor_features(x['r'], neighborlists)
            #     x['r'] = -(neighbor[:, :, 1:5, :].mean(dim=2))*2
            
            feat = assemble_neighbor_features(x['r'], neighborlists)
            samples_features.append(feat)

    all_feat = torch.cat(samples_features, dim=0) # (num_samples, N, 17, D)
    
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

    # Plot 1.1: Neighbor Histograms, input
    gen_feat = assemble_neighbor_features(data_gen_ordered().__next__()['r'], neighborlists)
    nn1_hist, nn2_hist = compute_neighbor_histograms(gen_feat)
    
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    axes1[0].imshow(nn1_hist.numpy(), extent=[0,1,0,1], origin='lower')
    axes1[0].set_title("NN1 Composition Histogram")
    axes1[0].set_xlabel("Neighbor p_a")
    axes1[0].set_ylabel("Center p_a")
    
    axes1[1].imshow(nn2_hist.numpy(), extent=[0,1,0,1], origin='lower')
    axes1[1].set_title("NN2 Composition Histogram")
    axes1[1].set_xlabel("Neighbor p_a")
    axes1[1].set_ylabel("Center p_a")
    plt.savefig(f'{output_dir}/neighbor_histograms_input.png')
    
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
    argmax_p_a = (flat_p_a > 0.5).float()
    axes2[1,0].hist(flat_p_a.numpy(), bins=30)
    axes2[1,0].set_title(f"Composition Marginal: AVG p_a = {argmax_p_a.mean().item():.3f}")
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
    ga_percents = p_a_all
    as_percents = 1.0 - p_a_all
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
