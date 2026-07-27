from tkinter import X
import torch, math, copy, time
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from alchemist.flows import GlowBlock, MultiStep, Q, RealNVP, LeapFrog
from alchemist.pdbs import read_pdb_coords, write_pdb_trajectory
from alchemist.neighbors import compute_neighbor_masks, get_neighbor_indices, assemble_neighbor_features
from alchemist.ml_utils import train_and_summarize
from alchemist.analysis import compute_neighbor_histograms, compute_energy_parameterized
# --- Configuration ---
PDB_PATH = Path('examples/GaAs/GaAs.pdb')
BOX = torch.full((3,), 5.75 * 3)
CUTS = [2.5, 4.5]
BATCH_SIZE = 50
NA = 216
DIM = 2
SIGMA = 1.0
KT = 0.0257  #note k = 8.617333 *10^-5 eV/K, T = 298.15K, kT = 0.0257 eV but we are using arbitrary units so we can set kT = 1 for simplicity
LR = 1e-3
TRAIN_ITERS = 10
N_STEPS_FLOW = 10
EPOCHS = 20
HIDDEN_DIMS = [8,8,8]
output_dir = 'outputs/only_train_kT9'

# Energy Parameters
MU = torch.tensor([0,0], dtype=torch.float32)
E1 = torch.tensor([[-0.3,-0.5],[-0.5,-0.1]], dtype=torch.float32)
E2 = torch.tensor([[-0.15,-0],[-0,-0.05]], dtype=torch.float32)

def main(runtype = 'RealNVP', num ='1'):
    PDB_PATH = Path(f'examples/GaAs/GaAs{NA}.pdb')
    num_repeats = round((NA / 8)**(1/3),0)
    # 1. Setup Geometry
    coords = read_pdb_coords(PDB_PATH)
    masks = compute_neighbor_masks(coords, BOX, CUTS)
    neighborlists = get_neighbor_indices(masks)
    
    # 3. Training Utilities
    normal = torch.distributions.normal.Normal(0, 1)

    def data_gen_NVP():
        r_coord = torch.tensor(coords, dtype=torch.float32)
        while True:
            r_chem = Q(normal.sample((BATCH_SIZE, NA, DIM))) * SIGMA
            x ={
                'r': r_chem,
                'r_coord': r_coord.repeat(BATCH_SIZE, 1, 1),
                't': torch.tensor(0.0)
            }
            yield x

    def data_gen():
        while True:
            x = {
                'r': Q(normal.sample((BATCH_SIZE, NA, DIM))) * SIGMA, 
                'p': Q(normal.sample((BATCH_SIZE, NA, DIM))),
                't': torch.tensor(0.0),
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
                't': torch.tensor(0.0),
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
        percents_A_per_batch = percents.mean(dim=1)[:,0]
        return percents_A_per_batch 

    # Initialize with initial samples to get a baseline for the features
    # print("Starting initialization...")
    # means, vars, losses_init = train_and_summarize(
    #     model=flow,
    #     loss_fn=loss_MLE,
    #     data_generator=data_gen_ordered,
    #     optimizer=optimizer,
    #     epochs=EPOCHS,
    #     batches_per_epoch=TRAIN_ITERS - 1,
    #     feature_extractor=feature_extractor,
    #     inverse=True
    # )
    # print("Initialization complete.")

    def graphing(compositions, output_dir,data_gen, losses = None):
        num_samples = 500
        samples_features = []
        
        t0 = time.perf_counter()
        if losses is not None:
            flow.eval()
            with torch.no_grad():
                for _ in range(num_samples):
                    if (_+1) % 200 == 0:
                        print(f"Generating sample {_+1}/{num_samples}")
                    x = data_gen().__next__()
                    x, _, _ = flow(x)
                    feat = assemble_neighbor_features(x['r'], neighborlists)
                    samples_features.append(feat)
        else: 
            for _ in range(num_samples):
                if (_+1) % 200 == 0:
                    print(f"Generating sample {_+1}/{num_samples}")
                x = data_gen().__next__()
                x, _, _ = flow(x)
                feat = assemble_neighbor_features(x['r'], neighborlists)
                samples_features.append(feat)
        t1 = time.perf_counter()

        all_feat = torch.cat(samples_features, dim=0).detach() # (num_samples*B, N, 17, D)

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

        # Plot 3: Loss and composition double plot
        fig3, axes3 = plt.subplots(1, 1, figsize=(8, 5))
        #average compositions over batches
        compositions = torch.stack(compositions, dim=0).mean(dim=-1).detach().numpy() # (EPOCHS, D) -> (EPOCHS,)
        color = 'tab:red'
        axes3.set_xlabel('Batch')
        axes3.set_ylabel('Composition', color=color)
        axes3.scatter(range(compositions.shape[0]), compositions, color=color)

        axes4 = axes3.twinx()  # instantiate a second Axes that shares the same x-axis
        if losses is not None:
            color = 'tab:blue'
            axes4.set_ylabel('Loss', color=color)
            axes4.plot(losses, color=color)
            axes3.set_title("Loss and Composition Analysis")
        else:
            axes3.set_title("Composition Analysis")

        fig3.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(f'{output_dir}/ Composition_and_Loss_Analysis.png')

        # 6. Save Trajectory
        # We need Ga percents for the PDB writer
        ga_percents = p_a_all
        as_percents = 1.0 - p_a_all
        #downsample by factor of 10
        ga_percents = ga_percents[::50]
        as_percents = as_percents[::50]
        write_pdb_trajectory(Path(f'{output_dir}/generated_samples.pdb'), coords, ga_percents, as_percents)

        # 7. Save model
        if losses is not None:
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
        p_A_overall = p_a_all.mean().item()
        try:
            K = p_A_overall / (1 - p_A_overall)
        except ZeroDivisionError:
            K = 10
        return K, (t1 - t0)/num_samples
    
    print("Starting training...")
    if runtype == 'RealNVP':
        output_dir = f'Fig5/RealNVP_{num}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        loss_fn = loss_KL
        data_gen = data_gen_NVP
        flow = RealNVP(DIM, NA, hidden_dims=HIDDEN_DIMS, n_layers=N_STEPS_FLOW)
        optimizer = optim.Adam(flow.parameters(), lr=LR)

        vals, losses_train = train_and_summarize(
            model=flow,
            loss_fn=loss_fn,
            data_generator=data_gen,
            optimizer=optimizer,
            epochs=EPOCHS,
            batches_per_epoch=TRAIN_ITERS - 1,
            feature_extractor=feature_extractor,
            inverse=False
        )
        compositions = vals
        losses = losses_train
        return graphing(compositions, output_dir, data_gen, losses)
    elif runtype == 'Glow':
        output_dir = f'Fig5/Glow_{num}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        loss_fn = loss_KL
        data_gen = data_gen
        def data_expansion(r):
            return assemble_neighbor_features(r, neighborlists).reshape(r.shape[0], r.shape[1], -1)
        
        glow = GlowBlock(dim=DIM, dt=0.001, hidden_dims=HIDDEN_DIMS, data_size = 17, data_expansion=data_expansion)
        flow = MultiStep(glow, N_STEPS_FLOW)
        
        optimizer = optim.Adam(flow.parameters(), lr=LR)

        vals, losses_train = train_and_summarize(
            model=flow,
            loss_fn=loss_fn,
            data_generator=data_gen,
            optimizer=optimizer,
            epochs=EPOCHS,
            batches_per_epoch=TRAIN_ITERS - 1,
            feature_extractor=feature_extractor,
            inverse=False
        )
        compositions = vals
        losses = losses_train
        return graphing(compositions, output_dir, data_gen, losses)
    elif runtype == 'leapfrog':
        output_dir = f'Fig5/LeapFrog_{num}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        def U(r, t):
            return compute_energy_parameterized(assemble_neighbor_features(r, neighborlists), MU, E1, E2).sum(dim=1)
        
        flow = LeapFrog(U)
        vals = []
        x0 = next(data_gen())
        x = copy.deepcopy(x0)
        for epoch in range(EPOCHS):
            for _ in range(TRAIN_ITERS-1):
                x, lJ, info = flow(x)

                feat = feature_extractor(x)
                vals.append(feat)
        compositions = vals
        print("Leapfrog training complete. Generating samples and analysis...")
        return graphing(compositions, output_dir, data_gen)
    else:
        assert False, f"Unknown runtype: {runtype}"
    print("Training complete.")

if __name__ == "__main__":
    if True:
        KT = 0.0257  #note k = 8.617333 *10^-5 eV/K, T = 298.15K, kT = 0.0257 eV but we are using arbitrary units so we can set kT = 1 for simplicity
        output_dir = 'outputs/only_train_kT9'
        NA = 216

        KTs = [0.02397,0.02569,0.02715]
        for KT in KTs:
            K_NVP, time_NVP = main(runtype='RealNVP', num=str(NA))
            K_Glow, time_Glow = main(runtype='Glow', num=str(NA))
