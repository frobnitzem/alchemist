from tkinter import X
import torch, math, copy, time, json
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from alchemist.summaryAnalysis import _build_neighbor_pairs, _compute_interactions
from alchemist.flows import GlowBlock, MultiStep, Q, RealNVP, LeapFrog, fix_kT
from alchemist.pdbs import read_pdb_coords, write_pdb_trajectory
from alchemist.neighbors import compute_neighbor_masks, get_neighbor_indices, assemble_neighbor_features
from alchemist.ml_utils import train_and_summarize
from alchemist.analysis import compute_neighbor_histograms, compute_energy_parameterized
# --- Configuration ---
PDB_PATH = Path('examples/GaAs/GaAs.pdb')

CUTS = [2.5, 4.5]
BATCH_SIZE = 100
# NA = 216
DIM = 2
# SIGMA = 10.0
# KT = 0.0257  #note k = 8.617333 *10^-5 eV/K, T = 298.15K, kT = 0.0257 eV but we are using arbitrary units so we can set kT = 1 for simplicity
LR = 1e-3
TRAIN_ITERS = 11
N_STEPS_FLOW = 4
EPOCHS = 25
HIDDEN_DIMS = [8,8,8]
# output_dir = 'outputs/only_train_kT9'

# Energy Parameters
MU = torch.tensor([0,0], dtype=torch.float32)
E1 = torch.tensor([[-0.3,-0.5],[-0.5,-0.1]], dtype=torch.float32)
E2 = torch.tensor([[-0.05,-0],[-0,-0.05]], dtype=torch.float32)
# E1 = torch.tensor([[-0.1,-0.5],[-0.5,-0.1]], dtype=torch.float32)
# E2 = torch.tensor([[-0.05,-0],[-0,-0.05]], dtype=torch.float32)

def main(runtype = 'RealNVP'):
    PDB_PATH = Path(f'examples/GaAs/GaAs{NA}.pdb')
    num_repeats = round((NA / 8)**(1/3),0)
    BOX = torch.full((3,), 5.75 * num_repeats)

    # 1. Setup Geometry
    coords = read_pdb_coords(PDB_PATH)
    masks = compute_neighbor_masks(coords, BOX, CUTS)
    neighborlists = get_neighbor_indices(masks)
    
    # 3. Training Utilities
    normal = torch.distributions.normal.Normal(0, 1)
    first_pairs = _build_neighbor_pairs(neighborlists[0])
    second_pairs = _build_neighbor_pairs(neighborlists[1])

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
                'p': Q(fix_kT(normal.sample((BATCH_SIZE, NA, DIM)),KT)),
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

        if runtype == 'Glow':
            bound = 0.5 * (x["r"] / SIGMA).square().sum(dim=(1, 2)) + 0.5 * x["p"].square().sum(dim=(1, 2)) / KT
        else:
            bound = (x["r"] / SIGMA).square().sum(dim=(1, 2))

        loss = 1 / KT * (energy.sum(1) - energy0.sum(1)) - logJ + bound
        
        # energy is (B, N), energy.sum(1) is (B,)
        # logJ is (B,)
        # loss = 1 / KT * (energy.sum(1) - energy0.sum(1)) - logJ
        return loss.mean(), bound.mean()

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
        percents_A = percents.mean(dim=(0,1))[0]

        percents = torch.softmax(x['r'], dim=-1) # (B, N, D)
        percents_A_forinteractions = percents[:,:,0]
        #average over all batches after calculation interactions
        interactions = torch.zeros((B, 6), dtype=torch.float32)
        for i in range(B):
            percent_A = percents_A_forinteractions[i]
            interactions[i] = _compute_interactions(percent_A, first_pairs, second_pairs)

        return [percents_A.detach(), interactions[:,1].mean().detach()]

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
        num_samples = 20
        # per_atom_energy = torch.zeros(num_samples * BATCH_SIZE, NA)
        # p_a_all = torch.zeros(num_samples * BATCH_SIZE, NA)

        #split compositions into comps and interactions
        interactions = [comp[1]/864 for comp in compositions]
        compositions = [comp[0] for comp in compositions]
        
        t0 = time.perf_counter()
        if losses is not None:
            flow.eval()
            p_a_all = torch.empty((num_samples * BATCH_SIZE, NA), dtype=torch.float32)
            with torch.no_grad():
                for i in range(num_samples):
                    if (i+1) % 200 == 0:
                        print(f"Generating sample {i+1}/{num_samples}")
                    x = data_gen().__next__()
                    x, _, _ = flow(x)
                    # feat = assemble_neighbor_features(x['r'], neighborlists)
                    # per_atom_energy[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = compute_energy_parameterized(feat, MU, E1, E2)
                    # p_a_all[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = torch.softmax(feat, dim=-1)[..., 0, 0]
                    p_a_all[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = torch.softmax(x['r'], dim=-1)[..., 0]
                    
        else: 
            x = data_gen().__next__()
            num_samples = 1
            for i in range(3000):
                # if (i+1) % 200 == 0:
                #     print(f"Generating sample {i+1}/{num_samples}")
                x, _, _ = flow(x)
                x = {k: v.detach().requires_grad_(True) for k, v in x_new.items()}
                # feat = assemble_neighbor_features(x['r'], neighborlists)
                # per_atom_energy[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = compute_energy_parameterized(feat, MU, E1, E2)
                # p_a_all[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = torch.softmax(feat, dim=-1)[..., 0, 0]

        t1 = time.perf_counter()
        
        # per_atom_energy = per_atom_energy.detach()
        # p_a_all = p_a_all.detach()

        # flat_energy = per_atom_energy.flatten()
        # flat_p_a = p_a_all.flatten()
        
        # fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
        
        # # (0,0) 2D Histogram: p_a vs Energy
        # axes2[0,0].hist2d(flat_p_a.numpy(), flat_energy.numpy(), bins=30)
        # axes2[0,0].set_title("Energy vs Composition (p_a)")
        # axes2[0,0].set_xlabel("p_a")
        # axes2[0,0].set_ylabel("Energy")
        
        # # (0,1) Marginal: Energy Distribution
        # axes2[0,1].hist(flat_energy.numpy(), bins=30)
        # axes2[0,1].set_title("Energy Marginal")
        # axes2[0,1].set_xlabel("Energy")
        
        # # (1,0) Marginal: p_a Distribution
        # argmax_p_a = (flat_p_a > 0.5).float()
        # axes2[1,0].hist(flat_p_a.numpy(), bins=30)
        # axes2[1,0].set_title(f"Composition Marginal: AVG p_a = {argmax_p_a.mean().item():.3f}")
        # axes2[1,0].set_xlabel("p_a")
        
        # axes2[1,1].axis('off') # Empty panel
        
        # plt.tight_layout()
        # plt.savefig(f'{output_dir}/energy_analysis.png')

        # Plot 3: Loss and composition double plot
        fig3, axes3 = plt.subplots(1, 1, figsize=(6, 4))
        #average compositions over batches
        compositions = torch.stack(compositions, dim=0).detach().numpy() # (EPOCHS, D) -> (EPOCHS,)
        color = 'tab:red'
        
        axes3.set_ylabel('Composition', color=color)
        
        if losses is not None:
            axes3.scatter(range(compositions.shape[0]), compositions, color=color, label='A Composition')
            axes3.scatter(range(compositions.shape[0]), interactions, color='tab:blue', label='A-B Interactions')

            axes3.set_xlabel('Batch')
            losses = torch.tensor(losses).detach().numpy() # (EPOCHS, 2) -> (EPOCHS, 2)
            all_losses = losses[:, 0]
            lJ_losses = -losses[:, 1]
            boundary_losses = losses[:,2]
            U_losses = all_losses - lJ_losses - boundary_losses

            axes4 = axes3.twinx()  # instantiate a second Axes that shares the same x-axis
            color = 'tab:blue'
            axes4.set_ylabel('Loss', color=color)
            axes4.plot(all_losses, color=color, label='Total Loss')
            axes4.plot(lJ_losses, color='tab:orange', label='-logJ')
            axes4.plot(U_losses, color='tab:green', label=r'$\Delta U$/kT')
            axes4.plot(boundary_losses, color='tab:purple', label='Boundary Loss')
            axes4.tick_params(axis='y', labelcolor=color)
            axes4.set_ylim(-100, 500)
            lines_1, labels_1 = axes3.get_legend_handles_labels()
            lines_2, labels_2 = axes4.get_legend_handles_labels()

            # Add combined legend to the primary axis
            axes3.legend(lines_1 + lines_2, labels_1 + labels_2)
            # axes3.set_title(f"Loss and Composition Analysis with {runtype} Flow, NA={NA}, kT={KT}")
        else:
            axes3.scatter((torch.arange(compositions.shape[0])+1)*10, compositions, color=color, label='A Composition')
            axes3.scatter((torch.arange(compositions.shape[0])+1)*10, interactions, color='tab:blue', label='A-B Interactions')
            axes3.set_xlabel('Time step (dt)')
            # axes3.set_title(f"Composition Analysis with Leapfrog Flow, NA={NA}, kT={KT}")
            axes3.legend()
        
        axes3.set_ylim(0, 1)

        fig3.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(f'{output_dir}/ Composition_and_Loss_Analysis.png')

        # 6. Save Trajectory
        # We need Ga percents for the PDB writer
        # ga_percents = p_a_all
        # as_percents = 1.0 - p_a_all
        # #downsample by factor of 10
        # ga_percents = ga_percents[::50]
        # as_percents = as_percents[::50]
        # write_pdb_trajectory(Path(f'{output_dir}/generated_samples.pdb'), coords, ga_percents, as_percents)

        # 7. Save model
        if losses is not None:
            torch.save(flow.state_dict(), f'{output_dir}/trained_flow_model.pth')

        # 8. Save configuration
        # config = {
        #     'PDB_PATH': str(PDB_PATH),
        #     'BOX': BOX.tolist(),
        #     'CUTS': CUTS,
        #     'BATCH_SIZE': BATCH_SIZE,
        #     'NA': NA,
        #     'DIM': DIM,
        #     'SIGMA': SIGMA,
        #     'KT': KT,
        #     'LR': LR,
        #     'TRAIN_ITERS': TRAIN_ITERS,
        #     'N_STEPS_FLOW': N_STEPS_FLOW,
        #     'EPOCHS': EPOCHS,
        #     'MU': MU.tolist(),
        #     'E1': E1.tolist(),
        #     'E2': E2.tolist()
        # }
        # with open(f'{output_dir}/config.txt', 'w') as f:
        #     for key, value in config.items():
        #         f.write(f"{key}: {value}\n")

        # print("Results saved to output directory:", output_dir)
        # p_A_overall = p_a_all.mean().item()
        # try:
        #     K = p_A_overall / (1 - p_A_overall)
        # except ZeroDivisionError:
        #     K = 10

        # del p_A_overall, p_a_all, per_atom_energy, flat_energy, flat_p_a
        torch.cuda.empty_cache()

        return None, (t1 - t0)/num_samples, _
    
    print("Starting training...")
    if runtype == 'RealNVP':
        output_dir = f'Fig5/RealNVP_NA{NA}_KT{KT}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        loss_fn = loss_KL
        data_gen = data_gen_NVP
        flow = RealNVP(DIM, NA, hidden_dims=HIDDEN_DIMS, n_layers=N_STEPS_FLOW)
        optimizer = optim.Adam(flow.parameters(), lr=LR)

        train_start = time.perf_counter()
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
        train_end = time.perf_counter()
        compositions = vals
        losses = losses_train
        K, sample_time, _ = graphing(compositions, output_dir, data_gen, losses)
        return K, sample_time, (train_end - train_start)
    elif runtype == 'Glow':
        output_dir = f'Fig5/Glow_NA{NA}_KT{KT}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        loss_fn = loss_KL
        data_gen = data_gen
        def data_expansion(r):
            return assemble_neighbor_features(r, neighborlists).reshape(r.shape[0], r.shape[1], -1)
        
        glow = GlowBlock(dim=DIM, dt=0.001, hidden_dims=HIDDEN_DIMS, data_size = 17, data_expansion=data_expansion)
        flow = MultiStep(glow, N_STEPS_FLOW)
        
        optimizer = optim.Adam(flow.parameters(), lr=LR)

        train_start = time.perf_counter()
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
        train_end = time.perf_counter()
        compositions = vals
        losses = losses_train
        K, sample_time, _ = graphing(compositions, output_dir, data_gen, losses)
        return K, sample_time, (train_end - train_start)
    elif runtype == 'leapfrog':
        output_dir = f'Fig5/LeapFrog_NA{NA}_KT{KT}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        def U(r, t):
            return compute_energy_parameterized(assemble_neighbor_features(r, neighborlists), MU, E1, E2).sum(dim=1) + KT*(r*r).sum((-2,-1))/(2*SIGMA**2)
        
        flow = LeapFrog(U, const_kT = KT, dt = leapfrogdt)
        vals = []
        x0 = next(data_gen())
        x = copy.deepcopy(x0)
        logJ = 0.0
        
        if run_times == True:
            t0 = time.perf_counter()
        for epoch in range(3000):
            x_new, lJ, info = flow(x)
            logJ += lJ.detach()
            x = {k: v.detach().requires_grad_(True) for k, v in x_new.items()}
            if (epoch+1) % 10 == 0:
                print(f"Leapfrog training epoch {epoch+1}/3000")
                feat = feature_extractor(x)
                vals.append(feat)
                if epoch == 9:
                    t2 = time.perf_counter()
        if run_times == True:
            t1 = time.perf_counter()
        compositions = vals
        print("Leapfrog training complete. Generating samples and analysis...")
        graphing(compositions, output_dir, data_gen)
        return 0, t2-t0, (t1 - t0)
    else:
        assert False, f"Unknown runtype: {runtype}"
    print("Training complete.")

if __name__ == "__main__":
    SIGMA = 5
    if True:
        leapfrogdt = 0.5
        run_times = True
        #times plot
        #get times for NA = 8, 64, 216, 512
        NAs = [216]#[64, 216, 512] #512
        KT = 0.5

        makedata = True

        if makedata:
            samples_times_NVP = []
            samples_times_Glow = []
            samples_times_LeapFrog = []
            train_times_NVP = []
            train_times_Glow = []
            train_times_LeapFrog = []
            for NA in NAs:
                K_NVP, samples_time_NVP, train_time_NVP = main(runtype='RealNVP')
                
                raise NotImplementedError("stop")
                K_Glow, samples_time_Glow, train_time_Glow = main(runtype='Glow')
                K_LeapFrog, samples_time_LeapFrog, train_time_LeapFrog = main(runtype='leapfrog')
                K_NVP, samples_time_NVP, train_time_NVP = main(runtype='RealNVP')

                samples_times_NVP.append(samples_time_NVP)
                samples_times_Glow.append(samples_time_Glow)
                samples_times_LeapFrog.append(samples_time_LeapFrog)
                train_times_NVP.append(train_time_NVP)
                train_times_Glow.append(train_time_Glow)
                train_times_LeapFrog.append(train_time_LeapFrog)
        else: #load data from json
            with open('Fig3/time_data.json', 'r') as f:
                time_data = json.load(f)
            samples_times_NVP = time_data['samples_NVP']
            samples_times_Glow = time_data['samples_Glow']
            samples_times_LeapFrog = time_data['samples_LeapFrog']
            train_times_NVP = time_data['train_NVP']
            train_times_Glow = time_data['train_Glow']
            train_times_LeapFrog = time_data['train_LeapFrog']
        
        # Plot the times
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(NAs, samples_times_NVP, label='Sample RealNVP', marker='o')
        ax.plot(NAs, samples_times_Glow, label='Sample Glow', marker='o')
        ax.plot(NAs, samples_times_LeapFrog, label='Sample LeapFrog', marker='o')
        ax.plot(NAs, train_times_NVP, label='Train RealNVP', marker='x')
        ax.plot(NAs, train_times_Glow, label='Train Glow', marker='x')
        ax.plot(NAs, train_times_LeapFrog, label='Train LeapFrog', marker='x')
        #log scale in y as it is so much slower for leapfrog
        ax.set_yscale('log')
        ax.set_xlabel(r'Number of Atoms ($N_A$)')
        ax.set_ylabel('Time per Sample (s)')
        # ax.set_title('Time per Sample vs Number of Atoms')
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'Fig5/time_per_sample.png')

        #save to a dict
        time_data = {
            'NAs': NAs,
            'samples_NVP': samples_times_NVP,
            'samples_Glow': samples_times_Glow,
            'samples_LeapFrog': samples_times_LeapFrog,
            'train_NVP': train_times_NVP,
            'train_Glow': train_times_Glow,
            'train_LeapFrog': train_times_LeapFrog
        }
        #save as json
        with open('Fig5/time_data.json', 'w') as f:
            json.dump(time_data, f, indent=4)