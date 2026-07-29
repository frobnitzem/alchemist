from tkinter import TRUE, X
import torch, math, copy, time, json
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from alchemist.flows import GlowBlock, MultiStep, Q, RealNVP, LeapFrog, fix_kT
from alchemist.pdbs import read_pdb_coords, write_pdb_trajectory
from alchemist.neighbors import compute_neighbor_masks, get_neighbor_indices, assemble_neighbor_features
from alchemist.ml_utils import train_and_summarize
from alchemist.analysis import compute_neighbor_histograms, compute_energy_parameterized
# --- Configuration ---
CUTS = [2.5, 4.5]
BATCH_SIZE = 100
# NA = 216
DIM = 2
# SIGMA = 1
# KT = 1
LR = 1e-3
TRAIN_ITERS = 11
N_STEPS_FLOW = 4
EPOCHS = 25
HIDDEN_DIMS = [8,8,8]
# output_dir = 'outputs/only_train_kT9'

# Energy Parameters
MU = torch.tensor([2,3], dtype=torch.float32)
E1 = torch.tensor([[0,0],[0,0]], dtype=torch.float32)
E2 = torch.tensor([[0,0],[0,0]], dtype=torch.float32)

def main(runtype = 'RealNVP'):
    PDB_PATH = Path(f'examples/GaAs/GaAs{NA}.pdb')
    num_repeats = round((NA / 8)**(1/3), 0)  # Calculate the number of repeats needed to achieve NA atoms
    BOX = torch.full((3,), 5.75 * num_repeats)
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
                'p': Q(fix_kT(normal.sample((BATCH_SIZE, NA, DIM)), KT)),
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
        return percents_A

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
        
        t0 = time.perf_counter()
        if losses is not None:
            flow.eval()
            with torch.no_grad():
                p_a_all = torch.empty((num_samples * BATCH_SIZE, NA), dtype=torch.float32)
                for i in range(num_samples):
                    x = data_gen().__next__()
                    x, _, _ = flow(x)
                    # feat = assemble_neighbor_features(x['r'], neighborlists)
                    # per_atom_energy[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = compute_energy_parameterized(feat, MU, E1, E2)
                    # p_a_all[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = torch.softmax(feat, dim=-1)[..., 0, 0]
                    p_a_all[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = torch.softmax(x['r'], dim=-1)[..., 0]
            p_A_overall = p_a_all.mean().item()
                    
        else: 
            last_compositions = torch.stack(compositions, dim=0).detach()[int(len(compositions)*4/5):] #(num_samples, D)
            # feat = assemble_neighbor_features(x['r'], neighborlists)
            # per_atom_energy[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = compute_energy_parameterized(feat, MU, E1, E2)
            # p_a_all[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = torch.softmax(feat, dim=-1)[..., 0, 0]
            p_A_overall = last_compositions.mean().item()
        t1 = time.perf_counter()
        
        # per_atom_energy = per_atom_energy.detach()
        # p_a_all = p_a_all.detach()

        # # flat_energy = per_atom_energy.flatten()
        # flat_p_a = p_a_all.flatten()
        
        # fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
        
        # (0,0) 2D Histogram: p_a vs Energy
        # axes2[0,0].hist2d(flat_p_a.numpy(), flat_energy.numpy(), bins=30)
        # axes2[0,0].set_title(f"Energy vs Composition with {runtype} Flow, NA={NA}, kT={KT}")
        # axes2[0,0].set_xlabel("p_a")
        # axes2[0,0].set_ylabel("Energy")
        
        # # (0,1) Marginal: Energy Distribution
        # axes2[0,1].hist(flat_energy.numpy(), bins=30)
        # axes2[0,1].set_title(f"Energy Marginal with {runtype} Flow, NA={NA}, kT={KT}")
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
        fig3, axes3 = plt.subplots(1, 1, figsize=(8, 5))
        #average compositions over batches
        compositions = torch.stack(compositions, dim=0).detach().numpy() # (EPOCHS, D) -> (EPOCHS,)
        color = 'tab:red'
        axes3.set_ylabel('Composition', color=color)

        if losses is not None:
            axes3.scatter(range(compositions.shape[0]), compositions, color=color)
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
            axes4.plot(U_losses, color='tab:green', label=r"$\Delta U$/kT")
            axes4.plot(boundary_losses, color='tab:purple', label='Boundary Loss')
            axes4.tick_params(axis='y', labelcolor=color)
            axes4.legend()
            axes3.set_title(f"Loss and Composition Analysis with {runtype} Flow, NA={NA}, kT={KT}")
        else:
            axes3.scatter((torch.arange(compositions.shape[0])+1)*10, compositions, color=color)
            axes3.set_xlabel('Time step (dt)')
            axes3.set_title(f"Composition Analysis with {runtype} Flow, NA={NA}, kT={KT}")
        
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
        
        try:
            K = (1 - p_A_overall)/p_A_overall
        except ZeroDivisionError:
            K = 10
            
        torch.cuda.empty_cache()

        #save results
        results = {
            'K': K,
            'time': (t1 - t0)/num_samples,
        }
        torch.save(results, f'{output_dir}/results.pt')

        return K, (t1 - t0)/num_samples
    
    print("Starting training...")
    if runtype == 'RealNVP':
        output_dir = f'Fig3/RealNVP_NA{NA}_KT{KT}'
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
        output_dir = f'Fig3/Glow_NA{NA}_KT{KT}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        loss_fn = loss_KL
        data_gen = data_gen
        glow = GlowBlock(dim=DIM, dt=0.001, hidden_dims=HIDDEN_DIMS)
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
        output_dir = f'Fig3/LeapFrog_NA{NA}_KT{KT}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        def U(r, t):
            percents = torch.softmax(r, dim=-1)
            return MU[0]*percents[..., 0].sum(dim=-1) + MU[1]*percents[..., 1].sum(dim=-1) + KT*(r*r).sum((-2,-1))/(2*SIGMA**2)
        
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
                vals.append(feat.detach())
        compositions = vals
        if run_times == True:
            t1 = time.perf_counter()
            return 0, (t1 - t0)
        return graphing(compositions, output_dir, data_gen)
    else:
        assert False, f"Unknown runtype: {runtype}"
    print("Training complete.")

if __name__ == "__main__":
    SIGMA = 5
    if False:
        Ks_NVP = []
        Ks_Glow = []
        Ks_LeapFrog = []

        #has units kT
        KTs = [0.25, 0.5, 1.0, 2.0, 4.0] #scale should be 1/kT from 0 to 4
        NA = 216
        leapfrogdts = [0.5, 0.5, 0.5, 0.5, 0.5]

        makedata = True

        if makedata:
            for i, KT in enumerate(KTs):
                leapfrogdt = leapfrogdts[i]
                K_LeapFrog, time_LeapFrog = main(runtype='leapfrog')
                K_Glow, time_Glow = main(runtype='Glow')
                K_NVP, time_NVP = main(runtype='RealNVP')
                
                Ks_NVP.append(K_NVP)
                Ks_Glow.append(K_Glow)
                Ks_LeapFrog.append(K_LeapFrog)

            #simulation results
            lnK_NVP = torch.log(torch.tensor(Ks_NVP, dtype=torch.float32))
            lnK_Glow = torch.log(torch.tensor(Ks_Glow, dtype=torch.float32))
            lnK_LeapFrog = torch.log(torch.tensor(Ks_LeapFrog, dtype=torch.float32))
        else: #load data from json
            with open('Fig3/vant_hoff_data.json', 'r') as f:
                vant_hoff_data = json.load(f)
            lnK_NVP = torch.tensor(vant_hoff_data['lnK_NVP'], dtype=torch.float32)
            lnK_Glow = torch.tensor(vant_hoff_data['lnK_Glow'], dtype=torch.float32)
            lnK_LeapFrog = torch.tensor(vant_hoff_data['lnK_LeapFrog'], dtype=torch.float32)
        # raise SystemExit("Finished running all KTs, exiting before plotting.")

        fig, ax = plt.subplots(figsize=(6, 4))
        one_kT = 1 / torch.tensor(KTs, dtype=torch.float32)
        
        ax.scatter(one_kT.numpy(), lnK_NVP.numpy(), label='RealNVP', color='b')
        ax.scatter(one_kT.numpy(), lnK_Glow.numpy(), label='Glow', color='g')
        ax.scatter(one_kT.numpy(), lnK_LeapFrog.numpy(), label='LeapFrog', color='r')

        #calculate fit lines for each method
        coeffs_NVP = np.polyfit(one_kT.numpy(), lnK_NVP.numpy(), 1)
        coeffs_Glow = np.polyfit(one_kT.numpy(), lnK_Glow.numpy(), 1)
        coeffs_LeapFrog = np.polyfit(one_kT.numpy(), lnK_LeapFrog.numpy(), 1)
        fit_line_NVP = np.polyval(coeffs_NVP, one_kT.numpy())
        fit_line_Glow = np.polyval(coeffs_Glow, one_kT.numpy())
        fit_line_LeapFrog = np.polyval(coeffs_LeapFrog, one_kT.numpy())

        #draw fit lines for each method
        ax.plot(one_kT.numpy(), fit_line_NVP, label='RealNVP Fit: ' + f'{coeffs_NVP[0]:.2f}x + {coeffs_NVP[1]:.2f}', color='b', linestyle='--')
        ax.plot(one_kT.numpy(), fit_line_Glow, label='Glow Fit: ' + f'{coeffs_Glow[0]:.2f}x + {coeffs_Glow[1]:.2f}', color='g', linestyle='--')
        ax.plot(one_kT.numpy(), fit_line_LeapFrog, label='LeapFrog Fit: ' + f'{coeffs_LeapFrog[0]:.2f}x + {coeffs_LeapFrog[1]:.2f}', color='r', linestyle='--')

        #thoeretical results
        lnK_theory = -(MU[1] - MU[0]) * one_kT/2
        print("Theoretical lnK values:", lnK_theory.numpy())
        ax.plot(one_kT.numpy(), lnK_theory.numpy(), label='Theory', linestyle='-', color='k') 

        ax.set_xlabel('1/kT')
        ax.set_ylabel('ln(K)')
        ax.set_title('Van\'t Hoff Plot')

        #Properly order legend with scatter data then fit lines then theory
        handles, labels = ax.get_legend_handles_labels()
        scatter_handles = [h for h in handles if isinstance(h, plt.Line2D) and h.get_linestyle() == '']
        fit_handles = [h for h in handles if isinstance(h, plt.Line2D) and h.get_linestyle() == '--']
        theory_handles = [h for h in handles if isinstance(h, plt.Line2D) and h.get_linestyle() == '-']
        ax.legend(scatter_handles + fit_handles + theory_handles, labels=labels)

        fig.savefig(f'Fig3/vant_hoff_plot.png')
        #save to a dict
        vant_hoff_data = {
            '1/kT': one_kT.tolist(),
            'lnK_NVP': lnK_NVP.tolist(),
            'lnK_Glow': lnK_Glow.tolist(),
            'lnK_LeapFrog': lnK_LeapFrog.tolist(),
            'lnK_theory': lnK_theory.tolist()
        }
        #save as json
        import json
        with open('Fig3/vant_hoff_data.json', 'w') as f:
            json.dump(vant_hoff_data, f, indent=4)

    #can only do this if other is False
    if True:
        leapfrogdt = 0.5
        run_times = True
        #times plot
        #get times for NA = 8, 64, 216, 512
        NAs = [64, 216, 512] #512
        KT = 1

        makedata = True

        if makedata:
            times_NVP = []
            times_Glow = []
            times_LeapFrog = []
            for NA in NAs:
                    
                K_Glow, time_Glow = main(runtype='Glow')
                K_LeapFrog, time_LeapFrog = main(runtype='leapfrog')
                K_NVP, time_NVP = main(runtype='RealNVP')

                times_NVP.append(time_NVP)
                times_Glow.append(time_Glow)
                times_LeapFrog.append(time_LeapFrog)
        else: #load data from json
            with open('Fig3/time_data.json', 'r') as f:
                time_data = json.load(f)
            times_NVP = time_data['times_NVP']
            times_Glow = time_data['times_Glow']
            times_LeapFrog = time_data['times_LeapFrog']
        
        # Plot the times
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(NAs, times_NVP, label='RealNVP', marker='o')
        ax.plot(NAs, times_Glow, label='Glow', marker='o')
        ax.plot(NAs, times_LeapFrog, label='LeapFrog', marker='o')
        #log scale in y as it is so much slower for leapfrog
        ax.set_yscale('log')
        ax.set_xlabel('Number of Atoms (NA)')
        ax.set_ylabel('Time per Sample (s)')
        ax.set_title('Time per Sample vs Number of Atoms')
        ax.legend()
        fig.savefig(f'Fig3/time_per_sample.png')

        #save to a dict
        time_data = {
            'NAs': NAs,
            'times_NVP': times_NVP,
            'times_Glow': times_Glow,
            'times_LeapFrog': times_LeapFrog
        }
        #save as json
        with open('Fig3/time_data.json', 'w') as f:
            json.dump(time_data, f, indent=4)