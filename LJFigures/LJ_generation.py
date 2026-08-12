import torch, math, copy, time, json
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from alchemist.flows import GlowBlock, MultiStep, Q, RealNVP, LeapFrog, fix_kT
from alchemist.pdbs import read_pdb_coords, write_pdb_trajectory
from alchemist.ml_utils import train_and_summarize
from alchemist.analysis import compute_energy_parameterized, compute_energy_LennardJones
# --- Configuration ---
CUTS = [2.5, 4.5]
BATCH_SIZE = 50
DIM = 1
LR = 1e-3
TRAIN_ITERS = 11
N_STEPS_FLOW = 4
EPOCHS = 25
HIDDEN_DIMS = [32,16,16,16]
folder = 'LJFigures'

def main(runtype = 'RealNVP', KT = 1.0, NA = 216, rho = 1.091531):
    PDB_PATH = Path(f'examples/HCPs/HCP{NA}.pdb')
    # 2. Read PDB
    coords = read_pdb_coords(PDB_PATH)
    
    # 3. Training Utilities
    normal = torch.distributions.normal.Normal(0, 1)

    a=1.09016685
    c_over_a=np.sqrt(8.0 / 3.0)
    nx = round((NA / 8)**(1/3), 0) * 2
    BOX = torch.tensor([a, 3.0**0.5 * a, c_over_a * a], dtype=torch.float32)*torch.tensor([nx, nx/2, nx/2],dtype=torch.float32)

    #rescale box for proper density
    volume = BOX.prod()
    desired_volume = NA / rho
    scale_factor = (desired_volume / volume)**(1/3)
    BOX *= scale_factor

    def data_gen():
        # 1 to front of all coords to be single energy identity
        r_chem = torch.ones((BATCH_SIZE, NA, DIM), dtype=torch.float32)
        r_coord = torch.tensor(coords, dtype=torch.float32).repeat(BATCH_SIZE, 1, 1)
        r = torch.cat([r_chem, r_coord], dim=2)  # (B, N, D)
        p_chem = torch.zeros((BATCH_SIZE, NA, DIM), dtype=torch.float32)
        while True:
            x = {
                'r': r,
                'p': torch.cat([p_chem, fix_kT(normal.sample((BATCH_SIZE, NA, 3)), KT)], dim=2),
                't': torch.tensor(0.0),
                }
            yield x

    def loss_KL(x0,x,logJ):
        # Extract positions from the input
        energy = compute_energy_LennardJones(x["r"][:,:,1:], epsilon_LJ=EPSILON_LJ, sigma_LJ=SIGMA_LJ, box=BOX, cutoff=CUTOFF).sum(1)
        
        energy0 = compute_energy_LennardJones(x0["r"][:,:,1:], epsilon_LJ=EPSILON_LJ, sigma_LJ=SIGMA_LJ, box=BOX, cutoff=CUTOFF).sum(1)

        loss = 1 / KT * (energy - energy0) - logJ
        return loss.mean()

    def feature_extractor(x):
        #calculate average closest neighbor distance
        positions = x["r"][:,:,1:] # (B, N, 3)
        # Compute pairwise distances
        delta = positions[:, :, None, :] - positions[:, None, :, :]
        delta = torch.remainder(delta + BOX / 2, BOX) - BOX / 2
        dists = torch.linalg.vector_norm(delta, dim=-1)

        self_mask = torch.eye(NA, dtype=torch.bool).unsqueeze(0)

        dists = dists.masked_fill(self_mask, torch.inf)
        # Find the minimum distance for each particle
        min_dists = dists.min(dim=-1).values
        # Return the average of the minimum distances
        # and return only 1 batch of x['r'] for graphing
        return [min_dists.detach().mean().item(), x['r'][0].unsqueeze(0).detach().cpu().numpy()]  # (1, N, DIM+3)
    
    def graphing(features, output_dir, data_gen, losses = None):
        num_samples = 20

        min_dists = [feat[0] for feat in features]
        rs = torch.cat([torch.tensor(feat[1]) for feat in features], dim=0)  # (num_samples * BATCH_SIZE, NA, DIM+3)
        
        t0 = time.perf_counter()
        pair_i, pair_j = torch.triu_indices(NA,NA,offset=1)
        if losses is not None:
            dists = torch.empty((num_samples,BATCH_SIZE, NA * (NA - 1) // 2), dtype=torch.float32)
            flow.eval()
            with torch.inference_mode():
                for i in range(num_samples):
                    x = data_gen().__next__()
                    x, _, _ = flow(x)
                    xyz = x["r"][..., -3:]
                    delta = xyz[:, pair_i, :] - xyz[:, pair_j, :]
                    delta = torch.remainder(delta + BOX / 2, BOX) - BOX / 2
                    dists[i] = torch.linalg.vector_norm(delta, dim=-1)
        else:
            sample_features = rs[torch.linspace(len(rs)*2/3, len(rs)-1, num_samples, dtype=torch.int64)]
            xyz = sample_features[..., -3:]
            delta = xyz[:, pair_i, :] - xyz[:, pair_j, :]
            delta = torch.remainder(delta + BOX / 2, BOX) - BOX / 2
            dists = torch.linalg.vector_norm(delta, dim=-1)
                    
        t1 = time.perf_counter()

        # Plot 3: Loss and avg first neighbor distance double plot
        fig3, axes3 = plt.subplots(1, 1, figsize=(6,4))
        min_dists = torch.tensor(min_dists).detach().numpy()
        color = 'tab:red'
        axes3.set_ylabel('Average First Neighbor Distance', color=color)

        if losses is not None:
            axes3.scatter(range(min_dists.shape[0]), min_dists, color=color, label = 'A Composition')
            axes3.set_xlabel('Batch')

            losses = torch.tensor(losses).detach().numpy() # (EPOCHS, 2) -> (EPOCHS, 2)
            all_losses = losses[:, 0]
            lJ_losses = -losses[:, 1]
            U_losses = all_losses - lJ_losses

            axes4 = axes3.twinx()  # instantiate a second Axes that shares the same x-axis
            color = 'tab:blue'
            axes4.set_ylabel('Loss', color=color)
            axes4.plot(all_losses, color=color, label='Total Loss')
            axes4.plot(lJ_losses, color='tab:orange', label='-logJ')
            axes4.plot(U_losses, color='tab:green', label=r"$\Delta U$/kT")
            axes4.tick_params(axis='y', labelcolor=color)
            axes4.set_ylim(-1000,2000)
            axes4.legend()
        else:
            axes3.scatter((torch.arange(min_dists.shape[0])+1)*10, min_dists, color=color, label = 'A Composition')
            axes3.set_xlabel('Time step (dt)')

        fig3.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(f'{output_dir}/ NeighborDist_and_Loss_Analysis.png')
        plt.close()

        #Plot g(r) for the samples
        fig, ax = plt.subplots(figsize=(6, 4))
        r_max = 0.5 * BOX.min().item()
        n_bins = 150

        edges = torch.linspace(0.0,r_max,n_bins + 1)

        counts = torch.histogram(dists.reshape(-1),bins=edges)[0]

        r_inner = edges[:-1]
        r_outer = edges[1:]
        r_centers = 0.5 * (r_inner + r_outer)

        shell_volumes = (4.0* math.pi/ 3.0* (r_outer**3 - r_inner**3))

        n_frames = dists.shape[0]
        n_pairs = NA * (NA - 1) / 2
        volume = BOX.prod()

        expected_counts = (n_frames* n_pairs* shell_volumes/ volume)
        if losses is not None:
            expected_counts = (n_frames* n_pairs* shell_volumes* BATCH_SIZE/ volume)

        g_r = counts / expected_counts

        ax.plot(r_centers.detach().cpu(),g_r.detach().cpu(),)

        ax.axhline(1.0, linestyle="--")
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$g(r)$")
        ax.set_ylim(bottom=0.0)
        fig.tight_layout()
        plt.savefig(f'{output_dir}/g_r_plot.png')
        plt.close()


        write_pdb_trajectory(Path(f'{output_dir}/generated_samples.pdb'), rs)

        # 7. Save model
        if losses is not None:
            torch.save(flow.state_dict(), f'{output_dir}/trained_flow_model.pth')
        
        return 0, (t1 - t0)/num_samples, 0
    
    print("Starting training...")
    if runtype == 'RealNVP':
        output_dir = f'{folder}/RealNVP_NA{NA}_KT{KT}_rho{rho}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        flow = RealNVP(DIM, NA, hidden_dims=HIDDEN_DIMS, n_layers=N_STEPS_FLOW, dt = 1/N_STEPS_FLOW,do_cartesian=True, box=BOX)
        optimizer = optim.Adam(flow.parameters(), lr=LR)

        train_start = time.perf_counter()
        vals, losses_train = train_and_summarize(
            model=flow,
            loss_fn=loss_KL,
            data_generator=data_gen,
            optimizer=optimizer,
            epochs=EPOCHS,
            batches_per_epoch=TRAIN_ITERS - 1,
            feature_extractor=feature_extractor,
            inverse=False
        )
        train_end = time.perf_counter()
        losses = losses_train
        K, sample_time, _ = graphing(vals, output_dir, data_gen, losses)
        return K, sample_time, (train_end - train_start)
    elif runtype == 'Glow':
        output_dir = f'{folder}/Glow_NA{NA}_KT{KT}_rho{rho}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        def data_expansion(r: torch.Tensor, k: int = 8) -> torch.Tensor:
            xyz = r[..., 1:]  # (B, N, 3)

            if xyz.ndim != 3 or xyz.shape[-1] != 3:
                raise ValueError(
                    "Expected r with shape (B, N, 4): [feature, x, y, z]."
                )

            B, N, _ = xyz.shape

            if not 1 <= k < N:
                raise ValueError(f"k must satisfy 1 <= k < N; got k={k}, N={N}.")

            # Pairwise distances: (B, N, N)
            delta = xyz[:, :, None, :] - xyz[:, None, :, :]
            delta = torch.remainder(delta + BOX / 2, BOX) - BOX / 2
            dists = torch.linalg.vector_norm(delta, dim=-1)

            self_mask = torch.eye(N, dtype=torch.bool, device=xyz.device).unsqueeze(0)

            dists = dists.masked_fill(self_mask, torch.inf)

            # Neighbor indices: (B, N, k)
            idx = dists.topk(k=k,dim=-1,largest=False, sorted=True).indices

            # Select xyz[b, idx[b, i, j], :]
            batch = torch.arange(B,device=xyz.device).view(B, 1, 1)

            neighbors = xyz[batch, idx]  # (B, N, k, 3)

            XYZ_diffs = xyz.unsqueeze(2) - neighbors  # (B, N, k, 3)

            return XYZ_diffs.reshape(B, N, k * 3)  # (B, N, k*3)
        
        glow = GlowBlock(dim=DIM, dt=1/N_STEPS_FLOW, hidden_dims=HIDDEN_DIMS, data_size = 24, data_expansion=data_expansion, box=BOX)
        
        flow = MultiStep(glow, N_STEPS_FLOW)
        optimizer = optim.Adam(flow.parameters(), lr=LR)

        train_start = time.perf_counter()
        vals, losses_train = train_and_summarize(
            model=flow,
            loss_fn=loss_KL,
            data_generator=data_gen,
            optimizer=optimizer,
            epochs=EPOCHS,
            batches_per_epoch=TRAIN_ITERS - 1,
            feature_extractor=feature_extractor,
            inverse=False
        )
        train_end = time.perf_counter()
        losses = losses_train
        K, sample_time, _ = graphing(vals, output_dir, data_gen, losses)
        return K, sample_time, (train_end - train_start)
    elif runtype == 'leapfrog':
        output_dir = f'{folder}/LeapFrog_NA{NA}_KT{KT}_rho{rho}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        def U(r, t):
            positions = r[:,:,1:] # (B, N, 3)
            return compute_energy_LennardJones(positions, epsilon_LJ=EPSILON_LJ, sigma_LJ=SIGMA_LJ, box=BOX, cutoff=CUTOFF).sum(1)
        
        flow = LeapFrog(U, const_kT = KT, dt = leapfrogdt, box = BOX)
        vals = []
        x0 = next(data_gen())
        x = copy.deepcopy(x0)
        logJ = 0.0

        t0 = time.perf_counter()
        for epoch in range(1000):
            x_new, lJ, info = flow(x)
            logJ += lJ.detach()
            x = {k: v.detach().requires_grad_(True) for k, v in x_new.items()}
            if (epoch+1) % 10 == 0:
                if (epoch+1) % 200 == 0:
                    print(f"Leapfrog training epoch {epoch+1}/1000")
                
                feat = feature_extractor(x)

                vals.append(feat)
                
                #for the first time, record time
                if epoch == 9:
                    t2 = time.perf_counter()
        t1 = time.perf_counter()
        K, _, _ = graphing(vals, output_dir, data_gen)
        return K, (t2-t0), (t1 - t0)
    else:
        assert False, f"Unknown runtype: {runtype}"
    print("Training complete.")

if __name__ == "__main__":
    leapfrogdt = 0.01

    EPSILON_LJ = 1.0
    SIGMA_LJ = 1.0
    CUTOFF = 2.5

    do_densities_plot = True
    densities_config = {
        'NA':216,
        'kTs':[0.25, 0.5, 1.5],
        'rhos':[0.5, 0.75, 1.0, 1.3],
        'makedata': True
    }

    do_temperatures_plot = False
    temperatures_config = {
        'NA': 216,
        'kTs': [0.15,0.30,0.5,0.75,1.0],
        'rho':1.091531,
        'makedata': True
    }

    do_times_plot = False
    times_config = {
        'NAs': [64, 216, 512],
        'kT': 1.0,
        'rho': 1.091531,
        'makedata': True
    }

    if do_densities_plot:
        KTs = densities_config['kTs']
        rhos = densities_config['rhos']
        NA = densities_config['NA']
        makedata = densities_config['makedata']

        if makedata:
            for i, rho in enumerate(rhos):
                for j, KT in enumerate(KTs):
                    print(f"Running simulations for rho={rho}, kT={KT} ({i+1}/{len(rhos)}, {j+1}/{len(KTs)})")
                    K_LeapFrog, samples_time_LeapFrog, train_time_LeapFrog = main(runtype='leapfrog', KT=KT, NA=NA, rho=rho)
                    K_Glow, samples_time_Glow, train_time_Glow = main(runtype='Glow', KT=KT, NA=NA, rho=rho)
                    K_NVP, samples_time_NVP, train_time_NVP = main(runtype='RealNVP', KT=KT, NA=NA, rho=rho)


    if do_temperatures_plot:
        kTs = temperatures_config['kTs']
        NA = temperatures_config['NA']
        makedata = temperatures_config['makedata']
        Ks_NVP = []
        Ks_Glow = []
        Ks_LeapFrog = []
        
        if makedata:
            for i, KT in enumerate(kTs):
                print(f"Running simulations for kT={KT} ({i+1}/{len(kTs)})")
                K_LeapFrog, samples_time_LeapFrog, train_time_LeapFrog = main(runtype='leapfrog', KT=KT, NA=NA)
                K_Glow, samples_time_Glow, train_time_Glow = main(runtype='Glow', KT=KT, NA=NA)
                K_NVP, samples_time_NVP, train_time_NVP = main(runtype='RealNVP', KT=KT, NA=NA)

        #         Ks_NVP.append(K_NVP)
        #         Ks_Glow.append(K_Glow)
        #         Ks_LeapFrog.append(K_LeapFrog)
        #         interactions[0] /= 864
        #         interactions[1] /= 864
        #         interactions[2] /= 864
        #         interactions[3] /= 2592
        #         interactions[4] /= 2592
        #         interactions[5] /= 2592
        #         LeapFrog_interactions.append(interactions)

        #     #simulation results
        #     lnK_NVP = torch.log(torch.tensor(Ks_NVP, dtype=torch.float32))
        #     lnK_Glow = torch.log(torch.tensor(Ks_Glow, dtype=torch.float32))
        #     lnK_LeapFrog = torch.log(torch.tensor(Ks_LeapFrog, dtype=torch.float32))
        # else: #load data from json
        #     with open('Figures/temperatures_data.json', 'r') as f:
        #         vant_hoff_data = json.load(f)
        #     lnK_NVP = torch.tensor(vant_hoff_data['lnK_NVP'], dtype=torch.float32)
        #     lnK_Glow = torch.tensor(vant_hoff_data['lnK_Glow'], dtype=torch.float32)
        #     lnK_LeapFrog = torch.tensor(vant_hoff_data['lnK_LeapFrog'], dtype=torch.float32)
        # # raise SystemExit("Finished running all KTs, exiting before plotting.")

        # fig, ax = plt.subplots(figsize=(6, 4))
        # one_kT = 1 / torch.tensor(kTs, dtype=torch.float32)
        
        # ax.scatter(one_kT.numpy(), lnK_NVP.numpy(), label='RealNVP', color='b')
        # ax.scatter(one_kT.numpy(), lnK_Glow.numpy(), label='Glow', color='g')
        # ax.scatter(one_kT.numpy(), lnK_LeapFrog.numpy(), label='LeapFrog', color='r')

        # #calculate fit lines for each method
        # coeffs_NVP = np.polyfit(one_kT.numpy(), lnK_NVP.numpy(), 1)
        # coeffs_Glow = np.polyfit(one_kT.numpy(), lnK_Glow.numpy(), 1)
        # coeffs_LeapFrog = np.polyfit(one_kT.numpy(), lnK_LeapFrog.numpy(), 1)
        # fit_line_NVP = np.polyval(coeffs_NVP, one_kT.numpy())
        # fit_line_Glow = np.polyval(coeffs_Glow, one_kT.numpy())
        # fit_line_LeapFrog = np.polyval(coeffs_LeapFrog, one_kT.numpy())

        # #draw fit lines for each method
        # # ax.plot(one_kT.numpy(), fit_line_NVP, label='RealNVP Fit: ' + f'{coeffs_NVP[0]:.2f}x + {coeffs_NVP[1]:.2f}', color='b', linestyle='--')
        # # ax.plot(one_kT.numpy(), fit_line_Glow, label='Glow Fit: ' + f'{coeffs_Glow[0]:.2f}x + {coeffs_Glow[1]:.2f}', color='g', linestyle='--')
        # # ax.plot(one_kT.numpy(), fit_line_LeapFrog, label='LeapFrog Fit: ' + f'{coeffs_LeapFrog[0]:.2f}x + {coeffs_LeapFrog[1]:.2f}', color='r', linestyle='--')

        # #thoeretical results
        # lnK_theory = -(MU[1] - MU[0]) * one_kT
        # print("Theoretical lnK values:", lnK_theory.numpy())
        # ax.plot(one_kT.numpy(), lnK_theory.numpy(), label='Theory', linestyle='-', color='k') 

        # ax.set_xlabel('1/kT')
        # ax.set_ylabel('ln(K)')
        # ax.set_title('Van\'t Hoff Plot')

        # #Properly order legend with scatter data then fit lines then theory
        # handles, labels = ax.get_legend_handles_labels()
        # scatter_handles = [h for h in handles if isinstance(h, plt.Line2D) and h.get_linestyle() == '']
        # fit_handles = [h for h in handles if isinstance(h, plt.Line2D) and h.get_linestyle() == '--']
        # theory_handles = [h for h in handles if isinstance(h, plt.Line2D) and h.get_linestyle() == '-']
        # ax.legend(scatter_handles + fit_handles + theory_handles, labels=labels)

        # fig.tight_layout()
        # fig.savefig(f'{folder}/temperatures/vant_hoff_plot.png')
        # #save to a dict
        # vant_hoff_data = {
        #     '1/kT': one_kT.tolist(),
        #     'lnK_NVP': lnK_NVP.tolist(),
        #     'lnK_Glow': lnK_Glow.tolist(),
        #     'lnK_LeapFrog': lnK_LeapFrog.tolist(),
        #     'lnK_theory': lnK_theory.tolist()
        # }
        # #save as json
        # with open(f'{folder}/temperatures/vant_hoff_data.json', 'w') as f:
        #     json.dump(vant_hoff_data, f, indent=4)

        # data = {
        #     'KTs': kTs,
        #     'interactions': [interactions.tolist() for interactions in LeapFrog_interactions],
        # }
        # with open(f'{folder}/LeapFrog_interactions_vs_temperature.json', 'w') as f:
        #     json.dump(data, f, indent=4)

    if do_times_plot:
        NAs = times_config['NAs']
        KT = times_config['kT']
        makedata = times_config['makedata']

        if makedata:
            samples_times_NVP = []
            samples_times_Glow = []
            samples_times_LeapFrog = []
            train_times_NVP = []
            train_times_Glow = []
            train_times_LeapFrog = []
            for NA in NAs:
                    
                K_Glow, samples_time_Glow, train_time_Glow = main(runtype='Glow', KT=KT, NA=NA)
                K_LeapFrog, samples_time_LeapFrog, train_time_LeapFrog, interactions = main(runtype='leapfrog', KT=KT, NA=NA)
                K_NVP, samples_time_NVP, train_time_NVP = main(runtype='RealNVP', KT=KT, NA=NA)

                samples_times_NVP.append(samples_time_NVP)
                samples_times_Glow.append(samples_time_Glow)
                samples_times_LeapFrog.append(samples_time_LeapFrog)
                train_times_NVP.append(train_time_NVP)
                train_times_Glow.append(train_time_Glow)
                train_times_LeapFrog.append(train_time_LeapFrog)
        else: #load data from json
            with open(f'{folder}/times/time_data.json', 'r') as f:
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
        fig.savefig(f'{folder}/times/time_per_sample.png')

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
        with open(f'{folder}/times/time_data.json', 'w') as f:
            json.dump(time_data, f, indent=4)