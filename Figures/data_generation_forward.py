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
DIM = 2
LR = 1e-3
TRAIN_ITERS = 11
N_STEPS_FLOW = 4
EPOCHS = 150
HIDDEN_DIMS = [32,16,16,16]

def main(runtype = 'RealNVP', KT = 1.0, NA = 216):
    PDB_PATH = Path(f'examples/GaAs/GaAs{NA}.pdb')
    num_repeats = round((NA / 8)**(1/3), 0)  # Calculate the number of repeats needed to achieve NA atoms
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
                'p': Q(fix_kT(normal.sample((BATCH_SIZE, NA, DIM)), KT)),
                't': torch.tensor(0.0),
                }
            yield x

    def loss_KL(x0,x,logJ):
        
        assembled = assemble_neighbor_features(x['r'], neighborlists)
        energy = compute_energy_parameterized(assembled, MU, E1, E2, sigma=SIGMA).sum(1)
        
        assembled0 = assemble_neighbor_features(x0['r'], neighborlists)
        energy0 = compute_energy_parameterized(assembled0, MU, E1, E2, sigma=SIGMA).sum(1)

        if runtype == 'Glow':
            energy += 0.5 * x["p"].square().sum(dim=(1, 2))
            energy0 += 0.5 * x0["p"].square().sum(dim=(1, 2))


        loss = 1 / KT * (energy - energy0) - logJ
        return loss.mean(), x["r"].new_zeros(())


    def feature_extractor(x):
        B, N, D = x['r'].shape
        percents = torch.softmax(x['r'], dim=-1) # (B, N, D)
        percents_A = percents.mean(dim=(0,1))[0]

        if do_interactions:
            percents = torch.softmax(x['r'], dim=-1) # (B, N, D)
            percents_A_forinteractions = percents[:,:,0]
            #average over all batches after calculation interactions
            interactions = torch.zeros((B, 6), dtype=torch.float32)
            for i in range(B):
                percent_A = percents_A_forinteractions[i]
                interactions[i] = _compute_interactions(percent_A, first_pairs, second_pairs)
            if runtype == 'leapfrog':
                return [percents_A.detach().cpu(), interactions[:,1].mean().detach().cpu()], interactions.detach().cpu()
            else:
                return [percents_A.detach().cpu(), interactions[:,1].mean().detach().cpu()]
        else:
            return percents_A.detach().cpu()
    
    def graphing(compositions, output_dir,data_gen, losses = None):
        num_samples = 20
        # per_atom_energy = torch.zeros(num_samples * BATCH_SIZE, NA)
        # p_a_all = torch.zeros(num_samples * BATCH_SIZE, NA)

        if do_interactions:
            interactions = [comp[1] for comp in compositions]
            compositions = [comp[0] for comp in compositions]
        
        t0 = time.perf_counter()
        if losses is not None:
            flow.eval()
            with torch.inference_mode():
                p_a_all = torch.empty((num_samples * BATCH_SIZE, NA), dtype=torch.float32)
                for i in range(num_samples):
                    x = data_gen().__next__()
                    x, _, _ = flow(x)
                    p_a_all[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = torch.softmax(x['r'], dim=-1)[..., 0]
            p_A_overall = p_a_all.mean().item()
                    
        else: 
            last_compositions = torch.stack(compositions, dim=0).detach()[int(len(compositions)*4/5):] #(num_samples, D)
            p_A_overall = last_compositions.mean().item()
            #if interactions
            # x = data_gen().__next__()
            # num_samples = 1
            # for i in range(3000):
            #     # if (i+1) % 200 == 0:
            #     #     print(f"Generating sample {i+1}/{num_samples}")
            #     x, _, _ = flow(x)
            #     x = {k: v.detach().requires_grad_(True) for k, v in x_new.items()}
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
        fig3, axes3 = plt.subplots(1, 1, figsize=(6,4))
        #average compositions over batches
        compositions = torch.stack(compositions, dim=0).detach().numpy() # (EPOCHS, D) -> (EPOCHS,)
        color = 'tab:red'
        axes3.set_ylabel('Composition', color=color)

        if losses is not None:
            axes3.scatter(range(compositions.shape[0]), compositions, color=color, label = 'A Composition')
            axes3.set_xlabel('Batch')
            if do_interactions:
                interactions = torch.stack(interactions, dim=0).detach().numpy() # (EPOCHS, 6) -> (EPOCHS, 6)
                interactions = interactions/(NA*4)
                axes3.scatter(range(compositions.shape[0]), interactions, color='tab:blue', label='A-B Interactions')

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
            # axes4.set_ylim(-100,500)
            if do_interactions:
                lines_1, labels_1 = axes3.get_legend_handles_labels()
                lines_2, labels_2 = axes4.get_legend_handles_labels()

                # Add combined legend to the primary axis
                axes3.legend(lines_1 + lines_2, labels_1 + labels_2)
            else:
                axes4.legend()
        else:
            axes3.scatter((torch.arange(compositions.shape[0])+1)*10, compositions, color=color, label = 'A Composition')
            axes3.set_xlabel('Time step (dt)')
            if do_interactions:
                interactions = torch.stack(interactions, dim=0).detach().numpy() # (EPOCHS, 6) -> (EPOCHS, 6)
                interactions = interactions/(NA*4)
                # print(interactions)
                axes3.scatter((torch.arange(compositions.shape[0])+1)*10, interactions, color='tab:blue', label='A-B Interactions')

        axes3.set_ylim(0, 1)

        fig3.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(f'{output_dir}/ Composition_and_Loss_Analysis.png')
        plt.close()

        # #for leapfrog, plot interactions over steps
        # if losses is None:
        #     fig5, axes5 = plt.subplots(1, 1, figsize=(8, 5))
        #     # Plot interactions over steps
        #     interactions = torch.stack(interactions, dim=0).detach().numpy() # (EPOCHS, 6) -> (EPOCHS, 6)
        #     axes5.set_xlabel('Time step (dt)')
        #     axes5.set_ylabel('Interaction Count')
        #     axes5.set_title('Interactions vs Steps')
        #     # Plot each interaction type
        #     axes5.plot(interactions.mean(1)[:,0], label=f'1st neighbor A-A')
        #     axes5.plot(interactions.mean(1)[:,1], label=f'1st neighbor A-B')
        #     axes5.plot(interactions.mean(1)[:,2], label=f'1st neighbor B-B')
        #     axes5.plot(interactions.mean(1)[:,3], label=f'2nd neighbor A-A')
        #     axes5.plot(interactions.mean(1)[:,4], label=f'2nd neighbor A-B')
        #     axes5.plot(interactions.mean(1)[:,5], label=f'2nd neighbor B-B')
        #     axes5.legend()
        #     plt.savefig(f'{output_dir}/interactions_over_steps.png')

        #     #histogram of  neighbor interactions
        #     #overall interactions is from final 1/5 of interactions and flatten along B axis of (frame, B, N)
        #     overall_interactions = interactions[-(num_samples//5):].reshape(-1, 6)
        #     overall_interactions[:, 0] /= 864
        #     overall_interactions[:, 1] /= 864
        #     overall_interactions[:, 2] /= 864
        #     overall_interactions[:, 3] /= 2592
        #     overall_interactions[:, 4] /= 2592
        #     overall_interactions[:, 5] /= 2592
        #     plt.figure(figsize=(6, 4))
        #     bin_range = torch.arange(0, 1.01, 0.01)
        #     plt.hist(overall_interactions[:, 0], bins=bin_range, alpha=0.5, label='1st neighbor A-A')
        #     plt.hist(overall_interactions[:, 1], bins=bin_range, alpha=0.5, label='1st neighbor A-B')
        #     plt.hist(overall_interactions[:, 2], bins=bin_range, alpha=0.5, label='1st neighbor B-B')

        #     plt.hist(overall_interactions[:, 3], bins=bin_range, alpha=0.5, label='2nd neighbor A-A')
        #     plt.hist(overall_interactions[:, 4], bins=bin_range, alpha=0.5, label='2nd neighbor A-B')
        #     plt.hist(overall_interactions[:, 5], bins=bin_range, alpha=0.5, label='2nd neighbor B-B')
        #     plt.xlabel('Interaction Count')
        #     plt.ylabel('Frequency')
        #     plt.xlim(0, 1)
        #     plt.ylim(0, 3000//10)
        #     # plt.title(f'Histogram of Interactions for {nn_model.split("/")[-1].split(".")[0]} (Test Count: {test_count})')
        #     plt.legend()
                
        #     plt.savefig(f'{output_dir}/interactions_histogram.png', dpi=150)


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

        return K, (t1 - t0)/num_samples, 0
    
    print("Starting training...")
    if runtype == 'RealNVP':
        output_dir = f'Figures/RealNVP_NA{NA}_KT{KT}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        flow = RealNVP(DIM, NA, hidden_dims=HIDDEN_DIMS, n_layers=N_STEPS_FLOW, dt = 1/N_STEPS_FLOW)
        optimizer = optim.Adam(flow.parameters(), lr=LR)

        train_start = time.perf_counter()
        vals, losses_train = train_and_summarize(
            model=flow,
            loss_fn=loss_KL,
            data_generator=data_gen_NVP,
            optimizer=optimizer,
            epochs=EPOCHS,
            batches_per_epoch=TRAIN_ITERS - 1,
            feature_extractor=feature_extractor,
            inverse=False
        )
        train_end = time.perf_counter()
        losses = losses_train
        K, sample_time, _ = graphing(vals, output_dir, data_gen_NVP, losses)
        return K, sample_time, (train_end - train_start)
    elif runtype == 'Glow':
        output_dir = f'Figures/Glow_NA{NA}_KT{KT}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if do_interactions:
            def data_expansion(r):
                return assemble_neighbor_features(r, neighborlists).reshape(r.shape[0], r.shape[1], -1)
            
            glow = GlowBlock(dim=DIM, dt=1/N_STEPS_FLOW, hidden_dims=HIDDEN_DIMS, data_size = 17, data_expansion=data_expansion)
        else:
            glow = GlowBlock(dim=DIM, hidden_dims=HIDDEN_DIMS, dt = 1/N_STEPS_FLOW)
        
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
        output_dir = f'Figures/LeapFrog_NA{NA}_KT{KT}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        def U(r, t):
            return compute_energy_parameterized(assemble_neighbor_features(r, neighborlists), MU, E1, E2, sigma = SIGMA).sum(1)
        
        flow = LeapFrog(U, const_kT = KT, dt = leapfrogdt)
        vals = []
        x0 = next(data_gen())
        x = copy.deepcopy(x0)
        logJ = 0.0

        overall_interactions = torch.zeros((300*BATCH_SIZE, 6), dtype=torch.float32)
        
        t0 = time.perf_counter()
        for epoch in range(3000):
            x_new, lJ, info = flow(x)
            logJ += lJ.detach()
            x = {k: v.detach().requires_grad_(True) for k, v in x_new.items()}
            if (epoch+1) % 10 == 0:
                print(f"Leapfrog training epoch {epoch+1}/3000")
                
                if do_interactions:
                    feat, interactions = feature_extractor(x)
                    overall_interactions[(epoch//10)*BATCH_SIZE:((epoch//10)+1)*BATCH_SIZE] = interactions
                else:
                    feat = feature_extractor(x)
                vals.append(feat)
                
                #for the first time, record time
                if epoch == 9:
                    t2 = time.perf_counter()
        t1 = time.perf_counter()
        K, _, _ = graphing(vals, output_dir, data_gen)
        return K, (t2-t0), (t1 - t0), overall_interactions.mean(dim=0).detach().cpu()
    else:
        assert False, f"Unknown runtype: {runtype}"
    print("Training complete.")

if __name__ == "__main__":
    SIGMA = 5
    leapfrogdt = 0.5

    do_interactions = True

    if do_interactions:
        MU = torch.tensor([0,0], dtype=torch.float32)
        E1 = torch.tensor([[-0.3,-0.5],[-0.5,-0.1]], dtype=torch.float32)
        E2 = torch.tensor([[-0.05,-0],[-0,-0.05]], dtype=torch.float32)
    else:
        MU = torch.tensor([2,3], dtype=torch.float32)
        E1 = torch.tensor([[0,0],[0,0]], dtype=torch.float32)
        E2 = torch.tensor([[0,0],[0,0]], dtype=torch.float32)

    do_temperatures_plot = True
    temperatures_config = {
        'NA': 216,
        'kTs': [0.25, 0.5, 1.0, 2.0, 4.0],
        'makedata': True
    }

    do_times_plot = False
    times_config = {
        'NAs': [64, 216, 512],
        'kT': 1.0,
        'makedata': True
    }

    if do_temperatures_plot:
        kTs = temperatures_config['kTs']
        NA = temperatures_config['NA']
        makedata = temperatures_config['makedata']
        Ks_NVP = []
        Ks_Glow = []
        Ks_LeapFrog = []
        LeapFrog_interactions = []
        
        if makedata:
            for i, KT in enumerate(kTs):
                print(f"Running simulations for kT={KT} ({i+1}/{len(kTs)})")
                # K_LeapFrog, samples_time_LeapFrog, train_time_LeapFrog, interactions = main(runtype='leapfrog', KT=KT, NA=NA)
                K_Glow, samples_time_Glow, train_time_Glow = main(runtype='Glow', KT=KT, NA=NA)
                # K_NVP, samples_time_NVP, train_time_NVP = main(runtype='RealNVP', KT=KT, NA=NA)

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
        # fig.savefig(f'Figures/temperatures/vant_hoff_plot.png')
        # #save to a dict
        # vant_hoff_data = {
        #     '1/kT': one_kT.tolist(),
        #     'lnK_NVP': lnK_NVP.tolist(),
        #     'lnK_Glow': lnK_Glow.tolist(),
        #     'lnK_LeapFrog': lnK_LeapFrog.tolist(),
        #     'lnK_theory': lnK_theory.tolist()
        # }
        # #save as json
        # with open('Figures/temperatures/vant_hoff_data.json', 'w') as f:
        #     json.dump(vant_hoff_data, f, indent=4)

        # data = {
        #     'KTs': kTs,
        #     'interactions': [interactions.tolist() for interactions in LeapFrog_interactions],
        # }
        # with open(f'Figures/LeapFrog_interactions_vs_temperature.json', 'w') as f:
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
            with open('Figures/times/time_data.json', 'r') as f:
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
        fig.savefig(f'Figures/times/time_per_sample.png')

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
        with open('Figures/times/time_data.json', 'w') as f:
            json.dump(time_data, f, indent=4)