import torch, glob, json
import time
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from alchemist.flows import GlowBlock, MultiStep, Q, RealNVP, fix_kT
from alchemist.pdbs import read_pdb_coords, write_pdb_trajectory
from alchemist.neighbors import compute_neighbor_masks, get_neighbor_indices, assemble_neighbor_features
from alchemist.ml_utils import train_and_summarize
from alchemist.analysis import compute_neighbor_histograms, compute_energy_parameterized


import matplotlib.pyplot as plt


def _build_neighbor_pairs(neighborlist):
    src_idx = []
    dst_idx = []
    for atom_idx, nbrs in enumerate(neighborlist):
        if len(nbrs) == 0:
            continue
        src_idx.extend([atom_idx] * len(nbrs))
        dst_idx.extend(int(nbr_idx) for nbr_idx in nbrs)
    return torch.tensor(src_idx, dtype=torch.long), torch.tensor(dst_idx, dtype=torch.long)

def _compute_interactions(last_ga_frame, first_pairs, second_pairs):
    def _pair_interactions(src_idx, dst_idx):
        src = last_ga_frame[src_idx]
        dst = last_ga_frame[dst_idx]
        gg = (src * dst).sum()
        ga_as = (src * (1 - dst) + (1 - src) * dst).sum()
        as_as = ((1 - src) * (1 - dst)).sum()
        return torch.stack([gg, ga_as, as_as])

    first = _pair_interactions(*first_pairs)
    second = _pair_interactions(*second_pairs)
    return torch.cat([first, second])

if __name__ == "__main__":
    CUTS = [2.5, 4.5]
    test_count = 1000
    DIM = 2
    N_STEPS_FLOW = 4
    SIGMA = 5
    scale = test_count // 2
    folder = 'Fig7'
    run_type = 'RealNVP'
    NA = 216
    average_interactions = []
    KTs = [0.25, 0.5, 1, 2, 4]
    for KT in KTs:
        nn_model = glob.glob(f'{folder}/{run_type}_NA{NA}_KT{KT}*/trained_flow_model.pth')[0]
        print(f"Testing model: {nn_model}")
        filename = folder + '/' + nn_model.split('/')[1] + '/'
        networkdims = [8,8,8]
        # NA = int(nn_model.split('/')[1].split('_')[1])
        PDB_PATH = Path(f'examples/GaAs/GaAs{NA}.pdb')
        num_repeats = round((NA / 8)**(1/3),0)
        BOX = torch.full((3,), 5.75 * num_repeats)
        coords = read_pdb_coords(PDB_PATH)
        masks = compute_neighbor_masks(coords, BOX, CUTS)
        neighborlists = get_neighbor_indices(masks)
        first_pairs = _build_neighbor_pairs(neighborlists[0])
        second_pairs = _build_neighbor_pairs(neighborlists[1])

        def data_expansion(r):
            return assemble_neighbor_features(r, neighborlists).reshape(r.shape[0], r.shape[1], -1)
        
        if run_type == 'RealNVP':
            flow = RealNVP(DIM, NA, hidden_dims=networkdims, n_layers=N_STEPS_FLOW, dt = 1/N_STEPS_FLOW*2)
        elif run_type == 'Glow':
            glow = GlowBlock(dim=DIM, dt=0.001, hidden_dims=networkdims, data_size = 17, data_expansion=data_expansion)
            flow = MultiStep(glow, N_STEPS_FLOW, dt = 1/N_STEPS_FLOW)
        weights = torch.load(nn_model)
        flow.load_state_dict(weights)
        flow.eval()

        overall_interactions = torch.empty((test_count, 6), dtype=torch.float32)
        overall_energies = torch.empty((test_count,), dtype=torch.float32)
        last_Ga_composition = torch.empty((test_count,), dtype=torch.float32)
        sample_time_s = 0.0
        interaction_time_s = 0.0

        if run_type == 'RealNVP':
            r_buf = torch.empty((1, NA, DIM))
            r_coord_buf = torch.empty((1, NA, DIM))
            t_buf = torch.zeros((1,), dtype=torch.float32)
        elif run_type == 'Glow':
            r_buf = torch.empty((1, NA, DIM))
            p_buf = torch.empty((1, NA, DIM))
            t_buf = torch.zeros((1,), dtype=torch.float32)

        with torch.no_grad():
            for i in range(test_count):
                if (i+1) % 100 == 0:
                    print(f"Generating sample {i+1}/{test_count}")
                
                t0 = time.perf_counter()
                if run_type == 'RealNVP':
                    torch.randn(r_buf.shape, out=r_buf)
                    x = {'r': Q(r_buf) * SIGMA,
                        'r_coord': torch.tensor(coords, dtype=torch.float32).repeat(1, 1, 1),
                        't': t_buf}
                elif run_type == 'Glow':
                    torch.randn(r_buf.shape, out=r_buf)
                    torch.randn(p_buf.shape, out=p_buf)
                    x = {'r': Q(r_buf) * SIGMA,
                        'p': Q(fix_kT(p_buf, KT)),
                        't': t_buf}
                
                x, _, _ = flow(x)

                feat = assemble_neighbor_features(x['r'], neighborlists)
                p_a_all = torch.softmax(feat, dim=-1)[..., 0, 0]

                t1 = time.perf_counter()
                last_Ga_frame = p_a_all[-1]
                last_Ga_composition[i] = last_Ga_frame.mean().item()
                interactions = _compute_interactions(last_Ga_frame, first_pairs, second_pairs)
                t2 = time.perf_counter()
                overall_interactions[i] = interactions

                energy = interactions[0] * -0.1 + interactions[1] * -0.5 + interactions[2] * -0.1 + \
                        interactions[3] * -0.05 + interactions[4] * 0.0 + interactions[5] * -0.05

                # energy = interactions[0] * 0.3 + interactions[1] * 0.1 + interactions[2] * 0.5
                t3 = time.perf_counter()
                overall_energies[i] = energy

                sample_time_s += t1 - t0
                interaction_time_s += t2 - t1
                energy_time_s = t3 - t2

        print(
            f"Timing for {nn_model.split('/')[-1]}: "
            f"sampling={sample_time_s:.3f}s, interactions={interaction_time_s:.3f}s"
            f"sampling_per_sample={sample_time_s / test_count:.6f}s, "
            f"interactions_per_sample={interaction_time_s / test_count:.6f}s"
        )

        #Rescale overall interactions to % of total possible interactions
        overall_interactions[:, 0] /= 864
        overall_interactions[:, 1] /= 864
        overall_interactions[:, 2] /= 864
        overall_interactions[:, 3] /= 2592
        overall_interactions[:, 4] /= 2592
        overall_interactions[:, 5] /= 2592
        bin_range = torch.arange(0, 1.01, 0.01)


        #histogram of neighbor interactions
        plt.figure(figsize=(6, 4))
        plt.hist(overall_interactions[:, 0].numpy(), bins=bin_range, alpha=0.5, label='1st neighbor A-A')
        plt.hist(overall_interactions[:, 1].numpy(), bins=bin_range, alpha=0.5, label='1st neighbor A-B')
        plt.hist(overall_interactions[:, 2].numpy(), bins=bin_range, alpha=0.5, label='1st neighbor B-B')

        plt.hist(overall_interactions[:, 3].numpy(), bins=bin_range, alpha=0.5, label='2nd neighbor A-A')
        plt.hist(overall_interactions[:, 4].numpy(), bins=bin_range, alpha=0.5, label='2nd neighbor A-B')
        plt.hist(overall_interactions[:, 5].numpy(), bins=bin_range, alpha=0.5, label='2nd neighbor B-B')
        plt.xlabel('Interaction Count')
        plt.ylabel('Frequency')
        plt.xlim(0, 1)
        plt.ylim(0, scale)
        # plt.title(f'Histogram of Interactions for {nn_model.split("/")[-1].split(".")[0]} (Test Count: {test_count})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{filename}interactions_histogram.png', dpi=150)

        # bin_width = 5
        # #histogram of energies
        # plt.figure(figsize=(10, 6))
        # plt.hist(overall_energies.numpy(), bins=torch.arange(min(overall_energies.numpy()), max(overall_energies.numpy()) + bin_width, bin_width), alpha=0.5, label='Potential Energy')
        # plt.xlabel('Potential Energy')
        # plt.ylabel('Frequency')
        # plt.xlim(-600, -200)
        # plt.ylim(0, scale)
        # plt.title(f'Histogram of Potential Energies for {nn_model.split("/")[-1].split(".")[0]} (Test Count: {test_count})')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f'{filename}potential_energy_histogram.png', dpi=150)

        # #histogram of Ga compositions
        # bin_width = 0.01
        # plt.figure(figsize=(10, 6))
        # plt.hist(last_Ga_composition.numpy(), bins=torch.arange(min(last_Ga_composition.numpy()), max(last_Ga_composition.numpy()) + bin_width, bin_width), alpha=0.5, label='Ga Composition')
        # plt.xlabel('Ga Composition')
        # plt.ylabel('Frequency')
        # plt.xlim(0,1)
        # plt.ylim(0, scale)
        # plt.title(f'Histogram of Ga Compositions for {nn_model.split("/")[-1].split(".")[0]} (Test Count: {test_count})')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f'{filename}ga_composition_histogram.png', dpi=150)

        save_dict = {
            'overall_interactions': overall_interactions,
            'overall_energies': overall_energies,
            'last_Ga_composition': last_Ga_composition,
        }

        # Save the dictionary
        torch.save(save_dict, f'{filename}test_results.pt')
        # print(overall_interactions.shape)
        average_interactions.append(overall_interactions.mean(dim=0).numpy())
    
    plt.figure(figsize=(6, 4))
    #there are 6 interaction types, so we will plot the average of each type across all KTs
    average_interactions = torch.tensor(average_interactions)
    plt.plot(KTs, average_interactions[:, 0], label='1st neighbor A-A')
    plt.plot(KTs, average_interactions[:, 1], label='1st neighbor A-B')
    plt.plot(KTs, average_interactions[:, 2], label='1st neighbor B-B')
    plt.plot(KTs, average_interactions[:, 3], label='2nd neighbor A-A')
    plt.plot(KTs, average_interactions[:, 4], label='2nd neighbor A-B')
    plt.plot(KTs, average_interactions[:, 5], label='2nd neighbor B-B')
    plt.xlabel('kT')
    plt.ylim(0, 1)
    plt.ylabel('Average Interaction Count')
    # plt.title('Interactions vs Temperature')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{folder}/{run_type}_interactions_vs_temperature.png', dpi=150)
    
    #save data
    data = {
        'KTs': KTs,
        'interactions': average_interactions.numpy().tolist(),
    }
    with open(f'Fig7/{run_type}_interactions_vs_temperature.json', 'w') as f:
        json.dump(data, f, indent=4)