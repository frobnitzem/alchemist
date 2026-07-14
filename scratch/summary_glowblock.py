import torch, glob
import time
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from alchemistlib.flows import GlowBlock, MultiStep, Q, fix_kT
from alchemistlib.modules import FNN

import matplotlib.pyplot as plt

from utils import _read_pdb_coords, build_crystal_U
from glowblock import percentA
from traintest import test_glowblock_two_part


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

def generate_sample(batch_size, Na, dim, sigma, kT=1.0):
    normal = torch.distributions.normal.Normal(0, 1)
    x = {
        'r': Q(normal.sample((batch_size, Na, dim))) * sigma,
        'p': Q(fix_kT(normal.sample((batch_size, Na, dim)), kT)),
        't': 0.0
    }
    percent = percentA(x['r'])
    x['p'] = torch.stack([percent, 1 - percent], dim=2)
    return x


# ----------------------------------------------------------------------
# Main: train, test, write PDB, plot histogram
# ----------------------------------------------------------------------
if __name__ == "__main__":
    coords = _read_pdb_coords()
    # print(coords)
    test_count = 50000
    kT = 1
    scale = test_count // 5
    folder = 'glowblockenergyaltposter'
    for nn_model in glob.glob(f'{folder}/*_model_weights.pt'):
        print(f"Testing model: {nn_model}")
        filename = folder + '/' + nn_model.split('/')[-1][:-len('_model_weights.pt')]

        if '1layer' in filename:
            networkdims = []
        elif '2layer' in filename:
            networkdims = [32]
        elif '3layer' in filename:
            networkdims = [32, 32]
        elif '9layer' in filename:
            networkdims = [32, 32, 32, 16, 16, 16, 8, 8]

        U, neighborlists = build_crystal_U(1, 216, 2, periodic=True)
        first_pairs = _build_neighbor_pairs(neighborlists[0])
        second_pairs = _build_neighbor_pairs(neighborlists[1])
        glow = GlowBlock(dt=0.001, neighborlists=neighborlists, network_dims=networkdims, dim=2)
        flow = MultiStep(glow, 1)
        weights = torch.load(nn_model)
        glow.load_state_dict(weights)
        glow.eval()

        overall_interactions = torch.empty((test_count, 6), dtype=torch.float32)
        overall_energies = torch.empty((test_count,), dtype=torch.float32)
        last_Ga_composition = torch.empty((test_count,), dtype=torch.float32)
        sample_time_s = 0.0
        interaction_time_s = 0.0
        loss_func = lambda x, x0, logJ: 1 / kT * (U(x['r'], x['t']) - U(x0['r'], 0.0)) - logJ

        with torch.no_grad():
            for i in range(test_count):
                if i % 100 == 0:
                    print(f"Testing sample {i}/{test_count}")

                t0 = time.perf_counter()
                Ga_percents, _ = test_glowblock_two_part(
                    flow=flow,
                    generate_sample=generate_sample,
                    percent_func=percentA,
                    loss_func=loss_func,
                    batch_size=1,
                    Na=216,
                    dim=2,
                    sigma=1.0,
                    compute_loss=False,
                )
                t1 = time.perf_counter()
                last_Ga_frame = Ga_percents[0]
                last_Ga_composition[i] = last_Ga_frame.mean().item()

                interactions = _compute_interactions(last_Ga_frame, first_pairs, second_pairs)
                t2 = time.perf_counter()
                overall_interactions[i] = interactions

                # energy = interactions[0] * 0.3 + interactions[1] * 0.5 + interactions[2] * 0.1 + \
                #          interactions[3] * 0.15 + interactions[4] * 0.0 + interactions[5] * 0.05

                energy = interactions[0] * 0.3 + interactions[1] * 0.1 + interactions[2] * 0.5
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
        
        #histogram of 1st neighbor interactions
        plt.figure(figsize=(10, 6))
        plt.hist(overall_interactions[:, 0].numpy(), bins=30, alpha=0.5, label='1st neighbor Ga-Ga')
        plt.hist(overall_interactions[:, 1].numpy(), bins=30, alpha=0.5, label='1st neighbor Ga-As')
        plt.hist(overall_interactions[:, 2].numpy(), bins=30, alpha=0.5, label='1st neighbor As-As')
        plt.xlabel('Interaction Count')
        plt.ylabel('Frequency')
        plt.xlim(0, 864)
        plt.ylim(0, scale)
        plt.title(f'Histogram of 1st Neighbor Interactions for {nn_model.split("/")[-1].split(".")[0]} (Test Count: {test_count})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{filename}_1st_neighbor_interactions_histogram.png', dpi=150)

        #histogram of 2nd neighbor interactions
        plt.figure(figsize=(10, 6))
        plt.hist(overall_interactions[:, 3].numpy(), bins=30, alpha=0.5, label='2nd neighbor Ga-Ga')
        plt.hist(overall_interactions[:, 4].numpy(), bins=30, alpha=0.5, label='2nd neighbor Ga-As')
        plt.hist(overall_interactions[:, 5].numpy(), bins=30, alpha=0.5, label='2nd neighbor As-As')
        plt.xlabel('Interaction Count')
        plt.ylabel('Frequency')
        plt.xlim(0, 2592)
        plt.ylim(0, scale)
        plt.title(f'Histogram of 2nd Neighbor Interactions for {nn_model.split("/")[-1].split(".")[0]} (Test Count: {test_count})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{filename}_2nd_neighbor_interactions_histogram.png', dpi=150)

        #histogram of energies
        plt.figure(figsize=(10, 6))
        plt.hist(overall_energies.numpy(), bins=30, alpha=0.5, label='Potential Energy')
        plt.xlabel('Potential Energy')
        plt.ylabel('Frequency')
        plt.xlim(200, 700)
        plt.ylim(0, scale)
        plt.title(f'Histogram of Potential Energies for {nn_model.split("/")[-1].split(".")[0]} (Test Count: {test_count})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{filename}_potential_energy_histogram.png', dpi=150)

        #histogram of Ga compositions
        plt.figure(figsize=(10, 6))
        plt.hist(last_Ga_composition.numpy(), bins=30, alpha=0.5, label='Ga Composition')
        plt.xlabel('Ga Composition')
        plt.ylabel('Frequency')
        plt.xlim(0,1)
        plt.ylim(0, scale)
        plt.title(f'Histogram of Ga Compositions for {nn_model.split("/")[-1].split(".")[0]} (Test Count: {test_count})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{filename}_ga_composition_histogram.png', dpi=150)

        save_dict = {
            'overall_interactions': overall_interactions,
            'overall_energies': overall_energies,
            'last_Ga_composition': last_Ga_composition,
        }

        # Save the dictionary
        torch.save(save_dict, f'{filename}_test_results.pt')