import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from alchemistlib.flows import GlowBlock, MultiStep, Q
import matplotlib.pyplot as plt

from utils import _format_pdb_atom, _write_pdb_trajectory, neighbor_masks, _read_pdb_coords


def atom_energy_mixed_batched(pA_ga, f1_ga, f2_ga):
    pA_as = 1 - pA_ga
    f1_as = 1 - f1_ga
    f2_as = 1 - f2_ga

    E1 = (
        pA_ga * (f1_ga * 0.3 + f1_as * 0.5)
        + pA_as * (f1_ga * 0.5 + f1_as * 0.1)
    )
    E2 = (
        pA_ga * (f2_ga * 0.15 + f2_as * 0.0)
        + pA_as * (f2_ga * 0.0 + f2_as * 0.05)
    )

    return E1.sum(dim=-1) + E2.sum(dim=-1)

def percentA(r):
    return torch.softmax(r, dim=-1)[..., 0]

def hard_percentA(r):
    return (torch.argmax(r, dim=-1) == 0).float()

def build_U(batch_size, Na, dim, periodic=True):
    nbr1, nbr2 = neighbor_masks(periodic=periodic)

    nbr1_tensor = torch.tensor(nbr1, dtype=torch.long).unsqueeze(0)
    nbr2_tensor = torch.tensor(nbr2, dtype=torch.long).unsqueeze(0)

    def U(r, t, perfect_func=percentA):
        device = r.device
        B = r.shape[0]

        nbr1_dev = nbr1_tensor.to(device).expand(B, -1, -1)
        nbr2_dev = nbr2_tensor.to(device).expand(B, -1, -1)

        dr = perfect_func(r).unsqueeze(-1)

        dr_expanded1 = dr.expand(-1, -1, nbr1_dev.shape[-1])
        dr_expanded2 = dr.expand(-1, -1, nbr2_dev.shape[-1])

        nbr1_vals = torch.gather(dr_expanded1, dim=1, index=nbr1_dev)
        nbr2_vals = torch.gather(dr_expanded2, dim=1, index=nbr2_dev)

        return atom_energy_mixed_batched(dr, nbr1_vals, nbr2_vals).sum(1)

    return U, [nbr1, nbr2]


class BinaryArgmaxQ(nn.Module):
    def __init__(self, init_log_sigma=-0.5):
        super().__init__()
        self.embedding = nn.Embedding(2, 4)

        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[:, 2:] = init_log_sigma

    def forward(self, labels, margin=1.0):
        params = self.embedding(labels)

        mu = params[..., :2]
        log_sigma = params[..., 2:].clamp(-8.0, 4.0)
        sigma = torch.exp(log_sigma)

        q_u = Normal(mu, sigma)
        u = q_u.rsample()

        u0 = u[..., 0]
        u1 = u[..., 1]

        T0 = u0 - margin
        r0_if_0 = u0
        r1_if_0 = T0 - F.softplus(T0 - u1)
        log_jac_if_0 = torch.log(torch.sigmoid(T0 - u1) + 1e-12)

        T1 = u1 - margin
        r1_if_1 = u1
        r0_if_1 = T1 - F.softplus(T1 - u0)
        log_jac_if_1 = torch.log(torch.sigmoid(T1 - u0) + 1e-12)

        is_0 = labels == 0

        r0 = torch.where(is_0, r0_if_0, r0_if_1)
        r1 = torch.where(is_0, r1_if_0, r1_if_1)

        r = torch.stack([r0, r1], dim=-1)

        log_q_u = q_u.log_prob(u).sum(dim=(-1, -2))
        log_det = torch.where(is_0, log_jac_if_0, log_jac_if_1).sum(dim=-1)
        log_q_r = log_q_u - log_det

        return r, log_q_r

def standard_normal_logprob(y):
    dist = Normal(torch.zeros_like(y), torch.ones_like(y))
    return dist.log_prob(y).flatten(start_dim=1).sum(dim=1)

def clone_state(x):
    return {
        "r": x["r"].clone(),
        "p": x["p"].clone(),
        "t": x["t"],
    }

def log_p_rp_from_flow(r, p, t, flow):
    y = {
        "r": r,
        "p": p,
        "t": t,
    }

    z, logJ_inv, info = flow(clone_state(y), inverse=True)

    log_p_z = standard_normal_logprob(z["r"]) + standard_normal_logprob(z["p"])
    log_p_rp = log_p_z + logJ_inv

    return log_p_rp


def argmax_flow_loss(
    labels,
    q_module,
    flow,
    U=None,
    kT=1.0,
    t=0.0,
    margin=1.0,
    energy_weight=0.0,
):
    r, log_q_r_given_x = q_module(labels, margin=margin)

    p = torch.randn_like(r)
    log_q_p = standard_normal_logprob(p)

    log_q_rp_given_x = log_q_r_given_x + log_q_p
    log_p_rp = log_p_rp_from_flow(r, p, t, flow)

    objective = log_p_rp - log_q_rp_given_x

    if U is not None and energy_weight != 0.0:
        energy = U(r, t, perfect_func=hard_percentA)
        objective = objective - energy_weight * energy / kT
    else:
        energy = torch.zeros_like(objective)

    loss = -objective.mean()

    info = {
        "r": r.detach(),
        "p": p.detach(),
        "hard_labels": torch.argmax(r.detach(), dim=-1),
        "target_labels": labels.detach(),
        "log_p_rp": log_p_rp.detach(),
        "log_q_rp": log_q_rp_given_x.detach(),
        "energy": energy.detach(),
        "objective": objective.detach(),
    }

    return loss, info


def make_random_labels(batch_size, Na, device):
    return torch.randint(0, 2, (batch_size, Na), dtype=torch.long, device=device)


def make_half_half_labels(batch_size, Na, device):
    labels = torch.zeros(batch_size, Na, dtype=torch.long, device=device)
    labels[:, Na // 2 :] = 1
    return labels


def generate_sample(batch_size, Na, dim, sigma, device=None):
    normal = Normal(0, 1)

    r = Q(normal.sample((batch_size, Na, dim))) * sigma
    p = Q(normal.sample((batch_size, Na, dim)))

    if device is not None:
        r = r.to(device)
        p = p.to(device)

    return {
        "r": r,
        "p": p,
        "t": 0.0,
    }


def train_glowblock_two_part_argmax(
    batch_size=1,
    Na=216,
    dim=2,
    sigma=1.0,
    kT=1.0,
    periodic=True,
    n_steps_flow=10,
    epoch_count=25,
    lr=1e-3,
    network_dims=[16],
    margin=1.0,
    energy_weight=0.0,
    label_mode="random",
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    U, neighborlists = build_U(batch_size, Na, dim, periodic=periodic)

    glow = GlowBlock(
        en=U,
        dt=0.001,
        neighborlists=neighborlists,
        network_dims=network_dims,
        dim=dim,
    ).to(device)

    flow = MultiStep(glow, n_steps_flow).to(device)

    q_module = BinaryArgmaxQ().to(device)

    optimizer = optim.Adam(
        list(glow.parameters()) + list(q_module.parameters()),
        lr=lr,
    )

    losses = []

    for it in range(epoch_count):
        if label_mode == "half":
            labels = make_half_half_labels(batch_size, Na, device)
        else:
            labels = make_random_labels(batch_size, Na, device)

        loss, info = argmax_flow_loss(
            labels=labels,
            q_module=q_module,
            flow=flow,
            U=U,
            kT=kT,
            t=0.0,
            margin=margin,
            energy_weight=energy_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (it + 1) % 50 == 0:
            match = (info["hard_labels"] == info["target_labels"]).float().mean().item()
            print(
                f"[train] epoch {it+1:5d} "
                f"loss={loss.item():.4f} "
                f"match={match:.4f} "
                f"log_p={info['log_p_rp'].mean().item():.4f} "
                f"log_q={info['log_q_rp'].mean().item():.4f} "
                f"energy={info['energy'].mean().item():.4f}"
            )

    return glow, q_module, losses


def test_glowblock_two_part_argmax(
    glow,
    batch_size=1,
    Na=216,
    dim=2,
    sigma=1.0,
    periodic=True,
    n_steps_flow=1,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow = MultiStep(glow, n_steps_flow).to(device)

    x = generate_sample(batch_size, Na, dim, sigma, device=device)

    ga_percents = [percentA(x["r"]).detach().cpu()]

    x, logJ, info = flow(x)

    ga_percents.append(percentA(x["r"]).detach().cpu())

    hard_labels = torch.argmax(x["r"], dim=-1).detach().cpu()

    return torch.stack(ga_percents), hard_labels


if __name__ == "__main__":
    coords = _read_pdb_coords()

    folder = "argmaxresults"

    for epoch_count in [10000]:
        for n_steps_flow in [1]:
            batch_size = 2
            network_dims = [32]

            glow, q_module, train_losses = train_glowblock_two_part_argmax(
                periodic=True,
                n_steps_flow=n_steps_flow,
                epoch_count=epoch_count,
                batch_size=batch_size,
                network_dims=network_dims,
                energy_weight=1.0,
                label_mode="random",
            )

            ga_percents, hard_labels = test_glowblock_two_part_argmax(
                glow,
                periodic=True,
                batch_size=batch_size,
                n_steps_flow=n_steps_flow,
            )

            torch.save(
                glow.state_dict(),
                f"{folder}/{len(network_dims)+1}layer_epoch{epoch_count}_steps{n_steps_flow}_model_weights.pt",
            )

            torch.save(
                q_module.state_dict(),
                f"{folder}/{len(network_dims)+1}layer_epoch{epoch_count}_steps{n_steps_flow}_q_weights.pt",
            )

            as_percents = 1 - ga_percents

            output_pdb = (
                f"{folder}/{len(network_dims)+1}layer_epoch{epoch_count}_steps{n_steps_flow}.pdb"
            )

            _write_pdb_trajectory(
                output_pdb,
                coords,
                ga_percents[:, 0],
                as_percents[:, 0],
            )

            plt.plot(train_losses, label="train loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.suptitle(
                f"Argmax GlowBlock Training Loss "
                f"(epoch_count={epoch_count}, n_steps_flow={n_steps_flow})"
            )
            plt.savefig(
                f"{folder}/{len(network_dims)+1}layer_epoch{epoch_count}_steps{n_steps_flow}_loss.png",
                dpi=150,
            )
            plt.close()