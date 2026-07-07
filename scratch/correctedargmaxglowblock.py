import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

from alchemistlib.flows import GlowBlock, MultiStep, Q, fix_kT
from utils import _write_pdb_trajectory, _read_pdb_coords, clone_state, neighbor_masks, _format_pdb_atom
from glowblock import build_U

# ----------------------------------------------------------------------
# Probabilistic inverse q(v | x)
# ----------------------------------------------------------------------
def sample_v(x_onehot):
    """
    Binary softplus-threshold probabilistic inverse for Argmax Flow.
    """
    B, N, K = x_onehot.shape
    assert K == 2, "This helper is for binary categories only."

    device = x_onehot.device
    dtype = x_onehot.dtype

    labels = x_onehot.argmax(dim=-1)  # (B, N)

    # Sample u from iid standard logistic.
    # v = logit(eps), u ~ Uniform(0, 1)
    u = torch.rand(B, N, K, device=device, dtype=dtype).clamp(1e-6, 1.0 - 1e-6)
    v = torch.log(u) - torch.log1p(-u)

    # if argmax is wrong, flip the two channels
    wrong = v.argmax(dim=-1) != labels          # (B, N)
    v_flipped = v.flip(dims=[-1])               # swap class 0 and 1
    v = torch.where(wrong.unsqueeze(-1), v_flipped, v)

    # log q(v|x)
    # flip/order transform gives factor 2 per site
    logq = (F.logsigmoid(v) + F.logsigmoid(-v)).sum(dim=[1, 2]) + N * math.log(2.0)

    return v, logq


# ----------------------------------------------------------------------
# Argmax Flow loss for ideal gas
# ----------------------------------------------------------------------
def calc_argmax_flow_loss_ideal_gas(
    flow,
    x,
    sigma=1.0
):
    """
    Correct ideal-gas Argmax Flow loss.
        ELBO = E_{v ~ q(v|x)} [log p(v) - log q(v|x)]
    """

    # Sample v ~ q(v|x) satisfying argmax(v) = x.
    v, logq = sample_v(x["r"])

    v_flow = clone_state(x)
    v_flow["r"] = v

    z, logJ, info = flow(v_flow, inverse=True)

    # Base density log p(z)
    logpz = -0.5 * math.log(2.0 * math.pi * sigma ** 2) - 0.5 * (z["r"] / sigma) ** 2
    logpz = logpz.sum(dim=[1, 2])

    loss = -(logpz + logJ - logq)

    return loss, {
        "log_pz": logpz.detach(),
        "logJ": logJ.detach() if torch.is_tensor(logJ) else logJ,
        "log_q_v_given_x": logq.detach(),
        "v": v.detach(),
        "z_r": z["r"].detach(),
    }


def gen(batch_size,Na,num_classes,):
    return torch.ones(batch_size,Na,num_classes)

def genordered(batch_size,Na,num_classes):
    idx = (torch.arange(1, 55).repeat_interleave(4)) % 2
    base = torch.nn.functional.one_hot(idx, num_classes=2).float() #(N, 2)
    base = base.unsqueeze(0) #(1, N, 2)
    base = base.expand(batch_size, -1, -1) #(B, N, 2)
    return base

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_glowblock_ideal_gas_argmax(
    batch_size=32,
    Na=216,
    num_classes=2,
    p_class0=0.75,
    sigma=1.0,
    n_steps_flow=1,
    epoch_count=10000,
    lr=1e-3,
    network_dims=(32, 32),
    device=None
):
    """
    Train Argmax Flow on ideal-gas independent binary categorical variables.

    No U(r), no neighbor masks, no interaction terms.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    U, neighborlists = build_U(batch_size, Na, num_classes, periodic=True, percenttype='direct')

    glow = GlowBlock(
        dt=0.001,
        neighborlists=neighborlists,
        network_dims=list(network_dims),
        dim=num_classes,
    ).to(device)

    flow = MultiStep(glow, n_steps_flow)

    optimizer = optim.Adam(glow.parameters(), lr=lr)

    losses = []

    for it in range(epoch_count):
        x = {
            "r": genordered(batch_size, Na, num_classes).to(device),
            "p": torch.zeros(batch_size, Na, num_classes, device=device),
            "t": torch.tensor(0.0, device=device),
        }

        loss, info = calc_argmax_flow_loss_ideal_gas(
            flow=flow,
            x=x,
            sigma=sigma
        )

        mean_loss = loss.mean()

        optimizer.zero_grad()
        mean_loss.backward()
        optimizer.step()

        losses.append(mean_loss.item())

        if (it + 1) % 50 == 0:
            mean_log_pz = info["log_pz"].mean().item()
            mean_log_q = info["log_q_v_given_x"].mean().item()

            print(
                f"[train] epoch {it + 1:5d} | "
                f"loss = {mean_loss.item(): .4f} | "
                f"log p(z) = {mean_log_pz: .4f} | "
                f"log q(v|x) = {mean_log_q: .4f}"
            )

    return glow, losses


# ----------------------------------------------------------------------
# Testing / visualization
# ----------------------------------------------------------------------
@torch.no_grad()
def test_glowblock_ideal_gas_argmax(
    glow,
    batch_size=4,
    Na=216,
    num_classes=2,
    p_class0=0.75,
    sigma=1.0,
    n_steps_flow=1,
    device=None
):
    if device is None:
        device = next(glow.parameters()).device

    flow = MultiStep(glow, n_steps_flow)

    x = {
        "r": genordered(batch_size, Na, num_classes).to(device),
        "p": torch.zeros(batch_size, Na, num_classes, device=device),
        "t": torch.tensor(0.0, device=device),
    }

    loss, info = calc_argmax_flow_loss_ideal_gas(
        flow=flow,
        x=x,
        sigma=sigma
    )

    original = x["r"].detach().cpu()
    lifted = torch.sigmoid(info["v"]).detach().cpu()
    base = torch.sigmoid(info["z_r"]).detach().cpu()

    frames = torch.stack([original, lifted, base], dim=0)

    return frames, loss.mean().item()


# ----------------------------------------------------------------------
# Optional: sample from trained model
# ----------------------------------------------------------------------

@torch.no_grad()
def sample_from_argmax_flow(
    glow,
    batch_size=4,
    Na=216,
    num_classes=2,
    p_class0=0.75,
    sigma=1.0,
    n_steps_flow=1,
    device=None,
):
    """
    Generates categorical samples from the learned Argmax Flow.

    Sampling procedure from the paper:
        z ~ base
        v = g(z)
        x = argmax(v)

    Warning:
        This assumes MultiStep(...)(..., inverse=False) maps z -> v.
        If your library uses the opposite convention, adjust accordingly.
    """
    if device is None:
        device = next(glow.parameters()).device

    flow = MultiStep(glow, n_steps_flow)
    normal = torch.distributions.normal.Normal(0, 1)

    z_state = {
        "r": Q(normal.sample((batch_size, Na, num_classes))) * sigma,
        "p": torch.zeros(batch_size, Na, num_classes, device=device),
        "t": torch.tensor(0.0, device=device),
    }
    z_init = clone_state(z_state)

    v_state, logJ, info = flow(z_state, inverse=False)

    v = v_state["r"]
    labels = v.argmax(dim=-1)
    onehot = F.one_hot(labels, num_classes=num_classes).float()

    return {
        "v": v,
        "labels": labels,
        "onehot": onehot,
        "sigmoid_v": torch.sigmoid(v),
        "raw": torch.stack([z_init["r"], v_state["r"]], dim=0),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    coords = _read_pdb_coords()

    folder = "neighbor_argmax"
    Path(folder).mkdir(parents=True, exist_ok=True)

    batch_size = 32
    Na = 216
    num_classes = 2
    n_steps_flow = 1

    for epoch_count in [10000]:
        for network_dims in [(),[32],(32, 32)]:
            filename = f"{folder}/{len(network_dims) + 1}layer_epoch{epoch_count}"

            glow, train_losses = train_glowblock_ideal_gas_argmax(
                batch_size=batch_size,
                Na=Na,
                num_classes=num_classes,
                sigma=1.0,
                n_steps_flow=n_steps_flow,
                epoch_count=epoch_count,
                lr=1e-3,
                network_dims=network_dims
            )

            frames, test_loss = test_glowblock_ideal_gas_argmax(
                glow,
                batch_size=batch_size,
                Na=Na,
                num_classes=num_classes,
                sigma=1.0,
                n_steps_flow=n_steps_flow
            )

            torch.save(glow.state_dict(), f"{filename}_model_weights.pt")

            # Interpret channel 0 as Ga and channel 1 as As.
            Ga_percents = frames[..., 0]
            As_percents = frames[..., 1]

            print("Ga_percents shape:", Ga_percents.shape)
            print("As_percents shape:", As_percents.shape)

            _OUTPUT_PDB = f"{filename}.pdb"

            # Write only first batch item.
            _write_pdb_trajectory(
                _OUTPUT_PDB,
                coords,
                Ga_percents[:, 0],
                As_percents[:, 0],
            )

            # Optional: generate samples from the trained model.
            samples = sample_from_argmax_flow(
                glow,
                batch_size=32,
                Na=Na,
                num_classes=num_classes,
                sigma=1.0,
                n_steps_flow=n_steps_flow,
            )
            p_class0_sampled = samples["onehot"][..., 0].mean().item()

            plt.plot(train_losses, label="train loss")
            plt.xlabel("Epochs")
            plt.ylabel("Negative ELBO")
            plt.legend()
            plt.suptitle(
                f"Ideal Gas Argmax Flow "
                f"(epoch_count={epoch_count}, n_steps_flow={n_steps_flow})"
            )
            plt.title(f"Test Loss: {test_loss:.4f}, Fraction Ga: {p_class0_sampled:.2f}")
            plt.savefig(f"{filename}_loss.png", dpi=150)
            plt.close()

            print(samples["raw"].shape)

            _write_pdb_trajectory(
                f"{filename}_sampled.pdb",
                coords,
                samples["raw"][:, 0, :, 0],
                samples["raw"][:, 0, :, 1],
            )

            print("Sampled labels shape:", samples["labels"].shape)
            print("Sampled class-0 fraction:", p_class0_sampled)