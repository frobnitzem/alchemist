import math
from pathlib import Path

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from alchemistlib.flows import simpleGlowBlock, MultiStep, Q, fix_kT
from utils import _write_pdb_trajectory, _read_pdb_coords, clone_state
from traintest import train_argmax_flow, test_argmax_flow

# ----------------------------------------------------------------------
# Ideal-gas categorical samples
# ----------------------------------------------------------------------

def generate_ideal_gas_sample(
    batch_size,
    Na,
    num_classes=2,
    ratio=0.75,
    device=None,
    dtype=torch.float32,
    shuffle=True,
):
    """
    Generate ideal-gas binary categorical samples with a fixed composition.

    Returns
    -------
    x:
        Dictionary with:
            x["labels"]: integer labels, shape (B, Na)
            x["r"]: one-hot categorical tensor, shape (B, Na, 2)
            x["p"]: dummy tensor, shape (B, Na, 2)
            x["t"]: scalar tensor
    """
    assert num_classes == 2, "This function currently supports binary categories only."
    assert 0.0 <= ratio <= 1.0, "ratio must be between 0 and 1."

    if device is None:
        device = torch.device("cpu")

    n_class0 = int(round(ratio * Na))
    n_class1 = Na - n_class0

    base_labels = torch.cat(
        [
            torch.zeros(n_class0, dtype=torch.long, device=device),
            torch.ones(n_class1, dtype=torch.long, device=device),
        ],
        dim=0,
    )  # shape: (Na,)

    labels = base_labels.unsqueeze(0).expand(batch_size, -1).clone()

    if shuffle:
        # Randomly permute sites independently for each batch item.
        rand = torch.rand(batch_size, Na, device=device)
        perm = rand.argsort(dim=1)
        labels = torch.gather(labels, dim=1, index=perm)

    onehot = F.one_hot(labels, num_classes=num_classes).to(dtype)
    normal = torch.distributions.normal.Normal(0, 1)
    x = {
        "labels": labels,
        "r": onehot,
        "p": normal.sample((batch_size, Na, num_classes)),
        "t": torch.tensor(0.0, device=device, dtype=dtype),
    }

    return x


# ----------------------------------------------------------------------
# Probabilistic inverse q(v | x)
# ----------------------------------------------------------------------
def sample_v(x_onehot):
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

    # q(a | v, x) = N(0, I), chosen here to be independent.
    # Keep the original tensor because flow mutates/replaces state entries.
    p_aux = sigma * torch.randn_like(v)

    observed_state = {
        "r": v,
        "p": p_aux,
        "t": torch.tensor(0.0, device=v.device),
    }

    z, logJ, info = flow(observed_state, inverse=True)

    # Base density log p(z)
    logpz = -0.5 * math.log(2.0 * math.pi * sigma ** 2) - 0.5 * (z["r"] / sigma) ** 2
    logpz = logpz.sum(dim=[1, 2])

    logpp = -0.5 * math.log(2.0 * math.pi * sigma ** 2) - 0.5 * (z['p'] / sigma) ** 2
    logpp = logpp.sum(dim=[1, 2])
    logq_p_aux = -0.5 * math.log(2.0 * math.pi * sigma ** 2) - 0.5 * (p_aux / sigma) ** 2
    logq_p_aux = logq_p_aux.sum(dim=[1, 2]) 

    loss = -(logpz + logJ - logq - logq_p_aux + logpp)

    return loss, {
        "log_pz": logpz.detach(),
        "logJ": logJ.detach() if torch.is_tensor(logJ) else logJ,
        "log_q_v_given_x": logq.detach(),
        "v": v.detach(),
        "z_r": z["r"].detach(),
        "log_pp": logpp.detach(),
        "log_q_p_aux": logq_p_aux.detach(),
    }


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

    z_r = sigma * torch.randn(
        batch_size,
        Na,
        num_classes,
        device=device,
    )

    z_p = sigma * torch.randn_like(z_r)

    z_state = {
        "r": z_r,
        "p": z_p,
        "t": torch.tensor(
            0.0,
            device=device
        ),
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

    folder = "ideal_gas_argmax"
    Path(folder).mkdir(parents=True, exist_ok=True)

    batch_size = 32
    Na = 216
    num_classes = 2
    n_steps_flow = 1

    # For ideal gas independent binary mixture.
    # class 0 probability.
    # p_class0 = 0.75

    for p_class0 in [0.1,0.5,0.75]:
        for num_batches in [3000]:
            for network_dims in [(),[32],(32, 32)]:
                filename = f"{folder}/{len(network_dims) + 1}layer_num_batches{num_batches}_comp{p_class0:.2f}"
                sigma = 1.0

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                glow = simpleGlowBlock(
                    dt=0.001,
                    network_dims=list(network_dims),
                    dim=num_classes,
                ).to(device)
                flow = MultiStep(glow, n_steps_flow)

                generate_state = lambda: generate_ideal_gas_sample(
                    batch_size=batch_size,
                    Na=Na,
                    num_classes=num_classes,
                    ratio=p_class0,
                    device=device,
                )
                loss_func = lambda flow_obj, state: calc_argmax_flow_loss_ideal_gas(flow_obj, state, sigma)

                frame_builder = lambda state, info: torch.stack([
                    state["r"].detach().cpu(),
                    torch.sigmoid(info["v"]).detach().cpu(),
                    torch.sigmoid(info["z_r"]).detach().cpu(),
                ], dim=0)

                glow, train_losses = train_argmax_flow(
                    flow=flow,
                    model=glow,
                    generate_state=generate_state,
                    loss_func=loss_func,
                    num_batches=num_batches,
                    lr=1e-3,
                )

                frames, test_loss = test_argmax_flow(
                    flow=flow,
                    generate_state=generate_state,
                    loss_func=loss_func,
                    frame_builder=frame_builder,
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
                    batch_size=1000,
                    Na=Na,
                    num_classes=num_classes,
                    sigma=1.0,
                    n_steps_flow=n_steps_flow,
                )
                p_class0_sampled = samples["onehot"][..., 0].mean().item()

                plt.plot(train_losses, label="train loss")
                plt.xlabel("Batches")
                plt.ylabel("Negative ELBO")
                plt.legend()
                plt.suptitle(
                    f"Ideal Gas Argmax Flow "
                    f"(num_batches={num_batches}, n_steps_flow={n_steps_flow})"
                )
                plt.title(f"Test Loss: {test_loss:.4f}, Fraction Ga: {p_class0_sampled:.2f}")
                plt.savefig(f"{filename}_loss.png", dpi=150)
                plt.close()

                plt.hist(torch.mean(1-samples["labels"].float(), dim=1), bins=20, density=True)
                plt.xlabel("Class Label")
                plt.ylabel("Density")
                plt.title(f"Distribution of Sampled Class Labels (Fraction Ga: {p_class0_sampled:.2f})")
                plt.savefig(f"{filename}_class_label_distribution.png", dpi=150)
                plt.close()

                print("Sampled labels shape:", samples["labels"].shape)
                print("Sampled class-0 fraction:", p_class0_sampled)