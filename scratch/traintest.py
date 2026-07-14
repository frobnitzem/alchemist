import torch
import torch.optim as optim
from utils import clone_state

def train_glowblock_two_part(
    flow,
    model,
    generate_sample,
    loss_func,
    batch_size=1,
    Na=216,
    dim=2,
    sigma=1.0,
    num_batches=25,
    lr=1e-3,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for it in range(num_batches):
        x = generate_sample(batch_size, Na, dim, sigma)
        x0 = clone_state(x)

        x, logJ, info = flow(x)
        loss = loss_func(x, x0, logJ)

        losses.append(loss.mean().item())
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        if (it + 1) % 50 == 0:
            print(f"[train] batch {it+1:4d}  loss = {loss.mean().item():.4f}")

    return model, losses


def test_glowblock_two_part(
    flow,
    generate_sample,
    percent_func,
    loss_func,
    batch_size=1,
    Na=216,
    dim=2,
    sigma=1.0,
    compute_loss=True,
):
    x = generate_sample(batch_size, Na, dim, sigma)
    x0 = clone_state(x)

    Ga_percents = [percent_func(x0['r'])]

    x, logJ, info = flow(x)
    Ga_percents.append(percent_func(x['r']))

    if not compute_loss:
        return torch.stack(Ga_percents), None

    loss = loss_func(x, x0, logJ)

    return torch.stack(Ga_percents), loss.mean().item()


def train_argmax_flow(flow, model, generate_state, loss_func, num_batches=10000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for it in range(num_batches):
        x = generate_state()
        loss, info = loss_func(flow, x)

        mean_loss = loss.mean()

        optimizer.zero_grad()
        mean_loss.backward()
        optimizer.step()

        losses.append(mean_loss.item())

        if (it + 1) % 50 == 0:
            print(f"[train] epoch {it + 1:5d} | loss = {mean_loss.item(): .4f}")

    return model, losses


@torch.no_grad()
def test_argmax_flow(flow, generate_state, loss_func, frame_builder):
    x = generate_state()
    loss, info = loss_func(flow, x)
    frames = frame_builder(x, info)

    return frames, loss.mean().item()
