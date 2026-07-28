import torch, copy
from typing import Callable, Any, Tuple, Generator

def train_and_summarize(
    model: torch.nn.Module,
    loss_fn: Callable[[Any], torch.Tensor],
    data_generator: Callable[[], Generator[Any, None, None]],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batches_per_epoch: int,
    feature_extractor: Callable[[torch.nn.Module, Any], torch.Tensor],
    inverse: bool = False
) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardized training loop that optimizes a model and summarizes the 
    statistical properties of a feature extractor over the final epoch.

    Args:
        model: The NN module to train.
        loss_fn: Function taking a batch of data and returning a scalar loss.
        data_generator: Function returning a generator of data batches.
        optimizer: Optimizer for the model parameters.
        epochs: Number of training epochs.
        batches_per_epoch: Number of batches per epoch.
        feature_extractor: Function taking (model, batch) and returning 
                           the features to summarize (B, M, D).
        reverse: Whether to train in reverse direction.

    Returns:
        Trained model, mean of features (M, D), variance of features (M, D), and loss values.
    """
    means = []
    vars = []
    losses = []
    all_features = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # ignored, Final epoch is for evaluation (no parameter updates)
        model.train()
        
        gen = data_generator()
        for _ in range(batches_per_epoch):
            try:
                x0 = next(gen)
            except StopIteration:
                break

            optimizer.zero_grad()
            x = copy.deepcopy(x0)  # Ensure original batch is not modified

            x, lJ, info = model(x, inverse=inverse)

            feat = feature_extractor(x)
            all_features.append(feat)

            loss = loss_fn(x0, x, lJ)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Loss is NaN/Inf at epoch {epoch}")
            
            losses.append([loss.mean().item(),lJ.mean().item()])
            # if losses[-1] > 1e6:
            #     print(x0)
            #     raise RuntimeError(f"Loss is too large at epoch {epoch}: {losses[-1]}")

            loss.backward()
            optimizer.step()

        #if True:
            # Stack all batches: (Total_B, M, D)
            #stacked_features = torch.cat(all_features, dim=0)
            
    return all_features, losses
