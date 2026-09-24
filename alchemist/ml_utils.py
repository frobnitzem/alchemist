import torch, copy
from typing import Callable, Any, Tuple, Generator

def train_and_summarize(
    model: torch.nn.Module,
    loss_fn: Callable[[Any, Any, torch.Tensor], torch.Tensor],
    data_generator: Callable[[], Generator[Any, None, None]],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batches_per_epoch: int,
    feature_extractor: Callable[[torch.nn.Module, Any], torch.Tensor],
    inverse: bool = False
) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    """
    Standardized training loop that optimizes a model and summarizes the 
    statistical properties of a feature extractor over the final epoch.

    Args:
        model: The NN module to train.
        loss_fn: Function taking (x0, x, lJ) and returning a scalar loss.
        data_generator: Function returning a generator of data batches.
        optimizer: Optimizer for the model parameters.
        epochs: Number of training epochs.
        batches_per_epoch: Number of batches per epoch.
        feature_extractor: Function taking (model, batch) and returning 
                           the features to summarize (B, M, D).
        inverse: Whether to train in reverse direction.

    Returns:
        Trained model, mean of features (M, D), variance of features (M, D).
    """
    for epoch in range(epochs - 1):
        model.train()
        gen = data_generator()
        for _ in range(batches_per_epoch):
            try:
                x0 = next(gen)
            except StopIteration:
                break

            optimizer.zero_grad()
            x = copy.deepcopy(x0)

            x, lJ, info = model(x, inverse=inverse)
            loss = loss_fn(x0, x, lJ)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Loss is NaN/Inf at epoch {epoch}")
            
            loss.backward()
            optimizer.step()

    # Final epoch for evaluation
    model.eval()
    eval_features = []
    gen = data_generator()
    for _ in range(batches_per_epoch):
        try:
            x0 = next(gen)
        except StopIteration:
            break
        with torch.no_grad():
            x, _, _ = model(x0, inverse=inverse)
            feat = feature_extractor(model, x)
            eval_features.append(feat)
    
    stacked_features = torch.cat(eval_features, dim=0) # (Total_B, M, D)
    mean = stacked_features.mean(dim=0)
    var = stacked_features.var(dim=0)
    
    return model, mean, var
