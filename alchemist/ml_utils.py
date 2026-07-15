import torch
from typing import Callable, Any, Tuple, Generator

def train_and_summarize(
    model: torch.nn.Module,
    loss_fn: Callable[[Any], torch.Tensor],
    data_generator: Callable[[], Generator[Any, None, None]],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batches_per_epoch: int,
    feature_extractor: Callable[[torch.nn.Module, Any], torch.Tensor],
) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
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

    Returns:
        Trained model, mean of features (M, D), and variance of features (M, D).
    """
    for epoch in range(epochs):
        # Final epoch is for evaluation (no parameter updates)
        is_eval = (epoch == epochs - 1)
        
        if is_eval:
            model.eval()
        else:
            model.train()

        all_features = []
        
        gen = data_generator()
        for _ in range(batches_per_epoch):
            try:
                batch = next(gen)
            except StopIteration:
                break

            if not is_eval:
                optimizer.zero_grad()
                loss = loss_fn(batch)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"Loss is NaN/Inf at epoch {epoch}")
                
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    feat = feature_extractor(model, batch)
                    all_features.append(feat)

        if is_eval:
            # Stack all batches: (Total_B, M, D)
            stacked_features = torch.cat(all_features, dim=0)
            
            mean = torch.mean(stacked_features, dim=0)
            var = torch.var(stacked_features, dim=0)
            
            if torch.isnan(mean).any() or torch.isnan(var).any():
                raise RuntimeError("NaN detected in final feature statistics")
            
            return model, mean, var

    raise RuntimeError("Training loop failed to reach evaluation phase")
