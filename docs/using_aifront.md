# Adopting Alchemist to AIfront Framework

This tutorial documents the process of migrating the `alchemist/experiment.py` script to the AIfront configuration-driven framework.

## Goals
- Decompose `experiment.py` into YAML specifications.
- Implement data loading, model instantiation, and training using `ai` CLI.
- Replicate the energy and neighbor analysis plots.

## Progress

### Step 1: Data Loading
The original script used a generator function to yield batches. In AIfront, we move this to a `torch.utils.data.Dataset`.

**Before (`experiment.py`):**
```python
def data_gen():
    while True:
        x = {
            'r': Q(normal.sample((BATCH_SIZE, NA, DIM))) * SIGMA, 
            'p': Q(normal.sample((BATCH_SIZE, NA, DIM))),
            't': torch.tensor(0.0)
        }
        yield x
```

**After (`alchemist/aifront_data.py` & `data.yaml`):**
```python
# alchemist/aifront_data.py
class AlchemistDataset(Dataset):
    def __getitem__(self, idx):
        r = Q(self.normal.sample((self.batch_size, self.na, self.dim))) * self.sigma
        p = Q(self.normal.sample((self.batch_size, self.na, self.dim)))
        return {'r': r, 'p': p, 't': torch.full((self.batch_size,), 0.0)}
```
```yaml
# data.yaml
_target_: torch.utils.data:DataLoader
dataset:
  _target_: alchemist.aifront_data:AlchemistDataset
  batch_size: 10
  na: 216
  dim: 2
  sigma: 1.0
batch_size: null
shuffle: true
```

### Step 2: Model Definition
The model was previously instantiated in Python. We now use a YAML spec to define the `MultiStep` flow and a wrapper class to handle the neighbor feature expansion.

**Before (`experiment.py`):**
```python
glow = GlowBlock(dim=DIM, dt=0.001, hidden_dims=HIDDEN_DIMS, data_size=17, data_expansion=data_expansion)
flow = MultiStep(glow, N_STEPS_FLOW)
```

**After (`alchemist/aifront_utils.py` & `module.yaml`):**
```python
# alchemist/aifront_utils.py
class NeighborExpansion(nn.Module):
    def __init__(self, pdb_path, box, cuts):
        # ... computes neighborlists and registers as buffers ...
    def __call__(self, features):
        res = assemble_neighbor_features(features, self.neighborlists)
        return res.view(B, N, M * D)
```
```yaml
# module.yaml
_target_: alchemist.flows:MultiStep
step:
  _target_: alchemist.flows:GlowBlock
  dim: 2
  dt: 0.001
  hidden_dims: [8, 8, 8]
  data_size: 17
  data_expansion:
    _target_: alchemist.aifront_utils:NeighborExpansion
    pdb_path: examples/GaAs/GaAs.pdb
    box: [17.25, 17.25, 17.25]
    cuts: [2.5, 4.5]
n: 10
```

### Step 3: Training Setup
The loss function and optimizer are now decoupled from the main script and defined in a dedicated loss module and a trainer YAML.

**Before (`experiment.py`):**
```python
def loss_KL(x0, x, logJ):
    # ... energy calculations ...
    loss = 1 / KT * (energy.sum(1) - energy0.sum(1)) - logJ
    return loss.mean()

optimizer = optim.Adam(flow.parameters(), lr=LR)
```

**After (`alchemist/losses.py` & `trainer.yaml`):**
```python
# alchemist/losses.py
class KLLoss:
    def __call__(self, predictions, batch):
        x, logJ, info = predictions
        # ... device-aware energy calculations ...
        loss = 1 / self.KT * (energy.sum(1) - energy0.sum(1)) - logJ
        return loss.mean()
```
```yaml
# trainer.yaml
Trainer:
  max_epochs: 30
Lightning:
  loss_fn:
    _target_: alchemist.losses:KLLoss
    # ... parameters ...
  optimizer_name: "torch.optim:Adam"
  optimizer_kwargs:
    lr: 0.001
```
