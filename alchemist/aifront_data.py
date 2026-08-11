import torch
from torch.utils.data import Dataset
from alchemist.flows import Q

class AlchemistDataset(Dataset):
    def __init__(self, batch_size=10, na=216, dim=2, sigma=1.0):
        self.batch_size = batch_size
        self.na = na
        self.dim = dim
        self.sigma = sigma
        self.normal = torch.distributions.normal.Normal(0, 1)

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        # Return a full batch as requested
        r = Q(self.normal.sample((self.batch_size, self.na, self.dim))) * self.sigma
        p = Q(self.normal.sample((self.batch_size, self.na, self.dim)))
        return {
            'r': r,
            'p': p,
            't': torch.full((self.batch_size,), 0.0)
        }
