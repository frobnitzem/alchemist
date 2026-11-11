import torch
from alchemist.flows import Q

class AlchemistDataset:
    def __init__(self, batch_size=1, na=216, dim=2, sigma=1.0):
        self.batch_size = batch_size
        self.na = na
        self.dim = dim
        self.sigma = sigma
        self.normal = torch.distributions.normal.Normal(0, 1)

    def __len__(self):
        # Return a fixed number of batches for the dataset
        return 100

    def __iter__(self):
        for _ in range(len(self)):
            # Generate a batch of coordinates r
            # r: (B, N, D)
            r = Q(self.normal.sample((self.batch_size, self.na, self.dim))) * self.sigma

            # Generate a batch of momenta/latent coordinates p
            # p: (B, N, D)
            p = Q(self.normal.sample((self.batch_size, self.na, self.dim)))

            yield {
                'r': r,
                'p': p,
                't': torch.full((self.batch_size,), 0.0)
            }


class IsingTargetDataset:
    """Samples x=(r, p) from the target pi(r, p) ~ exp(-beta H1) for a small
    lattice with exactly tabulated species probabilities (see ising.exact_table).

    s ~ table; r_i ~ N(0, sigma_r^2/beta)^2 with its two components swapped when
    argmax(r_i) != s_i (exact conditional of the Gaussian on the argmax region,
    by symmetry); p ~ N(0, M/beta).  `t` starts at t0 = n_layers*dt so the
    inverse flow sees the same times as the forward flow.
    """
    def __init__(self, pdb_path, box, cuts, MU, E1, E2=None, KT=1.0,
                 batch_size=100, n_batches=100, sigma_r=1.0, M=1.0, t0=0.0):
        from alchemist.ising import exact_table
        self.P, _ = exact_table(pdb_path, box, cuts, MU, E1, E2, beta=1.0 / KT)
        self.N = int(round(len(self.P).bit_length() - 1))
        self.batch_size, self.n_batches, self.t0 = batch_size, n_batches, t0
        self.sr, self.sp = sigma_r * KT ** 0.5, (M * KT) ** 0.5

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            idx = torch.multinomial(self.P, self.batch_size, replacement=True)
            s = (idx[:, None] >> torch.arange(self.N)) & 1                  # (B,N)
            r = torch.randn(self.batch_size, self.N, 2) * self.sr
            wrong = r.argmax(-1) != s
            r = torch.where(wrong[..., None], r.flip(-1), r)
            p = torch.randn(self.batch_size, self.N, 2) * self.sp
            yield {'r': r, 'p': p, 't': torch.full((self.batch_size,), self.t0)}
