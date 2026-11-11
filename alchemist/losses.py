import math
import torch
from alchemist.neighbors import compute_neighbor_masks, get_neighbor_indices, assemble_neighbor_features
from alchemist.analysis import identity_prob, interaction_energy
from alchemist.pdbs import read_pdb_coords


class KLLoss:
    """Forward-KL loss for the flow (r0, p0) -> (r, p), logJ = log|det df|.

    Target  pi(r, p) = exp(-beta*H1(r, p)) / Z,  beta = 1/KT, with

        H0(r, p) = (|r|^2 / sigma_r^2 + |p|^2 / M) / 2
        H1(r, p) = H0(r, p) + U(identity(r))
        U        = sum_i [ -mu.prob_i + NN1, NN2 pair terms ]     (E - mu.N)

    r holds element logits, p the conjugate momenta.  prob_i = identity_prob(r_i):
    softmax(r_i), or (identity="argmax") the argmax one-hot with a
    straight-through softmax gradient.

    loss = E_x0[ beta*H1(f(x0)) + log P0(x0) - logJ ]  =  KL(q||pi) - log Z.
    The latent is r0 ~ N(0, latent_sigma_r^2), p0 ~ N(0, latent_sigma_p^2).
    """
    def __init__(self, pdb_path, box, cuts, MU, E1, E2=None, KT=1.0,
                 sigma_r=1.0, M=1.0, identity="argmax",
                 latent_sigma_r=1.0, latent_sigma_p=1.0):
        coords = read_pdb_coords(pdb_path)
        box = torch.tensor(box, dtype=torch.float32)
        masks = compute_neighbor_masks(coords, box, cuts)
        self.neighborlists = get_neighbor_indices(masks)
        self.MU = torch.as_tensor(MU, dtype=torch.float32)
        self.E1 = torch.as_tensor(E1, dtype=torch.float32)
        self.E2 = None if E2 is None else torch.as_tensor(E2, dtype=torch.float32)
        self.sizes = tuple(n.shape[1] for n in self.neighborlists)
        self.KT = KT
        self.sigma_r = sigma_r
        self.M = M
        self.identity = identity
        self.latent_sigma_r = latent_sigma_r
        self.latent_sigma_p = latent_sigma_p

    @staticmethod
    def _log_gauss(x, sigma):
        n = x[0].numel()
        quad = (x ** 2).sum(dim=(-2, -1)) / (2 * sigma ** 2)
        return -quad - 0.5 * n * math.log(2 * math.pi * sigma ** 2)

    def components(self, predictions, batch):
        """Per-sample terms of the loss, each shape (B,)."""
        x, logJ, _ = predictions
        dev = x['r'].device
        nbrs = [n.to(dev) for n in self.neighborlists]
        mu, e1 = self.MU.to(dev), self.E1.to(dev)
        e2 = None if self.E2 is None else self.E2.to(dev)

        prob = assemble_neighbor_features(identity_prob(x['r'], self.identity), nbrs)
        U = interaction_energy(prob, mu, e1, e2, self.sizes).sum(1)
        H0 = 0.5 * ((x['r'] ** 2).sum(dim=(-2, -1)) / self.sigma_r ** 2
                    + (x['p'] ** 2).sum(dim=(-2, -1)) / self.M)
        logP0 = (self._log_gauss(batch['r'], self.latent_sigma_r)
                 + self._log_gauss(batch['p'], self.latent_sigma_p))
        beta = 1.0 / self.KT
        return {'betaH0': beta * H0, 'betaU': beta * U,
                'logP0': logP0, 'logJ': logJ}

    def __call__(self, predictions, batch):
        c = self.components(predictions, batch)
        return (c['betaH0'] + c['betaU'] + c['logP0'] - c['logJ']).mean()


class MLELoss(KLLoss):
    """Backward (maximum-likelihood) loss: -E_{x~pi}[log q(x)].

    Used with a model run in the inverse direction (flows.Directed(inverse=True)),
    so predictions = (f^-1(x), logJ_inv, info) and
        log q(x) = log P0(f^-1(x)) + logJ_inv.
    Batches hold samples x = (r, p) from the target pi.  Only the latent
    density is needed; no energy is evaluated.
    """
    def __call__(self, predictions, batch):
        z, logJinv, _ = predictions
        logq = (self._log_gauss(z['r'], self.latent_sigma_r)
                + self._log_gauss(z['p'], self.latent_sigma_p) + logJinv)
        return -logq.mean()
