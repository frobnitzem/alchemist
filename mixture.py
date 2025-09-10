import torch
from torch import nn

def index_last(x, z):
    # z are indices choosing along the last dimension of x
    # Outputs a subset of x with size z.shape.
    #
    # Assumes:
    #   z.shape == x.shape[:-1]
    #   z.min() >= 0
    #   z.max() < x.shape[-1]
    #
    if len(z.shape) == 0:
        return x[z]
    return torch.gather(x, -1, z.unsqueeze(-1)).squeeze(-1)

class MixtureNormal(nn.Module):
    """ Mixture Normal is an embedding that takes categories
        to Gaussian-s, centered on the emmbedding vectors.
    """
    def __init__(self, categories: int, dim: int, v2) -> None:
        super().__init__()
        self.embed = nn.Embedding(categories, dim)
        self.categories = categories
        self.dim = dim

        self.v2 = torch.asarray(v2)

        # variance-squared must be either zero or one-dimensional
        if len(self.v2.shape) == 1:
            assert len(self.v2) == dim
        else:
            assert len(self.v2.shape) == 0

        self.normal = torch.distributions.normal.Normal(0, v2**0.5)

    def alt_log_pzh(self, z, h):
        assert z.shape == h.shape[:-1], "Batch sizes must match"
        assert h.shape[-1] == self.dim, "Dimensions must match"

        # simpler, but numerically unstable calcuation of log_pzh
        dh = h[...,None,:] - self.embed.weight # batch x cat x dim
        arg1 = -0.5*(dh*dh/self.v2).sum(-1) # batch x cat
        x = nn.functional.softmax(arg1, dim=-1)
        #x = torch.exp(arg1)
        #x = x / x.sum(-1)[...,None]
        xz = index_last(x, z)
        return torch.log( xz )

    def calc_log_pzh(self, z, h):
        # log(P(z|h)) = log(q(h|z) q(z)) - log(\sum_k q(h|k) q(k))
        #  = -log(1 + \sum_{k\ne z} exp((h-[h_k+h_z]/2)\cdot(h_k-h_z)/sigma^2) q(k)/q(z))
        #  = -log(\sum_k exp((h-[h_k+h_z]/2)\cdot(h_k-h_z)/sigma^2) q(k)/q(z))

        assert z.shape == h.shape[:-1], "Batch sizes must match"
        assert h.shape[-1] == self.dim, "Dimensions must match"

        zk = self.embed(z).unsqueeze(-2) # batch x 1 x dim
        c = h.unsqueeze(-2) - 0.5*(self.embed.weight + zk) # batch x cat x dim
        d = self.embed.weight - zk           # batch x cat x dim
        arg = (c*d/self.v2).sum(-1)          # batch x cat
        y = torch.exp(arg)
        B = -torch.log( y.sum(-1) )

        return B

    def loss_prior(self):
        """ - log P(theta | I)
        """
        return torch.log(self.v2).sum() \
               - self.calc_log_pzh(torch.arange(self.categories),
                                   self.embed.weight).sum()

    def forward(self, z):
        """ Sample h values corresponding to a given z.
        """
        h = self.embed(z)
        return h + self.normal.sample(h.shape)

#def calc_h0(h):
#    return h*2.0 - 1.0, 2.0
#    #dh = F.relu( u(h)@A.T + b )

# L = -\log P(h) - \log P(z|h) + \log q(h|z)

#P(z|h) = \frac{q(h|z) q(z)}{\sum_k q(h|k) q(k)} \\
#   = \frac{1}{1 + \sum_{k\ne z}
#exp((h-[h_k+h_z]/2)\cdot(h_k-h_z)/sigma^2) q(k)/q(z)}

def test_log_pzh():
    mean = torch.Tensor(
     [[ 1., 2., 0.],
      [-2., 0., 0.],
      [ 0., 0., 2.]]
    )
    M = MixtureNormal(3, 3, 0.5)
    with torch.no_grad():
        M.embed.weight[:] = mean

    u = torch.Tensor([[0.0, 0.5, 0.5],
                      [1.1, -0.2, 0.1],
                      [0.9, 2.0, -0.1]])

    #idx = torch.asarray([0,1,2], dtype=torch.int)
    idx = torch.arange(3)
    P = M.calc_log_pzh(idx, u)
    P2 = M.alt_log_pzh(idx, u)
    err = torch.abs(P-P2).max()
    print(P, P2, err)
    assert err.item() < 1e-4

    x = M.calc_log_pzh(idx[0], u[0])
    y = M.calc_log_pzh(idx[1], u[1])
    z = M.calc_log_pzh(idx[2], u[2])
    ans = torch.Tensor([x,y,z])
    err = torch.abs(ans-P).max()
    print(ans, P, err)
    assert err.item() < 1e-10

    print(M.loss_prior())

    print(idx.shape)
    print(M(idx).shape)
