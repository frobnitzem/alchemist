import torch
from torch import nn

class MixtureNormal(nn.Module):
    """ Mixture Normal is an embedding that takes categories
        to Gaussian-s, centered on the emmbedding vectors.
    """
    def __init__(self, mean, v2):
        super().__init__()
        assert len(mean.shape) == 2

        self.mean = mean.unsqueeze(1) # add a batch dim. ~> Nz x 1 x d
        self.v2 = torch.asarray(v2)

        if len(self.v2.shape) == 1:
            assert len(self.v2) == mean.shape[1]
        else:
            assert len(self.v2.shape) == 0

        self.normal = torch.distributions.normal.Normal(0, v2**0.5)

    def alt_log_pzh(self, z, h):
        if len(h.shape) != 1:
            assert len(z) == len(h), "Batch sizes must match"

        # simpler, but numerically unstable calcuation of log_pzh
        dh = h-self.mean # Nz x batch x d
        arg1 = -0.5*(dh*dh/self.v2).sum(-1) # Nz x batch
        x = torch.exp(arg1)
        x = x / x.sum(0)
        if len(h.shape) != 1:
            xz = torch.asarray([x[zk,k] for k,zk in enumerate(z)])
        else:
            xz = x[z,0]
        return torch.log( xz )

    def calc_log_pzh(self, z, h):
        # log(P(z|h)) = log(q(h|z) q(z)) - log(\sum_k q(h|k) q(k))
        #  = -log(1 + \sum_{k\ne z} exp((h-[h_k+h_z]/2)\cdot(h_k-h_z)/sigma^2) q(k)/q(z))

        # Create c and d (Nz x batch x d)
        if len(h.shape) != 1:
            assert len(z) == len(h), "Batch sizes must match"
        c = h - 0.5*(self.mean+self.mean[z,0])
        d = self.mean - self.mean[z,0] # Nz x batch x d
        arg = (c*d/self.v2).sum(-1)
        y = torch.exp(arg)
        B = -torch.log( y.sum(0) )

        if len(h.shape) == 1:
            return B[0]
        return B

    def loss_prior(self):
        """ - log P(theta | I)
        """
        return torch.log(self.v2).sum() \
               - self.calc_log_pzh(torch.arange(len(self.mean)),
                       self.mean[:,0]).sum()

    def forward(self, z):
        """ Sample h values corresponding to a given z.
        """
        #self.embeds = nn.Embedding(atom_types, embedding_dim) # 3 element types into a 5D vector
        #r = self.embeds(samples)
        h = self.mean[z,0]
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
    M = MixtureNormal(mean, 0.5)

    u = torch.Tensor([[0.0, 0.5, 0.5],
                      [1.1, -0.2, 0.1],
                      [0.9, 2.0, -0.1]])

    idx = torch.asarray([0,1,2], dtype=torch.int)
    P = M.calc_log_pzh(idx, u)
    P2 = M.alt_log_pzh(idx, u)
    err = torch.abs(P-P2).max()
    print(P, P2, err)
    assert err.item() < 1e-4

    x = M.calc_log_pzh(0, u[0])
    y = M.calc_log_pzh(1, u[1])
    z = M.calc_log_pzh(2, u[2])
    ans = torch.Tensor([x,y,z])
    err = torch.abs(ans-P).max()
    print(ans, P, err)
    assert err.item() < 1e-10

    print(M.loss_prior())

    print(idx.shape)
    print(M(idx).shape)

if __name__=="__main__":
    test_log_pzh()
