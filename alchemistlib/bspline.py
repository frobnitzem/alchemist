import torch
from torch import nn
import numpy as np

from deBoor import deBoor

class BSpline(nn.Module):
    """ The BSpline module transforms from coordinates
        to basis representations in p-degree Cardinal B-splines.

        for example [x1, x2] with p=1
        becomes [ [0, 0, 1-(ceil(x1)-x1), ceil(x1)-x1],
                  [0, 1-(ceil(x2)-x2),   ceil(x2)-x2, 0]
                ]

        Technically, BSpline(N, p)(r) adds a last-dimension to r
        of size N, where B_p(r-0), B_p(r-1), ..., B_p(r-(N-1))
        are stored.
    """
    def __init__(self, N, p=1):
        assert p >= 0, "Invalid polynomial order"
        assert p < 7, "Inadvisable polynomial order"
        super().__init__()
        self.N = N
        self.p = p
        self.M = torch.tensor(np.array([x.coef for x in deBoor(p)]).T)

    def forward(self, x):
        # compute the values of the spline at the points, x
        idx = torch.ceil(x)
        u   = idx - x
        
        sh = x.shape + (self.p+1,)
        vals = self.M[0].broadcast_to(sh)
        for i in range(1, self.p+1):
            vals = vals*u[..., None]  + self.M[i]
        
        #return torch.sparse_coo_tensor(
        #        indices      = idx[...,None] \
        #                        +torch.arange(self.p+1)-(self.p+1)//2,
        #        values       = vals,
        #        size         = x.shape+(self.N,))
        P = self.p+1
        ldim = x.size(-1)
        vals = vals.reshape(x.shape[:-1]+(ldim*P,))
        return torch.sparse_csr_tensor(
                crow_indices = (P*torch.arange(ldim+1)).\
                            broadcast_to(x.shape[:-1]+(ldim+1,)),
                            col_indices  = (idx.int()[...,None]+torch.arange(P)).reshape(vals.shape),
                    values       = vals,
                    size         = x.shape+(self.N,))

def test():
    B = BSpline(10, p=1)
    x = torch.tensor([[3.0, 3.1, 3.5], [1.9, 2.01, 2.2]])
    y = B(x)
    print(y.shape)
    #print(y.to_dense())
    y0 = B(x[:,0])
    y1 = B(x[:,1])
    print(y0)
    print(y1)
    # outer products don't work...
    #y01 = torch.einsum('pi,pj->pij', y0, y1)
    #print(y01)

class SparseSplineTensor:
    """ Sparse multi-dimensional spline tensor value.
    """
    def __init__(self, idx, vals, spls):
        self.idx = idx
        self.vals = vals
        self.spls = spls

    def to_dense(self):
        ans = 0
        #for i, j

class JointSpline(nn.Module):
    """ Creates an outer product of a list of splines,

    joint_spline(r1, r2) = B1(r1)[...,:,None]*B2(r2)[...,None,:]
    has indices (..., i, k)

    Which can occupy a sub-range of i in i1(r1):j1(r1)
    and j in i2(r2):j2(r2).

    Assembling the matrix can be done by masking and slicing
    i1:j1, B1(r) = spline1(r1)
    i2:j2, B2(r) = spline2(r2)
    (i1:j1, i2:j2), B12 = spline12(r1, r2)
    """
    def __init__(self, *spl):
        self.spl = spl

    def get_dims(self, dims, nx: int):
        ns = len(self.spl)
        if dims is None:
            assert nx >= ns, "Not enough dimensions to apply inverse."
            dim0 = nx - ns
            return tuple(range(dim0, nx))

        assert len(dims) == ns
        return dims

    def apply_inverse(self, x, dims=None):
        """ Apply the inverses of component splines to
            the specified dims (len(dims) == len(self.spl)).
            If dims is None, the last len(dims) of x
            are transformed.
        """
        dims = self.get_dims(dims, len(x.shape))

        for dim, s in zip(dims, self.spl):
            x = s.apply_inverse(x, dim)
        return x

    def forward(self, x, dims=None):
        """ Spline the values along the last dimension of x
            with the specified dims ~> self.spl mapping
            (self.spl[0] applies to x[..., dims[0]], etc.).

            Two items are returned,

            - indices: int with shape (..., len(self.spl), 2)
            - values:  float with shape (..., j0-i0, j1-i1, ...,)

            where i0,j0 are the indices from the first array.
        """
        dims = self.get_dims(dims, x.size(-1))

        idxs = []
        vals = torch.ones_like(x[...,0])
        for i, (dim, s) in enumerate(zip(dims, self.spl)):
            idx, val = s(x[..., dim])
            for k in range(i): # broadcast val to correct shape
                val = val.unsqueeze(-2)
            idxs.append(idx)
            vals = vals[...,None]*val
        return torch.stack(idxs, -2), vals
