import math
from pydantic import BaseModel, Field

import torch
from torch import nn

class Tetrahedron:
    x = [[],[],[],[]]
    wts = [0.25, 0.25, 0.25, 0.25]

class HermiteSpline(nn.Module):
    def __init__(self, dim, r1, r0=0.0, dr=1.0):
        super().__init__()
        self.dim = dim # dim-dimensional function range
        num_intervals = math.ceil( (r1-r0)/dr )
        self.knots = num_intervals*2+1
        self.theta = nn.Parameter(torch.randn(self.knots, dim))
        self.r0 = r0
        self.dr = dr

    def interpolate(self, r):
        # compute the values of the spline at the points, r
        # return shape == r.shape
        x = (r-self.r0)/self.dr
        idx = torch.ceil(2*x-1.5)
        print(idx)
        u = 1.5 + idx - 2*x # 0.0 <= u < 1.0
        # { x:ceil(2*x-1.5) = i} = (i+.5)/2 < x <= (i+1.5)/2
        # { x:ceil(2*x-0.5) = i} = (i-.5)/2 < x <= (i+0.5)/2
        # { x:ceil(2*x+0.5) = i} = (i-1.5)/2 < x <= (i-0.5)/2
        ldim = r.size(-1)
        bfn = torch.stack([
                #u*u, -2*x*x+6*x-3, (3-x)**2
                #u*u, -2*x*x+6*x-3, (3-x)**2
                #u*u, -2*(u+1)*(u+1)+6*(u+1)-3, (3-(u+2))**2
                u*u, (-2*u+2)*u+1, (1-u)**2
                ], -1).reshape(r.shape[:-1]+(3*ldim,))*0.5
        return torch.sparse_csr_tensor(
                crow_indices = (3*torch.arange(ldim+1)).\
                            broadcast_to(r.shape[:-1]+(ldim+1,)),
                            col_indices  = (idx.int()[...,None]+torch.arange(3)).reshape(bfn.shape),
                    values       = bfn,
                    size         = r.shape+(self.knots,))
        #return torch.sparse_csr_tensor(
        #        crow_indices = (torch.arange(0,4,3)).\
        #                    broadcast_to(r.shape+(2,)),
        #                    col_indices  = (idx.int()[...,None]+torch.arange(3)).reshape(r.shape + (3,)),
        #            values       = bfn.reshape(r.shape + (3,)),
        #            size         = r.shape+(1,self.knots))

    def forward(self, r):
        # r is *batch
        x = (r-self.r0)/self.dr
        idx = torch.floor(x)
        u = x - idx
        idx = idx.int()

        #theta = self.theta[idx*2:idx*2+3]
        bfn = self.basis_fns(u).unsqueeze(-1) # 3, *batch, 1
        return bfn[0]*self.theta[idx*2]\
                + bfn[1]*self.theta[idx*2+1]\
                + bfn[2]*self.theta[idx*2+2]

    def basis_fns(self, u):
        return torch.stack([
                1.0+u*(2.0*u-3.0), 4.0*u*(1.0-u), u*(2.0*u-1.0)
            ])

class PointEmbedding(BaseModel):
    """ PointEmbedding Parameters
    """

    num_features: int # number of features (channels) per node
    num_abf: int # number of angular basis functions
    num_rbf: int # number of radial basis functions (csplines)

    cutoff_radius: float # cutoff radius
    spacing_exponent: float = 1.0 # r_i = cutoff_radius * (i/num_rbf)**spacing_exponent

    def dim(self):
        # Note: output 
        return (self.num_features, self.num_abf, self.num_rbf)

    def radial_grid(self):
        """ Return the radial grid (length num_rbf+1).
        """
        idx = torch.arange(self.num_rbf+1)/self.num_rbf
        return self.cutoff_radius * idx**self.spacing_exponent

    def __call__(self, r):
        # r is (..., 3)
        # returns an array of shape (...) + self.dim()
        pass

# Ideally, radial splines will be built out of sparse tensors.

# https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch

"""
t = torch.tensor([[[1., 0., 0, 0, 0],
                   [2., 3., 0, 0, 0]],
                  [[0, 0, 4., 0, 0],
                   [0, 0, 0, 5., 6.]]])
print(t)
print( t.to_sparse_csr() )

s = torch.sparse_coo_tensor(
        [[0, 0, 0, 1, 1, 1],
         [0, 1, 1, 0, 1, 1],
         [0, 0, 1, 2, 3, 4]] , 
        torch.Tensor([1., 2., 3., 4., 5., 6.]),
        (2, 2, 5))

print(s.to_dense())

s = torch.sparse_csr_tensor(
        crow_indices =torch.IntTensor([[0,1,3],
                                       [0,1,3]]),
        col_indices = torch.IntTensor([[0,0,1],
                                       [2,3,4]]),
        values      = torch.Tensor([[1., 2., 3.],
                                    [4., 5., 6.]]),
        size=(2, 2, 5))

print(s.to_dense())

z = 0
b = 1

t = torch.tensor([[z, z, b, z, z],
                  [z, b, z, z, z]])
print(t.shape)
print( t.to_sparse_csr() )
#tensor(crow_indices=tensor([0, 1, 2]),
#       col_indices=tensor([2, 1]),
#       values=tensor([1, 1]), size=(2, 5), nnz=2, layout=torch.sparse_csr)

t = torch.tensor([[[z,z,b,z,z], [z,b,z,z,z]],
                  [[b,z,z,z,z], [z,z,z,z,b]]])
print(t.to_sparse_csr())
#tensor(crow_indices=tensor([[0, 1, 2],
#                            [0, 1, 2]]),
#       col_indices=tensor([[2, 1],
#                           [0, 4]]),
#       values=tensor([[1, 1],
#                      [1, 1]]), size=(2, 2, 5), nnz=2, layout=torch.sparse_csr)

z = [[0,0],[0,0]]
b = [[1.,2.],[3.,4.]]

s = torch.sparse_csr_tensor(
        crow_indices =torch.IntTensor([0,1,2]),
        col_indices = torch.IntTensor([2,1]),
        values      = torch.Tensor([b, b]),
        size=(2, 5, 2, 2))
print(s.to_dense())
"""
