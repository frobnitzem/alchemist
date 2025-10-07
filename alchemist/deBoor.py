""" deBoor calculation - reference values:
gnuplot> B0(x) = x>=0 ? (x < 1 ? 1 : 0) : 0
gnuplot> plot [-0.5:4.5] B0(x)
gnuplot> B1(x) = x*B0(x) + (2-x)*B0(x-1)
gnuplot> plot [-0.5:4.5] B1(x)
gnuplot> B2(x) = (x*B1(x) + (3-x)*B1(x-1))/2
gnuplot> plot [-0.5:4.5] B2(x)
gnuplot> B3(x) = (x*B2(x) + (4-x)*B2(x-1))/3
gnuplot> plot [-0.5:4.5] B3(x)

gnuplot> print B2(0.1), B2(1.1), B2(2.1)
0.005 0.59 0.405
gnuplot> print B3(0.1), B3(1.1), B3(2.1), B3(3.1)
0.000166666666666667 0.221166666666667 0.657166666666667 0.1215

#print(deBoor(0.1, 0))
#print(deBoor(0.1, 1))
#print(deBoor(0.1, 2))
#print(deBoor(0.1, 3))
"""

import numpy as np

def deBoor(p: int) -> np.ndarray:
    """Use deBoor's recursion to symbolically
       evaluate B_p(x+i) for i=0, 1, ..., p

    Arguments:
      * p: Degree of B-spline.

    Returns: List of p+1 np.poly1d-s.
    """

    b = [np.poly1d([1])] + [0]*p
    x = np.poly1d([1, 0])
    for r in range(1, p+1):
        for i in range(r, 0, -1):
            b[i] = (  (x+i)*b[i]
                      + (r+1-i-x)*b[i-1]
                   ) / r
        b[0] *= x/r
    return b

def deBoor_np(x: np.ndarray, p: int) -> np.ndarray:
    """Evaluates B_{i,p}(x) = B_p(x-i)
       for i=0, 1, ..., p

    Arguments
    ---------
    x: Position -- assumed to be in [0,1)
    p: Degree of B-spline.
    """

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    b = np.zeros(x.shape + (p+1,))
    b[..., 0] = 1.0
    i = np.arange(p+1)
    for r in range(1, p+1):
        b[..., 1:r+1] = (  (x[...,None]+i[1:r+1])*b[..., 1:r+1]
                         + (r+1-i[1:r+1]-x[...,None])*b[..., :r]
                        ) / r
        b[..., 0] = ( x*b[..., 0] ) / r
    return b

"""
import torch
def deBoor_torch(x: torch.Tensor, p: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    b = torch.zeros(x.shape + (p+1,), device=x.device)
    b[..., 0] = 1.0
    i = torch.arange(p+1, device=x.device)
    for r in range(1, p+1):
        b[..., 1:r+1] = (  (x[...,None]+i[1:r+1])*b[..., 1:r+1]
                         + (r+1-i[1:r+1]-x[...,None])*b[..., :r]
                        ) / r
        b[..., 0] = ( x*b[..., 0] ) / r
    return b

#print(deBoor(torch.tensor([0.1, 0.2]), 1))
#print(deBoor(torch.tensor([0.1, 0.2]), 2))
#print(deBoor(torch.tensor([0.1, 0.2], requires_grad=True, device="cuda"), 3))
"""

#r = torch.tensor([0.1, 0.2], requires_grad=True)
#u = deBoor(r, 3)
#import torchviz
#graph = torchviz.make_dot(u, params={'r':r})
#print(graph.source)

#to_arr = lambda x: np.array([p.coef for p in x])
#for i in range(7):
#    print(to_arr(deBoor(i)))
