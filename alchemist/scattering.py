import torch
from scipy.integrate import lebedev_rule

# TODO?: use claude/codex api to write this code
#  - langchain / gemini cli
#  - idempotent file set as in/out params
#  
#
# B-fac: sigma_j = B_j/(8\pi^2)
# 
# Assuming elastic scattering, k = k_out - k_inc
# with |k_out| = |k_inc|
# then k(k_out) = k_out - |k_out| z
#
# Note: (R r) . k = r . (R^T k)

def intensity(r, q, k, sigma=None):
    """ Compute the intensity at k given by Gaussians
        of width sigma at points, r.

    k.shape == Nk + (3,)
    r.shape == Nb + (Na, 3)

    output shape = Nb + Nk

    r' = r-r_a
    r  = r'+r_a
    F[rho](k) = N(sigma) \sum_a \int exp(-ik.r) exp(-0.5*(r-r_a)**2/sigma**2) dr
              = \sum_a exp(-ik.r_a) exp(-0.5*k**2*\sigma**2)
    returns |F[rho](k)|^2
    """
    assert k.shape[-1] == r.shape[-1]
    assert q.shape == r.shape[:-1]
    # Nk + (1, 3) x (..., (1 x len(k.shape)-1), Na, 3)
    for i in range(len(k.shape)-1):
    #    r = r.unsqueeze(-3)
        q = q.unsqueeze(-2)
    #fac = (k.unsqueeze(-2)*r).sum(-1) # ..., Nk, Na
    fac = torch.matmul(k, r.transpose(-1,-2))
    Mc = (q*torch.cos(fac)).sum(-1) # ..., Nk
    Ms = (q*torch.sin(fac)).sum(-1) # ..., Nk
    if sigma is None:
        return Mc*Mc+Ms*Ms

    fg = torch.exp(-0.5/sigma**2 * (k*k).sum(-1)) # Nk
    return (Mc*Mc+Ms*Ms)*fg


def rotation_grid(Nth = 8, leb_n = 13):
    z = (2*torch.arange(Nth)+1)/Nth - 1.0
    htheta = 0.5*torch.arccos(z)
    # TODO: more lebedev points / rules
    x, w = lebedev_rule(13)
    leb_w = torch.tensor(w) / w.sum()
    leb_r = torch.tensor(x.T)
    # r = [rx, ry, rz]
    # q = [cos(th/2)] + sin(th/2)*r
    Nl = len(leb_w)
    wt = leb_w[None,:].broadcast_to(Nth, Nl)/Nth
    ch = torch.cos(htheta)[:,None,None] \
            .broadcast_to(Nth,len(leb_w),1)
    sh = torch.sin(htheta)[:,None,None]
    return torch.cat([sh*leb_r, ch], -1) \
                .reshape((Nth*Nl, 4)), \
           wt.reshape(Nth*Nl)

def test_intensity(shape):
    import math
    shape1 = shape + (3,)
    Nd = math.prod(shape1)
    k = torch.arange(Nd).reshape(shape1)/Nd
    r = torch.randn(4,100,3)
    b = torch.ones(4,100)
    intens = intensity(r, b, k)
    print(intens.shape, shape)
    assert intens.shape == (4,) + shape
    intens = intensity(r, b, k, 0.1)
    assert intens.shape == (4,) + shape

def rot_mat(x):
    """ Create a rotation matrix from a quaternion
    input   x : (..., 4)
    returns R : (..., 3, 3)
    """
    x00 = x[..., 0]*x[..., 0]
    x01 = x[..., 0]*x[..., 1]
    x02 = x[..., 0]*x[..., 2]
    x11 = x[..., 1]*x[..., 1]
    x12 = x[..., 1]*x[..., 2]
    x22 = x[..., 2]*x[..., 2]
    return torch.stack([
              1 - 2*(x11+x22),
              2*(x01 - x[..., 3]*x[..., 2]),
              2*(x02 + x[..., 3]*x[..., 1]), #]
              2*(x01 + x[..., 3]*x[..., 2]),
              1 - 2*(x00+x22),
              2*(x12 - x[..., 3]*x[..., 0]), #]
              2*(x02 - x[..., 3]*x[..., 1]),
              2*(x12 + x[..., 3]*x[..., 0]),
              1 - 2*(x00+x11) #]
          ], -1).reshape(x.shape[:-1] + (3,3))

def test_rot_mat(Nk = 8):
    x, wt = rotation_grid(Nk)
    assert wt.shape[0] % Nk == 0
    assert len(wt.shape) == 1
    Na = wt.shape[0] // Nk

    assert x.shape == (Na*Nk, 4)
    R = rot_mat(x)
    assert R.shape == (Na*Nk, 3, 3)

    k = torch.arange(30).reshape((10,3))/30.0
    r = torch.randn(4,100,3)

    # (10,6*8,3,3)
    Rk = (R*k[:,None,None,:]).sum(-1)
    b = torch.ones_like(r[...,0])
    intens = intensity(r, b, Rk) # (10,Na*Nk,3)
    assert intens.shape == (4,10,Na*Nk)

def angular_integral(r, sigma=None, leb_n=13):
    x, w = lebedev_rule(leb_n)
    wt = torch.tensor(w)/w.sum()
    x = torch.tensor(x.T)
    b = torch.ones_like(r[...,0])
    intens = intensity(r, b, x, sigma)
    return torch.dot(wt, intens)

def test_angular_integral():
    r = torch.arange(30).reshape((10,3))/30.0
    # exp(-lm) lm^n/n! = exp(-lm) [ 1, lm, lm^2/2, ...] = exp(-lm), 1-exp(-lm)
    aint = angular_integral(r)
    print(aint, aint.shape)
    assert aint.shape == ()

if __name__ == "__main__":
    test_intensity( (10,) )
    test_intensity( (2,5) )
    test_rot_mat()
    test_angular_integral()
