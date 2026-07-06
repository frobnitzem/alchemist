import torch
import numpy as np

import matplotlib.pyplot as plt

# Iexp(Q) = s Ibkd(Q) + P(r) (A(Q)/\alpha)[ Icoh(Q)+Iinc(Q) ]
# s: scale factor for sample holder / empty cell backgnd
#
# Icoh(Q) = Nf^2(Q)[1+\rho\int sinc(x)... ]
# S(Q) = Icoh(Q)/Nf^2(Q), f(Q) = form factor, electron cloud around atom

# Eggert PRB 65: 174105, 2022.
# Fe-B-Si metallic glass density and structure 
# proposed standard.
#
# (0, r_det, r_a)
# phase = 2*pi/lm*distance
# distance = |r_a+[x,0,0]| + |r_det-r_a|
# approx x+|r_det| + r_{ax} - r_a.r_det/|r_det|
#        = x+|r_det| - r_a . (\hat r_det - \hat x)
#  (linear expansion in r_a)
# 
# use typical Synchrotron
# λ = 0.41222 Å
#
# Note: Poisson detection probability is
# # exp(-lm) lm^n/n! = exp(-lm) [ 1, lm, lm^2/2, ...]
# but, if signal is 0 or >0 detection, probabilities are
# p_0 = exp(-lm), p_1 = 1-exp(-lm)

import math
from examples.GaAs import GaAs
from scattering import intensity, rotation_grid, rot_mat

xtalSize = 0.5 # mm
total_flux = 5e9
bea_size_mm = 0.01
detector_distance = 200 # mm
beam_centre = (170, 170) # mm
#pixel_size = (0.08854, 0.08854) # mm
#image_size = (3840, 3840)
pixel_size = (0.08854*3840/500, 0.08854*3840/500) # mm
image_size = (500,500)

nelec = {"Ga": 31, "As": 33}
# b_Ga ≈ 31 * 2.82 fm
# b_As ≈ 33 * 2.82 fm
# since r_e ≈ 2.82 fm

device = "cpu"
def main():
    array = lambda x: torch.tensor(x, dtype=torch.float32, device=device)

    names, crds = GaAs.mk_lattice(7, 7, 7)
    crds = array(crds).unsqueeze(0)
    # can rotate coords
    # crds = (crds[None,:,None,:]*R[:,None,:,:]).sum(-1)
    b = 2.82*array([nelec[n] for n in names]).unsqueeze(0)

    r2 = (crds*crds).sum(-1)
    mask = r2 < 20**2
    b = b[mask]
    crds = crds[mask]

    #print(crds)
    k0 = 2*math.pi/0.41222 # 1/Ang
    lattice = np.ndindex(image_size[0],image_size[1],1)
    detector = (array(list(lattice))+0.5) \
                    *array(list(pixel_size)+[0]) \
                - array(list(beam_centre)+[0])
    # detector pixel locations
    # detector and detector_distance just need to be consistent units
    rk = (detector**2+detector_distance**2)**0.5
    k = ((detector + array([0,0,detector_distance]))/rk \
          - array([0,0,1])) * k0
    if False:
        rgrid, _ = rotation_grid()
        R = rot_mat(rgrid)[:5]
        k = (k[None,:,None,:]*R[:,None,:,:]).sum(-1)
        # adds a "batch" axis to k

    print(crds.shape, k.shape)
    I = intensity(crds, b, k, 3.0).reshape((-1,)+image_size)
    print(I.shape)
    I = I.sum(0)
    I = torch.log(I+1e-10)
    plt.imshow(-I.cpu().numpy(), cmap='grey')
    plt.show()

if __name__=="__main__":
    main()
