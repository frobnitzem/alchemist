""" Inverse Radial Basis, an expansion permitting origin singularities
of order 1/r. The metric is <a|b> = \int a(r) b(r) r^2 dr

f(r) = r^{-1} \sum_{i=0}^N \Theta(r_i < r \le r_{i+1}) (c_i + m_i r)
     = \sum_{i=0}^{N-1} (r_i f(r_i)) b_i(r)
"""
import torch

""" # integrals
import sympy
r = sympy.Symbol('r')
r0 = sympy.Symbol('r0')
r1 = sympy.Symbol('r1')
r2 = sympy.Symbol('r2')
dr0 = r1-r0
dr1 = r2-r1
b0_R = (r1-r)/dr0
b0_R = (r1/r-1)/dr0
b1_R = (r2/r-1)/dr1
b1_L = (1-r0/r)/dr0
b2_L = (1-r1/r)/dr1

# <1, b_1>
(sympy.integrate(b1_L*r*r, (r, r0,r1))+sympy.integrate(b1_R*r*r, (r,r1,r2))).simplify()
# -r0**2/6 - r0*r1/6 + r1*r2/6 + r2**2/6

# <b_1, b_1>
(sympy.integrate(b1_L*b1_L*r*r, (r, r0,r1))+sympy.integrate(b1_R*b1_R*r*r, (r,r1,r2))).simplify()
# -r0/3 + r2/3

# <b_1, b_2>
sympy.integrate(b1_R*b2_L*r*r, (r, r1,r2)).simplify()
# -r1/6 + r2/6
"""

