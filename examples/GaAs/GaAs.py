# Construct a repeated GaAs lattice
# for visualization / input.

import numpy as np

unit = np.array( [
        [ 0.0, 0.0, 0.0],
        [ 0.0, 0.5, 0.5],
        [ 0.5, 0.0, 0.5],
        [ 0.5, 0.5, 0.0]
       ])
unit2 = unit+np.array([0.25,0.25,0.25])
print(unit2)

scale = 5.75
unames = ["Ga"]*4 + ["As"]*4
ucrds = np.vstack([unit, unit2])
ucrds -= np.floor(ucrds+0.5) # wrap closest to origin

def alt(n): # 0, 1, -1, 2, -2, 3, -3, 4, -4, 5, ...
    return (n != 0)*(2*(n%2) - 1)*( (n+1)//2 )

def mk_lattice(nx, ny, nz):
    L = np.eye(3)
    names = []
    crds = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                names.extend(unames)
                crds.append(ucrds+L[0]*alt(i)+L[1]*alt(j)+L[2]*alt(k))

    return names, scale*np.vstack(crds)

def write_pdb(names, crds) -> str:
    fmt = "ATOM  %5d  %-3s%4s %5d    %8.3f%8.3f%8.3f  1.00  1.00\n"
    resname = "XTL"
    resid = 1
    return "".join(
                fmt%(i+1,name,resname,resid,x[0],x[1],x[2]) \
            for i,(name,x) in enumerate(zip(names, crds)))

with open("GaAs.pdb", "w") as f:
    f.write( write_pdb(*mk_lattice(3, 3, 3)) )
