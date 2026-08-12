#!/usr/bin/env python3
"""Build an HCP crystal from an orthorhombic (rectangular) unit cell.

The conventional HCP primitive vectors are not orthogonal.  This module uses
an equivalent four-atom orthorhombic cell instead:

    Lx = a
    Ly = sqrt(3) * a
    Lz = c
    alpha = beta = gamma = 90 degrees

Repeating that cell therefore gives a rectangular simulation box, so periodic
minimum-image displacements can be computed component by component.

Example
-------
    python hcp_rectangular.py --nx 3 --ny 3 --nz 3 \
        --a 3.2094 --c-over-a 1.6235 --basis-names Mg \
        --output Mg_hcp_3x3x3.pdb --save-npz

To assign different labels to the four basis sites, pass four names.  The
first two sites are in the z = 0 layer and the last two are in the z = c/2
layer, for example:

    --basis-names A A B B
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


# Fractional coordinates in the orthorhombic cell (a, sqrt(3)*a, c).
# This four-site basis is exactly equivalent to an HCP lattice.
HCP_ORTHORHOMBIC_BASIS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 1.0 / 6.0, 0.5],
        [0.0, 2.0 / 3.0, 0.5],
    ],
    dtype=float,
)

IDEAL_C_OVER_A = np.sqrt(8.0 / 3.0)


def _expanded_basis_names(basis_names: str | Sequence[str]) -> list[str]:
    """Return one atom label for each of the four orthorhombic basis sites."""
    if isinstance(basis_names, str):
        names = [basis_names]
    else:
        names = list(basis_names)

    if len(names) == 1:
        return names * len(HCP_ORTHORHOMBIC_BASIS)
    if len(names) == len(HCP_ORTHORHOMBIC_BASIS):
        return names

    raise ValueError(
        "basis_names must contain either one name or exactly four names "
        f"(received {len(names)})."
    )


def mk_lattice(
    nx: int,
    ny: int,
    nz: int,
    *,
    a: float = 1.0,
    c_over_a: float = IDEAL_C_OVER_A,
    basis_names: str | Sequence[str] = "Mg",
    center: bool = True,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Construct a repeated HCP lattice in a rectangular periodic box.

    Parameters
    ----------
    nx, ny, nz
        Number of orthorhombic unit-cell repeats along x, y, and z.
    a
        Basal-plane nearest-neighbor spacing, in the desired length unit.
    c_over_a
        Ratio c/a.  The ideal hard-sphere HCP value is sqrt(8/3).
    basis_names
        Either one label for every site or four labels, one per basis site.
    center
        If False, wrap coordinates into [0, L) along each direction.  If True,
        wrap them into [-L/2, L/2), similar to the supplied FCC example.

    Returns
    -------
    names
        Atom labels, length 4*nx*ny*nz.
    coordinates
        Cartesian coordinates with shape (N, 3).
    box
        Orthorhombic box lengths [Lx, Ly, Lz].
    """
    repeats = np.asarray([nx, ny, nz], dtype=int)
    if repeats.shape != (3,) or np.any(repeats <= 0):
        raise ValueError("nx, ny, and nz must all be positive integers.")
    if a <= 0.0:
        raise ValueError("a must be positive.")
    if c_over_a <= 0.0:
        raise ValueError("c_over_a must be positive.")

    c = a * c_over_a
    unit_box = np.array([a, np.sqrt(3.0) * a, c], dtype=float)
    box = repeats * unit_box
    unit_names = _expanded_basis_names(basis_names)

    names: list[str] = []
    coordinate_blocks: list[np.ndarray] = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = np.array([i, j, k], dtype=float)
                fractional = HCP_ORTHORHOMBIC_BASIS + shift
                coordinate_blocks.append(fractional * unit_box)
                names.extend(unit_names)

    coordinates = np.vstack(coordinate_blocks)

    # Keep coordinates in a standard periodic representation.
    coordinates = wrap_positions(coordinates, box, center=center)
    return names, coordinates, box


def wrap_positions(
    coordinates: np.ndarray,
    box: np.ndarray,
    *,
    center: bool = False,
) -> np.ndarray:
    """Wrap Cartesian coordinates into an orthorhombic periodic box."""
    coordinates = np.asarray(coordinates, dtype=float)
    box = np.asarray(box, dtype=float)
    if coordinates.shape[-1] != 3 or box.shape != (3,):
        raise ValueError("coordinates must end in dimension 3 and box must be (3,).")
    if np.any(box <= 0.0):
        raise ValueError("All box lengths must be positive.")

    if center:
        return coordinates - box * np.floor(coordinates / box + 0.5)
    return coordinates - box * np.floor(coordinates / box)


def minimum_image(displacement: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Apply the orthorhombic minimum-image convention to displacement(s)."""
    displacement = np.asarray(displacement, dtype=float)
    box = np.asarray(box, dtype=float)
    if displacement.shape[-1] != 3 or box.shape != (3,):
        raise ValueError("displacement must end in dimension 3 and box must be (3,).")
    if np.any(box <= 0.0):
        raise ValueError("All box lengths must be positive.")

    return displacement - box * np.rint(displacement / box)

def pbc_distance(point_a: np.ndarray, point_b: np.ndarray, box: np.ndarray) -> float:
    """Return the minimum-image distance between two points."""
    delta = minimum_image(np.asarray(point_b) - np.asarray(point_a), box)
    return float(np.linalg.norm(delta))

def write_pdb(names, crds) -> str:
    fmt = "ATOM  %5d  %-3s%4s %5d    %8.3f%8.3f%8.3f  1.00  1.00\n"
    resname = "XTL"
    resid = 1
    return "".join(
                fmt%(i+1,name,resname,resid,x[0],x[1],x[2]) \
            for i,(name,x) in enumerate(zip(names, crds)))

with open("HCP.pdb", "w") as f:

    names, coordinates, box = mk_lattice(
        nx=4,#10
        ny=2,#6
        nz=2,#6
        a=1.09016685,
        c_over_a=np.sqrt(8.0 / 3.0),
        basis_names="LJ",
    )

    f.write(write_pdb(names, coordinates))
