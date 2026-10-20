"""Shared tensor-product derivative conventions for THB and LR bases."""
import numpy as np


def third_derivatives(basis, u, v, cell=None):
    """Return (N_uuu, N_uuv, N_uvv, N_vvv) on the chosen polynomial cell."""
    return np.array([basis(u, v, r1=i, r2=3-i, cell=cell) for i in (3, 2, 1, 0)])
