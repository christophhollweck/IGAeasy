"""Navier series for the uniformly loaded unit square."""
from functools import lru_cache
import numpy as np

ax = ay = 1.

@lru_cache(maxsize=32)
def navier_coefficients(n_terms=200, include_shear=True, *, E=10920., nue=.3,
                        thickness=.1, pressure=-1., kappa=5/6):
    """Navier series with odd modes: RM includes shear, KL omits it."""
    t, q = thickness, pressure
    D = E * t ** 3 / (12 * (1 - nue ** 2))
    G = E / (2 * (1 + nue))
    modes = np.arange(1, 2 * n_terms, 2, dtype=float)
    m, nn = modes[:, None], modes[None, :]
    k_squared = (m * np.pi / ax) ** 2 + (nn * np.pi / ay) ** 2
    q_mn = 16 * q / (np.pi ** 2 * m * nn)
    w_mn = q_mn / (D * k_squared ** 2)
    if include_shear:
        w_mn += q_mn / (kappa * G * t * k_squared)
    return modes, w_mn


def navier_center(n_terms=200, include_shear=True, **material_and_load):
    modes, coefficients = navier_coefficients(n_terms, include_shear, **material_and_load)
    center_modes = np.sin(modes * np.pi / 2)
    return center_modes @ coefficients @ center_modes


@lru_cache(maxsize=8192)
def navier_x_modes(x, n_terms=200):
    modes = np.arange(1, 2*n_terms, 2, dtype=float)
    return np.sin(modes * np.pi * x / ax)


@lru_cache(maxsize=8192)
def navier_y_sum(y, include_shear=True, n_terms=200, **material_and_load):
    modes, coefficients = navier_coefficients(n_terms, include_shear, **material_and_load)
    return coefficients @ np.sin(modes * np.pi * y / ay)


def navier_solution(x, y, include_shear=True, n_terms=200, **material_and_load):
    return navier_x_modes(x, n_terms) @ navier_y_sum(y, include_shear, n_terms, **material_and_load)
