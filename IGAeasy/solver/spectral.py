"""Inexpensive spectral indicators for time-step studies."""
import numpy as np
from scipy.sparse import issparse


def get_max_rayleigh(K, M):
    """
    Calculates the lower bound of the maximum eigenfrequency
    using the maximum ratio of diagonal entries (Rayleigh quotient for unit vectors).
    Works for both dense and sparse matrices.
    """
    # Extract diagonals safely
    if issparse(K):
        diag_k = K.diagonal()
    else:
        diag_k = np.diag(K)

    if issparse(M):
        diag_m = M.diagonal()
    else:
        diag_m = np.diag(M)

    # Avoid division by zero for constrained DOFs or empty cells
    mask = diag_m > 0
    ratios = np.zeros_like(diag_k)
    ratios[mask] = diag_k[mask] / diag_m[mask]

    omega_max_Q = np.sqrt(np.max(ratios))

    return omega_max_Q
