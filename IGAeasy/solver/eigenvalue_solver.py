import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

def solve(K, M, max_num_modes=None, constrained_dofs=None, tol=1e-12):
    """
    Solve a reduced EVP with explicitly constrained and zero-row DOFs.
    Fully sparse-capable, uses iterative solver for large problems.

    Parameters:
    - K, M             : ndarray or scipy.sparse, stiffness and mass matrices
    - max_num_modes    : int or None
                         None → compute all modes (dense fallback for small matrices)
                         > 0  → compute first N modes
                         -1   → compute 1 mode iteratively
    - constrained_dofs : list of constrained DOF indices (optional)
    - tol              : float, threshold for detecting zero rows/columns in K

    Returns:
    - eigenvalues      : (k,) ndarray
    - full_eigenvectors: (n x k) ndarray (zero-padded at constrained DOFs)
    """
    n = K.shape[0]
    assert K.shape == M.shape

    if constrained_dofs is None:
        constrained_dofs = set()
    else:
        constrained_dofs = set(constrained_dofs)

    # --- find zero rows/columns ---
    if sp.issparse(K):
        null_rows = np.where(np.abs(K).sum(axis=1).A1 < tol)[0]
        null_cols = np.where(np.abs(K).sum(axis=0).A1 < tol)[0]
    else:
        null_rows = np.where(np.all(np.abs(K) < tol, axis=1))[0]
        null_cols = np.where(np.all(np.abs(K) < tol, axis=0))[0]

    constrained_dofs.update(set(null_rows).union(set(null_cols)))
    constrained_dofs = sorted(list(constrained_dofs))

    # --- active DOFs ---
    active_dofs_mask = np.ones(n, dtype=bool)
    active_dofs_mask[constrained_dofs] = False
    active_dofs = np.where(active_dofs_mask)[0]

    # --- reduced matrices ---
    if sp.issparse(K):
        K_red = K[active_dofs, :][:, active_dofs]
        M_red = M[active_dofs, :][:, active_dofs]
    else:
        K_red = K[np.ix_(active_dofs, active_dofs)]
        M_red = M[np.ix_(active_dofs, active_dofs)]

    num_active = len(active_dofs)

    # --- choose solver ---
    use_sparse_solver = sp.issparse(K) or (max_num_modes is not None and max_num_modes > 0 and max_num_modes < num_active)

    if use_sparse_solver:
        # determine number of modes
        if max_num_modes is None or max_num_modes == -1:
            k = min(20, num_active-1)  # default small number of modes
        else:
            k = min(max_num_modes, num_active-1)
        evals, evecs = eigsh(K_red, M=M_red, k=k, which='LM')
    else:
        # dense fallback
        if max_num_modes is None:
            evals, evecs = eigh(K_red, M_red)
        else:
            k = min(max_num_modes, num_active)
            evals, evecs = eigh(K_red, M_red, subset_by_index=(0, k-1))

    # --- expand eigenvectors ---
    full_evecs = np.zeros((n, evecs.shape[1]))
    full_evecs[active_dofs, :] = evecs

    return evals, full_evecs
