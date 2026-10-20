"""Validation, scaling and inactive-DOF reduction for linear systems."""
import numpy as np
import scipy.sparse as sp


def condition_estimate(A):
    """Estimate the sparse matrix condition in the 1-norm, using an LU solve."""
    from scipy.sparse.linalg import splu, LinearOperator, onenormest
    lu = splu(A.tocsc())
    inverse = LinearOperator(A.shape, matvec=lu.solve,
                             rmatvec=lambda x: lu.solve(x, trans='T'), dtype=float)
    return float(np.asarray(abs(A).sum(axis=0)).max()*onenormest(inverse))


def equilibrate(K, f, scaling=False):
    """Return D*K*D, D*f and d for recovering x=d*y.

    'auto' scales whenever all active diagonal entries are positive; otherwise
    it leaves the system unchanged. True requires a positive diagonal.
    No DOFs are discarded by this transformation.
    """
    if scaling not in (False, True, 'auto'):
        raise ValueError("scaling must be False, True or 'auto'.")
    diagonal = K.diagonal()
    if scaling is False or not len(diagonal):
        return K, f, np.ones(len(f))
    if np.any(diagonal <= 0):
        if scaling == 'auto':
            return K, f, np.ones(len(f))
        raise ValueError('Symmetric diagonal scaling requires positive diagonal entries.')
    d = 1/np.sqrt(diagonal)
    if sp.issparse(K):
        D = sp.diags(d)
        scaled = (D@K@D).tocsr()
        finite = np.isfinite(scaled.data).all()
    else:
        scaled = d[:, None]*K*d[None, :]
        finite = np.isfinite(scaled).all()
    rhs = d*f
    if not finite or not np.isfinite(rhs).all():
        raise ValueError('Diagonal scaling produced nonfinite coefficients.')
    return scaled, rhs, d


def prepare(K, f, tol=1e-14):
    K = K.astype(float).tocsr() if sp.issparse(K) else np.asarray(K, dtype=float)
    f = np.asarray(f, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or f.shape != (K.shape[0],):
        raise ValueError("Expected a square matrix and matching one-dimensional RHS.")
    data = K.data if sp.issparse(K) else K
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(f)):
        raise ValueError("Linear system coefficients must be finite.")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("Inactive-DOF tolerance must be finite and nonnegative.")
    scale = float(np.max(np.abs(data), initial=0))
    if scale:
        K, f = K/scale, f/scale
    if sp.issparse(K):
        pattern = K.copy()
        pattern.data[np.abs(pattern.data) <= tol] = 0
        pattern.eliminate_zeros()
        rows = pattern.getnnz(axis=1) == 0
        columns = pattern.getnnz(axis=0) == 0
    else:
        rows, columns = np.all(np.abs(K) <= tol, axis=1), np.all(np.abs(K) <= tol, axis=0)
    if not np.array_equal(rows, columns):
        raise np.linalg.LinAlgError("Unmatched zero rows/columns indicate a singular system.")
    if np.any(f[rows] != 0):
        raise ValueError("A nonzero load is applied to an inactive DOF.")
    active = np.flatnonzero(~rows)
    reduced = K[active, :][:, active] if sp.issparse(K) else K[np.ix_(active, active)]
    return reduced, f[active], active, len(f), scale
