"""Direct solution of validated dense or sparse linear systems."""
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from IGAeasy.solver.linear_system import prepare, equilibrate


def solve(K, f, tol=1e-14, condnum=False, scaling=False):
    """Solve K*u=f; tol detects inactive DOFs relative to the largest entry.

    condnum=True evaluates the original reduced matrix, before equilibration.
    scaling=True or 'auto' preserves all nonzero DOFs and solves D*K*D*y=D*f.
    """
    reduced, rhs, active, size, _ = prepare(K, f, 0. if scaling else tol)
    solution = np.zeros(size)
    condition = 1.0
    if not len(active):
        equilibrate(reduced, rhs, scaling)
    if len(active):
        if condnum:
            condition = np.linalg.cond(reduced.toarray() if sp.issparse(reduced) else reduced)
        reduced, rhs, back_scale = equilibrate(reduced, rhs, scaling)
        if sp.issparse(reduced):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', spla.MatrixRankWarning)
                    values = spla.spsolve(reduced, rhs)
            except spla.MatrixRankWarning as error:
                raise np.linalg.LinAlgError('Sparse system is singular.') from error
        else:
            values = np.linalg.solve(reduced, rhs)
        values = back_scale*values
        if not np.all(np.isfinite(values)):
            raise np.linalg.LinAlgError('Direct solve produced nonfinite values.')
        solution[active] = values
    return (solution, condition) if condnum else solution
