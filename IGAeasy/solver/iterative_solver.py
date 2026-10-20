"""Sparse iterative solution with explicit convergence failure handling."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from IGAeasy.preconditioning import JacobiPreconditioner, ILUPreconditioner, SchwarzPreconditioner
from IGAeasy.preconditioning.schwarz import restrict_dof_blocks
from IGAeasy.solver.linear_system import prepare, equilibrate


def solve(K, f, preconditioner='Jacobi', atol=1e-12, maxiter=5000, rtol=1e-8, method=None,
          scaling=False, tol=1e-14, dof_blocks=None):
    """CG for symmetric positive-definite systems; GMRES with ILU.

    Tolerances apply after normalizing by the largest matrix entry. With
    diagonal scaling, the residual is also checked in that original system.
    scaling=True or 'auto' retains every nonzero DOF, irrespective of tol.
    Schwarz uses dof_blocks in original system numbering, before reduction.
    """
    reduced, rhs, active, size, _ = prepare(K, f, 0. if scaling else tol)
    original, original_rhs = reduced, rhs
    reduced, rhs, back_scale = equilibrate(reduced, rhs, scaling)
    solution = np.zeros(size)
    if preconditioner not in (None, 'None', 'Jacobi', 'ILU', 'Schwarz'):
        raise ValueError('Unknown preconditioner.')
    if preconditioner == 'Schwarz':
        if dof_blocks is None:
            raise ValueError('Schwarz requires dof_blocks in original system numbering')
        dof_blocks = restrict_dof_blocks(dof_blocks, active, size)
    method = method or ('gmres' if preconditioner == 'ILU' else 'cg')
    if method not in ('cg', 'gmres'):
        raise ValueError("method must be 'cg' or 'gmres'.")
    if method == 'cg' and preconditioner == 'ILU':
        raise ValueError('General ILU is not a symmetric positive-definite CG preconditioner; use GMRES.')
    if not len(active):
        return solution
    reduced = sp.csr_matrix(reduced)
    if preconditioner == 'Jacobi':
        if method == 'cg' and np.any(reduced.diagonal() <= 0):
            raise ValueError('Jacobi-preconditioned CG requires positive diagonal entries.')
        operator = JacobiPreconditioner(reduced).as_linear_operator()
    elif preconditioner == 'ILU':
        operator = ILUPreconditioner(reduced.tocsc()).as_linear_operator()
    elif preconditioner == 'Schwarz':
        operator = SchwarzPreconditioner(reduced, dof_blocks).as_linear_operator()
    else:
        operator = None
    algorithm = spla.gmres if method == 'gmres' else spla.cg
    if method == 'cg':
        difference = reduced - reduced.T
        if np.max(np.abs(difference.data), initial=0) > 1e-12:
            raise ValueError('CG requires a symmetric positive-definite matrix; use ILU/GMRES for nonsymmetric systems.')
    values, info = algorithm(reduced, rhs, M=operator, maxiter=maxiter, atol=atol, rtol=rtol)
    if info != 0 or not np.all(np.isfinite(values)):
        raise RuntimeError(f'{method} failed to converge (info={info}).')
    values = back_scale*values
    if not np.isfinite(values).all():
        raise RuntimeError('Back-transformed solution is nonfinite.')
    if scaling:
        residual = np.linalg.norm(original@values-original_rhs)
        limit = max(atol, rtol*np.linalg.norm(original_rhs))
        if residual > 10*limit:
            raise RuntimeError('Scaled solve failed the original-system residual check.')
    solution[active] = values
    return solution
