"""Overlapping additive Schwarz using dense local Cholesky solves."""
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator


def _validated_blocks(blocks, size):
    unique = set()
    for block in blocks:
        indices = np.asarray(block)
        if indices.ndim != 1 or (indices.size and indices.dtype.kind not in 'iu'):
            raise ValueError('DOF blocks must contain one-dimensional integer indices')
        if np.any(indices < 0) or np.any(indices >= size):
            raise ValueError('Block index outside the matrix')
        key = tuple(np.unique(indices).astype(int))
        if key and key not in unique:
            unique.add(key)
            yield np.asarray(key, dtype=int)


def element_dof_blocks(space, dof_per_cp):
    """Global coefficient-major DOFs of each active material element."""
    if isinstance(dof_per_cp, bool) or not isinstance(dof_per_cp, (int, np.integer)) or dof_per_cp < 1:
        raise ValueError('dof_per_cp must be a positive integer')
    blocks = []
    for cell in space.elements:
        if getattr(cell, 'void', False) or not getattr(cell, 'active', True):
            continue
        ids = sorted({b.id for b in cell.supported_b_splines if getattr(b, 'active', True)})
        blocks.append((np.asarray(ids, dtype=int)[:, None]*dof_per_cp
                       +np.arange(dof_per_cp)).ravel())
    return list(_validated_blocks(blocks, len(space.basis)*dof_per_cp))


def restrict_dof_blocks(blocks, active, size):
    """Map original system indices to the solver's active system."""
    mapping = np.full(size, -1, dtype=int)
    mapping[active] = np.arange(len(active))
    result = []
    for block in _validated_blocks(blocks, size):
        local = mapping[block]
        result.append(local[local >= 0])
    return result


class SchwarzPreconditioner:
    """SPD additive Schwarz: sum of restricted local inverse operators.

    Blocks use indices of A. Repeated blocks are ignored; uncovered DOFs
    receive singleton blocks. Local diagonal scaling improves factorization.
    Singular/indefinite blocks are rejected, never silently regularized.
    No global inverse or transformed spline basis is constructed.
    """
    def __init__(self, A, dof_blocks):
        self.A = sparse.csr_matrix(A, dtype=float)
        if self.A.shape[0] != self.A.shape[1] or not np.isfinite(self.A.data).all():
            raise ValueError('Schwarz requires a finite square matrix')
        difference = self.A-self.A.T
        if np.max(np.abs(difference.data), initial=0.) > 1e-12*max(np.max(np.abs(self.A.data), initial=0.), 1e-300):
            raise ValueError('Schwarz requires a symmetric matrix')
        self.blocks = list(_validated_blocks(dof_blocks, self.A.shape[0]))
        covered = np.zeros(self.A.shape[0], dtype=bool)
        for block in self.blocks:
            covered[block] = True
        self.blocks.extend(np.array([i]) for i in np.flatnonzero(~covered))
        self.compute_preconditioner()

    def compute_preconditioner(self):
        self.factors = []
        for block in self.blocks:
            local = self.A[block, :][:, block].toarray()
            diagonal = local.diagonal()
            if np.any(diagonal <= 0):
                raise ValueError('Schwarz blocks require positive coefficient energies')
            scale = 1/np.sqrt(diagonal)
            local = scale[:, None]*local*scale[None, :]
            try:
                factor = cho_factor((local+local.T)/2, lower=True)
            except np.linalg.LinAlgError as error:
                raise ValueError('Schwarz block is not positive definite; check constraints or cut conditioning') from error
            self.factors.append((scale, factor))

    def apply(self, r):
        r = np.asarray(r, dtype=float)
        if r.shape == (self.A.shape[0], 1):
            r = r[:, 0]
        if r.shape != (self.A.shape[0],):
            raise ValueError('Residual must match the matrix size')
        result = np.zeros_like(r)
        for block, (scale, factor) in zip(self.blocks, self.factors):
            result[block] += scale*cho_solve(factor, scale*r[block])
        return result

    def as_linear_operator(self):
        return LinearOperator(self.A.shape, matvec=self.apply, rmatvec=self.apply, dtype=float)


AdditiveSchwarzPreconditioner = SchwarzPreconditioner
