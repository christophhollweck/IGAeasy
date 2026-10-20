from scipy.sparse import diags
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator
import numpy as np
import scipy.sparse as sp
from .schwarz import AdditiveSchwarzPreconditioner

class JacobiPreconditioner:
    def __init__(self, A):
        """
        Jacobi Preconditioner using the diagonal of matrix A.
        :param A: The system matrix (sparse or dense).
        """
        self.A = A
        self.M_inv = None
        self.compute_preconditioner()

    def compute_preconditioner(self):
        """Computes the Jacobi preconditioner (inverse of the diagonal)."""
        # Extract diagonal elements
        diag = self.A.diagonal()
        if not np.all(np.isfinite(diag)) or np.any(diag == 0):
            raise ValueError('Jacobi requires finite nonzero diagonal entries.')

        # Invert the diagonal (avoid division by zero)
        self.M_inv = diags(1.0 / diag, 0)

    def apply(self, r):
        """Applies the preconditioner to the residual vector r."""
        return self.M_inv @ r

    def as_linear_operator(self):
        """Convert the preconditioner to a LinearOperator."""
        return LinearOperator(matvec=self.apply, dtype=self.A.dtype, shape=self.A.shape)

class ILUPreconditioner:
    def __init__(self, A, drop_tol=1e-6, fill_factor=10):
        """
        Incomplete LU (ILU) Preconditioner using scipy's sparse.linalg.

        :param A: The system matrix (sparse matrix).
        """
        if not np.isfinite(drop_tol) or drop_tol < 0 or not np.isfinite(fill_factor) or fill_factor < 1:
            raise ValueError('ILU requires drop_tol >= 0 and fill_factor >= 1.')
        self.A = sp.csc_matrix(A, dtype=float)
        self.drop_tol = drop_tol
        self.fill_factor = fill_factor
        self.M_inv = None
        self.compute_preconditioner()

    def compute_preconditioner(self):
        """
        Computes the Incomplete LU factorization of the matrix.
        """
        self.M_inv = spla.spilu(self.A, drop_tol=self.drop_tol, fill_factor=self.fill_factor)

    def apply(self, r):
        """
        Applies the preconditioner to the residual vector r.

        :param r: The residual vector.
        :return: Preconditioned residual vector.
        """
        return self.M_inv.solve(r)

    def as_linear_operator(self):
        """
        Converts the preconditioner to a LinearOperator.
        """
        return LinearOperator(matvec=self.apply, dtype=self.A.dtype, shape=self.A.shape)
