"""Mass-matrix lumping without changing dense or sparse storage type."""
import numpy as np
import scipy.sparse as sp


def row_summing_matrix(A):
    """Replace each matrix row with its row sum on the diagonal.
       Returns the same type (dense or sparse) as input."""
    if A is None:
        return None

    if sp.issparse(A):
        row_sums = np.array(A.sum(axis=1)).flatten()
        return sp.diags(row_sums, offsets=0, format=A.format)
    else:
        row_sums = A.sum(axis=1)
        return np.diag(row_sums)
