import numpy as np
import scipy.sparse as sp

# Keep historical imports valid; these functions have one definition in helpers.
from IGAeasy.constraints.helpers import (
    find_control_point_ids_on_edges, find_control_point_ids_in_box, get_dofs_from_cps,
)


def apply_bc_vector(f, fixed_DOFs):
    """ Set boundary conditions in the force vector """
    if f is not None and fixed_DOFs is not None:
        f[fixed_DOFs] = 0
    return f

def apply_bc_matrix(A, fixed_DOFs):
    """ Set boundary conditions in a matrix (K or M) """

    if A is None:
        return None

    is_sparse = sp.issparse(A)

    if is_sparse:
        A = A.tolil()
    else:
        A = np.asarray(A)

    # diriclet bc for inactive dofs
    for i in range(A.shape[0]):
        if is_sparse:
            if A.getrow(i).nnz == 0:
                A[i, i] = 1
        else:
            if np.all(A[i, :] == 0):
                A[i, i] = 1

    # Essential boundary conditions
    if fixed_DOFs is not None:
        A[:, fixed_DOFs] = 0
        A[fixed_DOFs, :] = 0
        A[fixed_DOFs, fixed_DOFs] = 1

    if is_sparse:
        return A.tocsr()
    return A

def set_dirichlet_bc(M=None, K=None, f=None, fixed_DOFs=None):
    """ Apply Dirichlet boundary conditions to M, K, and f and return only the provided ones """

    if fixed_DOFs is not None:
        fixed_DOFs = np.array(list(fixed_DOFs), dtype=int)

    K = apply_bc_matrix(K, fixed_DOFs)
    M = apply_bc_matrix(M, fixed_DOFs)
    f = apply_bc_vector(f, fixed_DOFs)


    result = []
    if M is not None:
        result.append(M)
    if K is not None:
        result.append(K)
    if f is not None:
        result.append(f)

    return tuple(result)
