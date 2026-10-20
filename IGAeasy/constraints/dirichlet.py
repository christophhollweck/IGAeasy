import numpy as np
import scipy.sparse as sp


def _indices(values, size):
    indices = np.asarray(list(values) if values is not None else [], dtype=None)
    if indices.size == 0:
        return np.array([], dtype=int)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu':
        raise ValueError("Prescribed DOFs must be a one-dimensional sequence of integers.")
    if np.any(indices < 0) or np.any(indices >= size) or len(np.unique(indices)) != len(indices):
        raise ValueError("Prescribed DOFs must be unique and within the system range.")
    return indices.astype(int)


def identify_inactive_dofs(A):
    """Identify DOFs where the row is entirely zero (inactive DOFs)"""
    if A is None:
        return np.array([], dtype=int)

    if sp.issparse(A):
        A = A.tocsr(copy=True)
        A.eliminate_zeros()
        return np.where(A.getnnz(axis=1) == 0)[0]
    else:
        return np.where(~A.any(axis=1))[0]


def apply_bc_vector(f, fixed_DOFs):
    """ Set boundary conditions in the force vector """
    if f is not None and fixed_DOFs is not None:
        f = f.copy()
        f[fixed_DOFs] = 0
    return f


def apply_bc_matrix(A, fixed_DOFs):
    """ Apply Dirichlet BC to matrix A (zero rows/columns + diagonal=1) """
    if A is None:
        return None

    is_sparse = sp.issparse(A)
    A = A.tolil() if is_sparse else np.asarray(A).copy()

    inactive_dofs = identify_inactive_dofs(A)

    all_fixed = np.unique(np.concatenate((_indices(fixed_DOFs, A.shape[0]), inactive_dofs)))

    for dof in all_fixed:
        A[dof, :] = 0
        A[:, dof] = 0
        A[dof, dof] = 1

    return A.tocsr() if is_sparse else A


def direct_matrix_modification_dbc(K=None, f=None, fixed_DOFs=None):
    """
    Apply homogeneous Dirichlet boundary conditions using direct modification.
    Handles both dense and sparse matrices, and also inactive DOFs.
    """
    size = K.shape[0] if K is not None else len(f)
    fixed_DOFs = _indices(fixed_DOFs, size)
    all_fixed = np.unique(np.r_[fixed_DOFs, identify_inactive_dofs(K)])

    K_mod = apply_bc_matrix(K, fixed_DOFs)
    f_mod = apply_bc_vector(f, all_fixed)

    return K_mod, f_mod


def matrix_reduction_dbc(K=None, f=None, M=None, fixed_DOFs=None):
    """
    Apply homogeneous Dirichlet boundary conditions by reducing the global system matrix,
    force vector, and optional mass matrix.
    Automatically removes inactive DOFs too.
    """
    if K is not None:
        inactive = identify_inactive_dofs(K)
    elif M is not None:
        inactive = identify_inactive_dofs(M)
    else:
        inactive = []

    size = K.shape[0] if K is not None else (M.shape[0] if M is not None else len(f))
    fixed_DOFs = _indices(fixed_DOFs, size)
    all_fixed = np.unique(np.concatenate((fixed_DOFs, inactive)))

    total_DOFs = np.arange(size)
    free_DOFs = np.setdiff1d(total_DOFs, all_fixed)

    def reduce_matrix(matrix):
        if matrix is None:
            return None
        if sp.issparse(matrix):
            return matrix.tocsr()[free_DOFs, :][:, free_DOFs]
        return matrix[np.ix_(free_DOFs, free_DOFs)]

    K_reduced = reduce_matrix(K)
    f_reduced = f[free_DOFs] if f is not None else None
    M_reduced = reduce_matrix(M)

    return K_reduced, f_reduced, M_reduced, free_DOFs


def block_partitioning_dbc(K=None, f=None, prescribed_DOFs=None, u_c=None):
    """
    Block partitioning for Dirichlet BC, with automatic inactive DOF detection.
    """
    if f is None or np.asarray(f).ndim != 1:
        raise ValueError("Block partitioning requires a one-dimensional force vector.")
    f = np.asarray(f, dtype=float)
    if K is not None and K.shape != (len(f), len(f)):
        raise ValueError("K and f must have matching system dimensions.")
    prescribed_DOFs = _indices(prescribed_DOFs, len(f))
    if u_c is not None:
        u_c = np.asarray(u_c, dtype=float)
        if u_c.shape != (len(prescribed_DOFs),):
            raise ValueError("One prescribed value is required per prescribed DOF.")

    inactive = identify_inactive_dofs(K) if K is not None else np.array([], dtype=int)
    all_fixed = np.unique(np.concatenate((prescribed_DOFs, inactive)))

    all_DOFs = np.arange(len(f))
    free_DOFs = np.setdiff1d(all_DOFs, all_fixed)

    def extract_submatrix(matrix, rows, cols):
        if matrix is None:
            return None
        if sp.issparse(matrix):
            return matrix.tocsr()[rows, :][:, cols]
        return matrix[np.ix_(rows, cols)]

    K_ff = extract_submatrix(K, free_DOFs, free_DOFs)
    K_fc = extract_submatrix(K, free_DOFs, all_fixed)
    f_f = f[free_DOFs] if f is not None else None

    # Ensure that u_c is the correct size by including inactive DOFs and setting them to 0
    if u_c is not None:
        full_u_c = np.zeros(len(all_fixed))
        for i, dof in enumerate(prescribed_DOFs):
            idx = np.where(all_fixed == dof)[0]
            if idx.size == 0:
                raise ValueError(f"Prescribed DOF {dof} not found in all_fixed.")
            full_u_c[idx[0]] = u_c[i]
        u_c = full_u_c

    # Adjusted f considering the Dirichlet BC
    adjusted_f = f_f - K_fc.dot(u_c) if u_c is not None and K_fc is not None else f_f

    return K_ff, adjusted_f, free_DOFs

