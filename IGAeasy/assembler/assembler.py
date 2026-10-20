"""Assembly with control-point-major local and global degree-of-freedom ordering."""
import numpy as np
from scipy.sparse import coo_matrix, lil_matrix


def _validate_dof_count(count):
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("dof_per_cp must be a positive integer.")
    return int(count)


def _active_cells(RS):
    return (cell for cell in RS.elements
            if getattr(cell, 'active', True) and not getattr(cell, 'void', False))


def _dofs(cell, dof_per_cp, n_cps):
    ids = np.asarray(getattr(cell, 'bfnums', None))
    if ids.ndim != 1 or ids.dtype.kind not in 'iu':
        raise ValueError("bfnums must be a one-dimensional sequence of integer basis IDs.")
    if np.any(ids < 0) or np.any(ids >= n_cps) or len(np.unique(ids)) != len(ids):
        raise ValueError("bfnums must contain unique basis IDs within the global basis range.")
    return (ids[:, None]*dof_per_cp + np.arange(dof_per_cp)).ravel()


def _local_data(cell, name, shape):
    value = getattr(cell, name, None)
    if value is None:
        raise ValueError(f"Element data {name!r} is missing.")
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"Element data {name!r} has shape {array.shape}; expected {shape}.")
    return array


class MatrixAssembler:
    def __init__(self, dof_per_cp):
        self.dof_per_cp = _validate_dof_count(dof_per_cp)

    def assemble(self, RS, matrix_type, use_sparse=False):
        n = len(RS.basis)*self.dof_per_cp
        matrix = None if use_sparse else np.zeros((n, n))
        rows, columns, values = [], [], []
        for cell in _active_cells(RS):
            dofs = _dofs(cell, self.dof_per_cp, len(RS.basis))
            local = _local_data(cell, matrix_type, (len(dofs), len(dofs)))
            if use_sparse:
                rows.append(np.repeat(dofs, len(dofs)))
                columns.append(np.tile(dofs, len(dofs)))
                values.append(local.ravel())
            else:
                matrix[np.ix_(dofs, dofs)] += local
        if not use_sparse:
            return matrix
        if not values:
            return lil_matrix((n, n))
        result = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(n, n)).tocsr()
        result.eliminate_zeros()
        return result.tolil()


class VectorAssembler:
    def __init__(self, dof_per_cp):
        self.dof_per_cp = _validate_dof_count(dof_per_cp)

    def assemble(self, RS, vector_type):
        vector = np.zeros(len(RS.basis)*self.dof_per_cp)
        for cell in _active_cells(RS):
            dofs = _dofs(cell, self.dof_per_cp, len(RS.basis))
            vector[dofs] += _local_data(cell, vector_type, (len(dofs),))
        return vector


class Assembler:
    def __init__(self, dof_per_cp):
        self.dof_per_cp = _validate_dof_count(dof_per_cp)

    def assemble(self, RS, use_sparse=False, assemble_K=True, assemble_M=False, assemble_f=True):
        matrices = MatrixAssembler(self.dof_per_cp)
        result = {}
        if assemble_K:
            result['K'] = matrices.assemble(RS, 'K_el', use_sparse)
        if assemble_M:
            result['M'] = matrices.assemble(RS, 'M_el', use_sparse)
        if assemble_f:
            result['f'] = VectorAssembler(self.dof_per_cp).assemble(RS, 'f_el')
        return result
