"""Tests for strong Dirichlet boundary conditions."""
import unittest
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from IGAeasy.constraints import dirichlet


class DirichletTests(unittest.TestCase):
    def test_nonzero_unsorted_constraints_match_known_solution(self):
        K = np.array([[4., -1, 0], [-1, 4, -1], [0, -1, 3]])
        solution = np.array([2., 5, -3])
        f = K @ solution
        for matrix in (K, csr_matrix(K)):
            reduced, rhs, free = dirichlet.block_partitioning_dbc(matrix, f, [2, 0], [-3, 2])
            dense = reduced.toarray() if hasattr(reduced, 'toarray') else reduced
            np.testing.assert_allclose(np.linalg.solve(dense, rhs), solution[free])
        np.testing.assert_array_equal(K, [[4, -1, 0], [-1, 4, -1], [0, -1, 3]])
        np.testing.assert_array_equal(f, K @ solution)

    def test_explicit_sparse_zeros_are_inactive(self):
        matrix = coo_matrix(([2., 0.], ([0, 1], [0, 1])), shape=(2, 2)).tocsr()
        np.testing.assert_array_equal(dirichlet.identify_inactive_dofs(matrix), [1])

    def test_homogeneous_modification_handles_inactive_dofs(self):
        K = np.diag([2., 0, 3])
        for matrix in (K, csr_matrix(K)):
            modified, rhs = dirichlet.direct_matrix_modification_dbc(matrix, np.array([1., 7, 4]), [0])
            np.testing.assert_array_equal(modified.toarray() if hasattr(modified, 'toarray') else modified,
                                          np.diag([1, 1, 3]))
            np.testing.assert_array_equal(rhs, [0, 0, 4])

    def test_mass_only_reduction(self):
        _, _, M, free = dirichlet.matrix_reduction_dbc(M=np.diag([2., 0, 3]), fixed_DOFs=[0])
        np.testing.assert_array_equal(M, [[3]])
        np.testing.assert_array_equal(free, [2])

    def test_invalid_constraints_are_rejected(self):
        for indices, values in (([0, 0], [1, 2]), ([.5], [1]), ([3], [1]), ([0, 1], [1])):
            with self.assertRaises(ValueError):
                dirichlet.block_partitioning_dbc(np.eye(3), np.zeros(3), indices, values)


if __name__ == '__main__':
    unittest.main(verbosity=2)
