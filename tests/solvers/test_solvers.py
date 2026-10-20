"""Regression tests for linear solvers."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from IGAeasy.solver import direct_solver, iterative_solver


class SolverTests(unittest.TestCase):
    def test_equilibrated_solvers_preserve_tiny_active_dofs(self):
        # The first DOF would be lost under magnitude-based inactive detection.
        K = np.diag([1e-20, 2., 0.])
        expected = np.array([3., -2., 0.])
        for matrix in (K, csr_matrix(K)):
            for scaling in (True, 'auto'):
                for solver in (direct_solver, iterative_solver):
                    options = {'atol': 0.} if solver is iterative_solver else {}
                    computed = solver.solve(matrix, matrix@expected, scaling=scaling, **options)
                    np.testing.assert_allclose(computed, expected, atol=1e-10)

    def test_scaled_ilu_coupled_system_and_original_condition(self):
        D = np.diag([1e-8, 1., 10.])
        K = D@np.array([[4., -1., 0.], [-1., 4., -1.], [0., -1., 3.]])@D
        expected = np.array([1e8, 2., -3.])
        np.testing.assert_allclose(iterative_solver.solve(csr_matrix(K), K@expected,
            preconditioner='ILU', scaling=True, atol=0., rtol=1e-12), expected, rtol=1e-10)
        _, condition = direct_solver.solve(np.diag([2., 4.]), [2., 8.],
                                           scaling=True, condnum=True)
        self.assertAlmostEqual(condition, 2.)

    def test_scaling_auto_leaves_indefinite_diagonal_unchanged(self):
        K = np.diag([-2., 3.])
        expected = np.array([2., -3.])
        np.testing.assert_allclose(direct_solver.solve(K, K@expected, scaling='auto'), expected)
        for solver in (direct_solver, iterative_solver):
            with self.assertRaises(ValueError):
                solver.solve(K, K@expected, scaling=True)
            with self.assertRaises(ValueError):
                solver.solve(np.eye(2), [1., 2.], scaling='invalid')

    def test_scaled_convergence_checks_untransformed_residual(self):
        K = np.diag([1e-8, 1.])
        with patch('IGAeasy.solver.iterative_solver.spla.cg', return_value=(np.zeros(2), 0)):
            with self.assertRaisesRegex(RuntimeError, 'original-system residual'):
                iterative_solver.solve(K, [1e-8, 1.], scaling=True)

    def test_jacobi_gmres_and_preconditioner_validation(self):
        from IGAeasy.preconditioning import JacobiPreconditioner, ILUPreconditioner
        K = np.array([[4., 2], [-1, 3]])
        expected = np.array([2., -3])
        np.testing.assert_allclose(iterative_solver.solve(K, K @ expected, method='gmres'), expected, atol=1e-10)
        with self.assertRaises(ValueError):
            JacobiPreconditioner(np.diag([1., 0]))
        with self.assertRaises(ValueError):
            ILUPreconditioner(K, fill_factor=.5)
        with self.assertRaises(ValueError):
            iterative_solver.solve(K, [1, 2], preconditioner='ILU', method='cg')

    def test_ilu_handles_nonsymmetric_system(self):
        K = csr_matrix([[4., 2], [-1, 3]])
        expected = np.array([2., -3])
        np.testing.assert_allclose(iterative_solver.solve(K, K @ expected, preconditioner='ILU'), expected, atol=1e-10)
        with self.assertRaises(ValueError):
            iterative_solver.solve(K, K @ expected)

    def test_empty_system_condition_number_and_invalid_shapes(self):
        solution, condition = direct_solver.solve(np.zeros((0, 0)), [], condnum=True)
        self.assertEqual(len(solution), 0)
        self.assertEqual(condition, 1)
        solution, condition = direct_solver.solve(np.diag([2., 4]), [2, 8], condnum=True)
        np.testing.assert_allclose(solution, [1, 2])
        self.assertAlmostEqual(condition, 2)
        for solver in (direct_solver, iterative_solver):
            with self.assertRaises(ValueError):
                solver.solve(np.ones((2, 3)), [1, 2])

    def test_known_solution_dense_sparse_and_scaled_system(self):
        K = np.array([[4., -1, 0], [-1, 3, 0], [0, 0, 0]])
        expected = np.array([2., -3, 0])
        for scale in (1, 1e-20):
            for matrix in (K*scale, csr_matrix(K*scale)):
                for solver in (direct_solver, iterative_solver):
                    np.testing.assert_allclose(solver.solve(matrix, matrix @ expected), expected, atol=1e-10)

    def test_sparse_iterative_solver_never_densifies(self):
        K = csr_matrix([[4., -1], [-1, 3]])
        with patch.object(csr_matrix, 'toarray', side_effect=AssertionError('Sparse matrix was densified')):
            np.testing.assert_allclose(iterative_solver.solve(K, [1, 2]), np.linalg.solve([[4, -1], [-1, 3]], [1, 2]))

    def test_direct_singular_system_raises(self):
        for matrix in (np.ones((2, 2)), csr_matrix(np.ones((2, 2)))):
            with self.assertRaises(np.linalg.LinAlgError):
                direct_solver.solve(matrix, np.ones(2))

    def test_load_on_inactive_dof_is_rejected(self):
        for solver in (direct_solver, iterative_solver):
            with self.assertRaises(ValueError):
                solver.solve(np.diag([1., 0]), [1, 1])

    def test_nonconvergence_is_not_returned_as_solution(self):
        K = np.array([[4., -1, 0], [-1, 4, -1], [0, -1, 3]])
        with self.assertRaises(RuntimeError):
            iterative_solver.solve(K, np.ones(3), preconditioner=None, maxiter=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
