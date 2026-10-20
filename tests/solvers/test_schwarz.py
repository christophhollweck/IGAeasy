"""Local inverse work, SPD symmetry and solver index reduction."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from IGAeasy.preconditioning import SchwarzPreconditioner, element_dof_blocks
from IGAeasy.solver.iterative_solver import solve


class SchwarzTests(unittest.TestCase):
    def test_overlapping_operator_matches_restricted_inverses(self):
        K = np.array([[4., -1., 0.], [-1., 3., -1.], [0., -1., 2.]])
        blocks = [[0, 1], [1, 2]]
        expected = np.zeros_like(K)
        for block in blocks:
            expected[np.ix_(block, block)] += np.linalg.inv(K[np.ix_(block, block)])
        pre = SchwarzPreconditioner(K, blocks+[[1, 0]])
        operator = pre.as_linear_operator()
        actual = operator@np.eye(3)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        np.testing.assert_allclose(actual, actual.T, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(actual).min(), 0.)

    def test_uncovered_singletons_and_local_scaling(self):
        K = np.diag([1e-20, 2., 3.])
        pre = SchwarzPreconditioner(csr_matrix(K), [[0, 1]])
        np.testing.assert_allclose(pre.apply(K@np.ones(3)), np.ones(3))

    def test_solver_remaps_blocks_after_inactive_reduction(self):
        K = np.array([[4., 0., -1.], [0., 0., 0.], [-1., 0., 2.]])
        expected = np.array([2., 0., -3.])
        for scaling in (False, True):
            for method in ('cg', 'gmres'):
                np.testing.assert_allclose(solve(K, K@expected, preconditioner='Schwarz',
                    dof_blocks=[[0, 1, 2]], scaling=scaling, method=method,
                    atol=0., rtol=1e-12), expected, atol=1e-12)

    def test_element_blocks_skip_void_inactive_and_duplicate_cells(self):
        basis = [SimpleNamespace(id=i, active=True) for i in range(3)]
        a = SimpleNamespace(supported_b_splines=basis[:2], void=False)
        b = SimpleNamespace(supported_b_splines=basis[1:], void=True)
        c = SimpleNamespace(supported_b_splines=basis, active=False)
        space = SimpleNamespace(basis=basis, elements=[a, b, c, a])
        blocks = element_dof_blocks(space, 2)
        self.assertEqual(len(blocks), 1)
        np.testing.assert_array_equal(blocks[0], [0, 1, 2, 3])

    def test_rejects_invalid_and_singular_blocks(self):
        for blocks in ([[0, 3]], [[-1]], [[.5]], [[[0]]]):
            with self.assertRaises(ValueError):
                SchwarzPreconditioner(np.eye(3), blocks)
        for K in (np.ones((2, 2)), np.array([[1., 2.], [0., 1.]])):
            with self.assertRaises(ValueError):
                SchwarzPreconditioner(K, [[0, 1]])
        with self.assertRaises(ValueError):
            solve(np.eye(2), np.ones(2), preconditioner='Schwarz')

    def test_trimmed_curved_kl_patch(self):
        from shapely.geometry import box
        from IGAeasy.examples.shells.verification.boundary_conditions.gradient_clamped_convergence import solve_level
        from IGAeasy.assembler import MatrixAssembler, VectorAssembler
        from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature
        rs, direct, _ = solve_level(4, power=3, boundary_kind='mixed', angle=.6,
                                    domain=box(0., 0., .25025, 1.))
        K = MatrixAssembler(dof_per_cp=3).assemble(rs, 'K_el', use_sparse=True)
        f = VectorAssembler(dof_per_cp=3).assemble(rs, 'f_el')
        computed = solve(K, f, preconditioner='Schwarz', dof_blocks=element_dof_blocks(rs, 3),
                         scaling=True, rtol=1e-11, atol=0.)
        difference = SurfaceL2Quadrature(rs, order=6).error((computed-direct).reshape(-1, 3))
        self.assertLess(difference, 1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
