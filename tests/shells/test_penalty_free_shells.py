"""Skew boundary work and independent loaded RM balance checks."""
import unittest
from types import SimpleNamespace
import numpy as np
from shapely.geometry import LineString
from IGAeasy.constraints.shell_weak import WeakRM6Boundary
from IGAeasy.constraints.kl_clamped_weak import WeakKLClampedBoundary
from IGAeasy.examples.shells.verification.boundary_conditions.plate_boundary_comparison import run_level
from IGAeasy.manufactured_solutions.shell_verification import solve_rm_patch as solve_level


class PenaltyFreeShellTests(unittest.TestCase):
    def test_skew_signs_nonzero_data_and_zero_penalty(self):
        boundary = WeakRM6Boundary(LineString([(0.,0.),(1.,0.)]),
            lambda x,y,z: np.zeros(6), ([0.],[2.]), method='penalty_free')
        N, T, g = np.eye(2), np.array([[2.,3.],[-1.,4.]]), np.array([.2,-.3])
        matrix, rhs, gamma = boundary._assemble(SimpleNamespace(K_el=np.eye(2)), [(1.,N,T,g)], 2)
        np.testing.assert_allclose(matrix, -N.T@T+T.T@N)
        np.testing.assert_allclose(rhs, T.T@g)
        np.testing.assert_allclose(matrix+matrix.T, 0.)
        self.assertEqual(gamma, 0.)
        with self.assertRaises(ValueError):
            WeakKLClampedBoundary(boundary.line, boundary.prescribed, method='penalty_free')

    def test_loaded_uniform_plate_reactions_and_work(self):
        for formulation in ('RM6','RM5','RM5_v2'):
            for row in run_level(2, formulation, kind='uniform'):
                with self.subTest(formulation=formulation, method=row['method']):
                    np.testing.assert_allclose(row['reaction_resultants'], [0.,0.,1.,.5,-.5,0.], atol=1e-10)
                    self.assertLess(row['force_balance'], 1e-10)
                    self.assertLess(row['moment_balance'], 1e-10)
                    self.assertLess(row['energy_balance'], 1e-10)
                    if row['method'] == 'penalty_free':
                        self.assertEqual(row['gamma'], 0.)
                        self.assertLess(abs(row['boundary_work']), 1e-12)

    def test_manufactured_plate_convergence_all_methods(self):
        for formulation in ('RM6','RM5','RM5_v2'):
            coarse = run_level(2, formulation)
            fine = run_level(4, formulation)
            for a,b in zip(coarse,fine):
                with self.subTest(formulation=formulation, method=a['method']):
                    self.assertLess(b['l2'], a['l2']/3)
                    self.assertLess(b['director_l2'], a['director_l2'])
                    self.assertLess(b['energy_error'], a['energy_error'])
                    if a['method'] != 'strong':
                        self.assertLess(b['boundary_trace_l2'], a['boundary_trace_l2'])

    def test_nonzero_flat_data_penalty_free(self):
        for formulation in ('RM6','RM5','RM5_v2'):
            _, _, row = solve_level(2, formulation, 'penalty_free', curvature=0., power=3)
            self.assertLess(row['l2'], 1e-10)
            self.assertLess(row['director_l2'], 1e-10)

    def test_curved_trimmed_penalty_free_convergence(self):
        for formulation in ('RM6','RM5','RM5_v2'):
            rows = [solve_level(n, formulation, 'penalty_free')[2] for n in (4,8)]
            self.assertGreater(np.log2(rows[0]['l2']/rows[1]['l2']), 3.5)
            self.assertEqual(rows[1]['gamma_max'], 0.)

    def test_refined_lr_nonzero_data_with_trimmed_hole(self):
        from shapely.geometry import box
        from shapely.affinity import rotate
        domain = box(.07,.09,.93,.91).difference(rotate(box(.37,.39,.63,.61),27.,origin=(.5,.5)))
        for formulation in ('RM6','RM5','RM5_v2'):
            _, _, row = solve_level(2, formulation, 'penalty_free', family='LR', power=3,
                                    local_refinement=True, domain=domain)
            self.assertLess(row['l2'], 1e-7)
            self.assertLess(row['director_l2'], 1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
