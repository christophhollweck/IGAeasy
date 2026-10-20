"""Separate trace bounds preserve virtual work and coefficient invariance."""
import unittest
from types import SimpleNamespace
import numpy as np
from shapely.geometry import LineString
from IGAeasy.constraints.shell_weak import WeakRM6Boundary
from IGAeasy.manufactured_solutions.shell_verification import solve_gradient_patch as solve_level


class SplitStabilizationTests(unittest.TestCase):
    def test_independent_bounds_and_change_of_coefficients(self):
        boundary = WeakRM6Boundary(LineString([(0., 0.), (1., 0.)]), lambda x, y, z: np.zeros(6), ([0.], [2.]))
        flux = np.diag([10., 1.])
        value = np.array([.2, -.1])
        cell = SimpleNamespace(K_el=np.eye(2))
        samples = [(1., np.eye(2), flux, value)]
        K, f, gamma = boundary._assemble(cell, samples, 2,
            energy_factor=np.eye(2), trace_groups=(slice(0, 1), slice(1, 2)))
        np.testing.assert_allclose(boundary.last_trace_penalties, [400., 4.])
        np.testing.assert_allclose(K, np.diag([380., 2.]))
        np.testing.assert_allclose(f, [78., -.3])
        self.assertEqual(gamma, 400.)
        C = np.array([[2., .3], [.2, .7]])
        transformed = SimpleNamespace(K_el=C.T@C)
        F = flux@C
        actual_K, actual_f, _ = boundary._assemble(transformed, [(1., C, F, value)], 2,
            energy_factor=C, trace_groups=(slice(0, 1), slice(1, 2)))
        np.testing.assert_allclose(boundary.last_trace_penalties, [400., 4.], rtol=1e-12)
        np.testing.assert_allclose(actual_K, C.T@K@C, rtol=1e-12)
        np.testing.assert_allclose(actual_f, C.T@f, rtol=1e-12)

    def test_curved_manufactured_fields_with_inner_trim(self):
        from shapely.geometry import box
        from shapely.affinity import rotate
        domain = box(.07, .09, .93, .91).difference(rotate(box(.37, .39, .63, .61), 27., origin=(.5, .5)))
        for formulation in ('KL', 'RM5_hier'):
            with self.subTest(formulation=formulation):
                _, _, row = solve_level(2, power=3, angle=.6, domain=domain,
                    formulation=formulation, stabilization='split')
                self.assertLess(row['l2'], 1e-7)
                self.assertLess(row['patch_residual'], 1e-8)
                self.assertGreater(row['gamma_u_max'], 0.)
                self.assertGreater(row['gamma_d_max'], 0.)
                _, _, original = solve_level(2, power=3, domain=domain,
                    formulation=formulation, stabilization='split')
                for name in ('gamma_u_max', 'gamma_d_max'):
                    self.assertAlmostEqual(row[name]/original[name], 1., places=7)
