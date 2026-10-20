"""Analytical flat-strip response and the independently documented Raasch geometry."""
import unittest
import numpy as np
from IGAeasy.examples.shells.trimmed import flat_cantilever_rm5, raasch_hook_rm5
from IGAeasy.geometry.get_geometry import raasch_hook
from IGAeasy.examples.shells.benchmarks.raasch_hook.rm5 import run as run_fitted_hook


class TrimmedShellBenchmarkTests(unittest.TestCase):
    def test_fitted_hook_has_same_geometry_and_response_without_trimmed_cells(self):
        trimmed = raasch_hook()
        fitted = raasch_hook(trimmed=False)
        parameters = np.array([(u, v) for u in np.linspace(0., 1., 11)
                               for v in np.linspace(1/12, 11/12, 7)])
        for derivative in ([0, 0], [1, 0], [0, 1]):
            np.testing.assert_allclose(trimmed.derivative(parameters, derivative),
                                       fitted.derivative(parameters, derivative), atol=1e-10)
        space, row = run_fitted_hook(show=False)
        _, reference = raasch_hook_rm5.run(show=False)
        self.assertFalse(any(cell.trimmed or cell.void for cell in space.elements))
        self.assertAlmostEqual(row['displacement'], reference['displacement'], places=7)

    def test_flat_strip_recovers_exact_field_after_rotation(self):
        for angle in (0., .6):
            _, row = flat_cantilever_rm5.run(4, show=False, rotation_angle=angle)
            self.assertLess(row['absolute_error'], 1e-10)
            self.assertLess(row['l2'], 1e-10)
            self.assertLess(row['residual'], 1e-9)

    def test_hook_geometry_matches_documented_endpoints_and_width(self):
        geometry = raasch_hook()
        expected = [[0., 0., 0.], [30*np.sqrt(3)+46, -16., 20.]]
        np.testing.assert_allclose(geometry.evaluate([[0., 1/12], [1., 11/12]]), expected, atol=1e-10)
        join = (14*np.pi/3)/(14*np.pi/3+46*5*np.pi/6)
        left = geometry.derivative([[join-1e-9, .5]], [1, 0])
        right = geometry.derivative([[join+1e-9, .5]], [1, 0])
        np.testing.assert_allclose(left, right, atol=1e-5)
        refined = raasch_hook(1)
        points = np.column_stack((np.linspace(0., 1., 21), np.full(21, .5)))
        np.testing.assert_allclose(geometry.evaluate(points), refined.evaluate(points), atol=1e-11)

    def test_hook_coarse_response_and_rotation_agree(self):
        rows = [raasch_hook_rm5.run(show=False, rotation_angle=angle)[1] for angle in (0., .6)]
        np.testing.assert_allclose(rows[0]['displacement'], rows[1]['displacement'], rtol=1e-7)
        self.assertLess(rows[1]['relative_benchmark_difference'], .01)
        self.assertLess(rows[1]['residual'], 1e-8)
        self.assertLess(rows[1]['radius_error'], .001)


if __name__ == '__main__':
    unittest.main(verbosity=2)
