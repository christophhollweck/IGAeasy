"""Physical checks on the public, complete shell setups."""
import importlib
import unittest
import numpy as np
from shapely.geometry import box

from IGAeasy.geometry.get_geometry import polynomial_shell
from IGAeasy.examples.shells.verification.trimmed.manufactured_hole_rm5 import run as solve_trimmed


class StandaloneShellTests(unittest.TestCase):
    def test_circular_hole_converges_with_moderate_cuts(self):
        rows = [solve_trimmed(n, show=False)[1] for n in (4, 8)]
        self.assertGreater(np.log2(rows[0]['l2']/rows[1]['l2']), 3.5)
        self.assertTrue(all(row['min_parametric_fraction'] >= 1/9-1e-12 for row in rows))
        self.assertLess(rows[1]['residual'], 1e-10)

    def test_rectangle_geometry_matches_background_positions_and_tangents(self):
        bounds = (1/12, 1/12, 11/12, 11/12)
        full = polynomial_shell(4, 4, curvature=(.4, -.2), mixed=(1.2, -.8))
        fitted = polynomial_shell(4, 4, curvature=(.4, -.2), mixed=(1.2, -.8), bounds=bounds)
        points = np.array([(u, v) for u in np.linspace(bounds[0], bounds[2], 7)
                          for v in np.linspace(bounds[1], bounds[3], 7)])
        for derivative in ([0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]):
            np.testing.assert_allclose(full.derivative(points, derivative), fitted.derivative(points, derivative),
                                       rtol=1e-11, atol=1e-12)

    def test_fitted_and_trimmed_rectangle_converge_to_same_physical_reference(self):
        domain = box(1/12, 1/12, 11/12, 11/12)
        for fitted in (False, True):
            rows = [solve_trimmed(n, show=False, material_domain=domain, fitted_rectangle=fitted)[1]
                    for n in (4, 8)]
            self.assertGreater(np.log2(rows[0]['l2']/rows[1]['l2']), 3.5)
            self.assertLess(rows[1]['l2'], 2e-8)
            self.assertLess(rows[1]['residual'], 1e-10)

    def test_plate_l2_convergence_for_all_formulations(self):
        for name in ('rm6', 'rm5', 'rm5_hierarchical', 'kirchhoff_love'):
            with self.subTest(formulation=name):
                setup = importlib.import_module('IGAeasy.examples.shells.benchmarks.simply_supported_plate.'+name)
                rows = [setup.run(n, show=False)[1] for n in (2, 4)]
                self.assertLess(rows[1]['l2'], rows[0]['l2']/5)
                self.assertLess(rows[1]['benchmark_difference'], rows[0]['benchmark_difference'])
                _, original = setup.run(2, show=False, rotation_angle=0.)
                np.testing.assert_allclose(rows[0]['displacement'], original['displacement'], rtol=1e-7)

    def test_roof_nitsche_agrees_with_strong_under_refinement(self):
        for name in ('rm6', 'rm5', 'rm5_hierarchical', 'kirchhoff_love'):
            setup = importlib.import_module('IGAeasy.examples.shells.benchmarks.scordelis_roof.'+name)
            differences = []
            for n in (4, 8):
                _, weak = setup.run(n, show=False)
                _, strong = setup.run(n, method='strong', show=False)
                differences.append(abs(weak['displacement']-strong['displacement']))
                self.assertLess(weak['residual'], 1e-7)
            self.assertLess(differences[1], differences[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
