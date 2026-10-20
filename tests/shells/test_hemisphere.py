"""Open hemisphere geometry, reflection supports and frame covariance."""
import unittest
import numpy as np

from IGAeasy.geometry.get_geometry import open_hemisphere
from IGAeasy.examples.shells.benchmarks.pinched_hemisphere import (
    rm6, rm5, rm5_hierarchical, kirchhoff_love)


def solve_level(n, formulation, method='nitsche', rotation_angle=.6):
    setup = {'RM6': rm6, 'RM5': rm5, 'RM5_hier': rm5_hierarchical, 'KL': kirchhoff_love}[formulation]
    return setup.run(n, method=method, show=False, rotation_angle=rotation_angle)


class HemisphereTests(unittest.TestCase):
    def test_geometry_is_regular_and_preserved_by_knot_insertion(self):
        geometry = open_hemisphere(8)
        points = np.array([(u, v) for v in np.linspace(0., 1., 11) for u in np.linspace(0., 1., 11)])
        positions = geometry.evaluate(points)
        self.assertLess(np.max(abs(np.linalg.norm(positions, axis=1)-10.)), 6e-5)
        np.testing.assert_allclose(positions[points[:, 0] == 0, 1], 0., atol=1e-14)
        np.testing.assert_allclose(positions[points[:, 0] == 1, 0], 0., atol=1e-14)
        du, dv = geometry.derivative(points, [1, 0]), geometry.derivative(points, [0, 1])
        self.assertGreater(np.linalg.norm(np.cross(du, dv), axis=1).min(), 50.)
        for direction in (0, 1):
            knots = np.unique(geometry.knot_vectors[direction])
            geometry.insert_knots(direction, (knots[:-1]+knots[1:])/2)
        np.testing.assert_allclose(geometry.evaluate(points), positions, atol=2e-14)

    def test_all_formulations_rotate_and_have_opposite_radial_responses(self):
        for formulation in ('RM6', 'RM5', 'RM5_hier', 'KL'):
            with self.subTest(formulation=formulation):
                _, original = solve_level(8, formulation, rotation_angle=0.)
                _, rotated = solve_level(8, formulation, rotation_angle=.6)
                self.assertTrue(.08 < rotated['radial_a'] < .10)
                self.assertLess(rotated['residual'], 1e-7)
                np.testing.assert_allclose(rotated['radial_a'], -rotated['radial_b'], rtol=1e-7, atol=1e-10)
                np.testing.assert_allclose(rotated['radial_a'], original['radial_a'], rtol=1e-7, atol=1e-10)

    def test_weak_and_strong_rm_symmetry_approach_the_same_response(self):
        for formulation in ('RM6', 'RM5'):
            with self.subTest(formulation=formulation):
                weak = [solve_level(n, formulation)[1]['radial_a'] for n in (8, 16)]
                strong = [solve_level(n, formulation, 'strong')[1]['radial_a'] for n in (8, 16)]
                self.assertLess(abs(weak[1]-strong[1]), abs(weak[0]-strong[0]))
                self.assertLess(abs(weak[1]-strong[1]), 2e-4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
