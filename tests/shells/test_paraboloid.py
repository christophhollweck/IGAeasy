"""Exact doubly curved geometry and independent manufactured RM equilibrium."""
import unittest
import numpy as np
from IGAeasy.geometry import get_geometry
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.differential_geometry.derivatives import get_full_geometric_data
from IGAeasy.examples.shells.verification.manufactured.paraboloid_convergence import solve_level


class ParaboloidTests(unittest.TestCase):
    def test_centered_cap_and_field_are_symmetric(self):
        from IGAeasy.manufactured_solutions.rm_paraboloid_reference import ParaboloidRMReference
        rotation = rotation_matrix(.6)
        geometry = get_geometry.paraboloid(2, 2, 3, (1., 1.), centered=True)
        parameters = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [.5, .5]])
        points = geometry.evaluate(parameters)
        np.testing.assert_allclose(points[:4, 2], -.5, atol=1e-14)
        np.testing.assert_allclose(points[4], 0., atol=1e-14)
        reference = ParaboloidRMReference((1., 1.), rotation=rotation, centered=True)
        parameters = np.array([[.2, .3], [.8, .3], [.2, .7], [.8, .7]])
        for position in geometry.evaluate(parameters):
            expected = rotation@[0., 0., .001*np.sin(.2*np.pi)*np.sin(.3*np.pi)]
            np.testing.assert_allclose(reference.displacement(*(rotation@position)), expected, atol=1e-15)
        space, _ = solve_level(4)
        values = [rotation.T@space.eval_displacement(u, v) for u, v in parameters]
        np.testing.assert_allclose(values[1], values[0]*[-1., 1., 1.], atol=1e-13)
        np.testing.assert_allclose(values[2], values[0]*[1., -1., 1.], atol=1e-13)

    def test_exact_geometry_metric_and_gaussian_curvature(self):
        a, b = .2, .3
        rotation = rotation_matrix(.6)
        for degree in (2, 3):
            geometry = get_geometry.paraboloid(3, 2, degree, (a, b))
            space = HierarchicalSpace(*geometry.knot_vectors, degree,
                                      geometry.control_points@rotation.T)
            for cell in space.elements:
                u, v = (cell.u_min+cell.u_max)/2, (cell.v_min+cell.v_max)/2
                basis = list(cell.supported_b_splines)
                point = sum((bf(u, v, cell=cell)*bf.cp for bf in basis), np.zeros(3))
                np.testing.assert_allclose(point, rotation@[u, v, a*u*u+b*v*v], atol=2e-14)
                _, _, _, metric, second, _ = get_full_geometric_data(cell, u, v)
                slope = np.array([2*a*u, 2*b*v])
                expected_metric = np.eye(2)+np.outer(slope, slope)
                np.testing.assert_allclose(metric, expected_metric, atol=3e-14)
                np.testing.assert_allclose(second, np.diag([2*a, 2*b])/np.sqrt(1+slope@slope), atol=3e-13)
                gaussian = np.linalg.det(second)/np.linalg.det(metric)
                self.assertAlmostEqual(gaussian, 4*a*b/(1+slope@slope)**2, places=12)

    def test_manufactured_displacement_converges_at_cubic_rate(self):
        for formulation in ('RM5', 'RM5_v2'):
            rows = [solve_level(n, formulation)[1] for n in (2, 4, 8)]
            with self.subTest(formulation=formulation):
                self.assertGreater(np.log2(rows[-2]['l2']/rows[-1]['l2']), 3.5)
                self.assertLess(rows[-1]['l2'], 3e-8)
                self.assertLess(rows[-1]['director_l2'], 4e-8)
                self.assertTrue(all(row['residual'] < 1e-11 for row in rows))

    def test_physical_error_is_invariant_under_general_rotation(self):
        for formulation in ('RM5', 'RM5_v2'):
            original = solve_level(4, formulation, rotation_angle=0.)[1]
            rotated = solve_level(4, formulation, rotation_angle=.6)[1]
            for field in ('l2', 'director_l2'):
                np.testing.assert_allclose(original[field], rotated[field], rtol=2e-7, atol=1e-14)


if __name__ == '__main__':
    unittest.main(verbosity=2)
