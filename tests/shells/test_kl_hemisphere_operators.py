"""Independent Cartesian and geometric checks on the curved KL element."""
import unittest
import numpy as np

from IGAeasy.examples.shells.verification.boundary_conditions.hemisphere_kl_diagnostic import (
    cartesian_stiffness, reflection_space, sphere_derivatives, run,
)
from IGAeasy.examples.shells.benchmarks.pinched_hemisphere import kirchhoff_love
from IGAeasy.geometry.get_geometry import open_hemisphere
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.elements.KLshell3p import KLShell
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.splines.batch_evaluation import evaluate_derivatives_batch


class KLHemisphereOperatorTests(unittest.TestCase):
    def setUp(self):
        geometry = open_hemisphere(4)
        self.space = HierarchicalSpace(*geometry.knot_vectors, 3, geometry.control_points)
        assign_gp_and_properties(self.space, LinearElasticMaterial(6.825e7, .3, 1.),
                                 {'t': .04, 'rho': 1.}, full_quad=True)

    def test_independent_cartesian_stiffness_matches_curvilinear_element(self):
        for cell in (self.space.elements[0], self.space.elements[5], self.space.elements[-1]):
            KLShell().element_routine(cell)
            independent = cartesian_stiffness(cell, exact_sphere=False)
            self.assertLess(np.linalg.norm(cell.K_el-independent)/np.linalg.norm(independent), 2e-14)

    def test_strains_match_finite_variation_of_metric_and_curvature(self):
        cell = self.space.elements[5]
        KLShell().element_routine(cell)
        gp = cell.gausspoints[5]
        basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
        data = evaluate_derivatives_batch(basis, [(gp.xi, gp.eta)], cell, max_order=2)
        cps = np.asarray([b.cp for b in basis])
        direction = np.random.default_rng(13).normal(size=cps.shape)

        def forms(points):
            tangents = np.array([data[1, 0][0]@points, data[0, 1][0]@points])
            normal = np.cross(*tangents)
            normal /= np.linalg.norm(normal)
            second = np.array([data[k][0]@points for k in ((2, 0), (0, 2), (1, 1))])
            metric = tangents@tangents.T
            return np.array([metric[0, 0]/2, metric[1, 1]/2, metric[0, 1]]), second@normal

        step = 1e-5
        plus_m, plus_b = forms(cps+step*direction)
        minus_m, minus_b = forms(cps-step*direction)
        Bm, Bb = gp.Bmatrices[0]
        np.testing.assert_allclose(Bm@direction.ravel(), (plus_m-minus_m)/(2*step), rtol=1e-7, atol=1e-7)
        curvature_change = -(plus_b-minus_b)/(2*step)*[1., 1., 2.]
        np.testing.assert_allclose(Bb@direction.ravel(), curvature_change, rtol=1e-7, atol=1e-7)

    def test_exact_sphere_derivatives_and_reflection_space(self):
        tangents, second = sphere_derivatives(.37, .61)
        normal = np.cross(*tangents)
        normal /= np.linalg.norm(normal)
        principal = np.linalg.eigvals(np.linalg.solve(tangents@tangents.T, second@normal))
        np.testing.assert_allclose(abs(principal), [.1, .1], atol=1e-15)
        T = reflection_space(7, 7)
        coefficients = (T@np.arange(T.shape[1])).reshape(7, 7, 3)
        np.testing.assert_array_equal(coefficients[:, 0, 1], 0.)
        np.testing.assert_array_equal(coefficients[:, -1, 0], 0.)
        np.testing.assert_array_equal(coefficients[:, 1, (0, 2)], coefficients[:, 0, (0, 2)])
        np.testing.assert_array_equal(coefficients[:, -2, (1, 2)], coefficients[:, -1, (1, 2)])

    def test_nitsche_approaches_independent_strong_reflection(self):
        differences = []
        for n in (8, 16):
            strong = run(n)
            _, weak = kirchhoff_love.run(n, show=False, rotation_angle=0.)
            differences.append(abs(strong['radial_a']-weak['radial_a']))
            self.assertLess(strong['residual'], 1e-7)
        # These impose different finite-dimensional spaces, not identical DOFs.
        self.assertLess(differences[1], differences[0]/10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
