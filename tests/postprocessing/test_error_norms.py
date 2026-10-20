"""Physical-coordinate L2 and H1 error validation."""
import unittest
import numpy as np
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.error_norms.norms_poisson import compute_L2_H1_errors


class ErrorNormTests(unittest.TestCase):
    def space(self, kind, cps):
        knots = np.array([0, 0, 0, 1, 1, 1.])
        space = HierarchicalSpace(knots, knots, 2, cps) if kind == 'THB' else LRSpline(2, knots, knots, cps)
        assign_gp_and_properties(space, full_quad=True)
        return space

    def test_affine_physical_linear_field_is_exact(self):
        for kind in ('THB', 'LR'):
            cps = np.array([[1+2*u+.4*v, -1+.3*u+3*v, 0]
                            for v in (0, .5, 1) for u in (0, .5, 1)])
            space = self.space(kind, cps)
            for b in space.basis:
                b.scalar_value = 1+2*b.cp[0]-3*b.cp[1]
            errors = compute_L2_H1_errors(space, lambda x, y: 1+2*x-3*y,
                                          lambda x, y: [2, -3], quadrature_order=6)
            np.testing.assert_allclose(errors, 0, atol=2e-13)

    def test_offset_error_matches_analytic_physical_norms(self):
        cps = [[1+2*u, -1+3*v, 0] for v in (0, .5, 1) for u in (0, .5, 1)]
        space = self.space('THB', cps)
        for b in space.basis:
            b.scalar_value = b.cp[0]+1
        errors = compute_L2_H1_errors(space, lambda x, y: x, lambda x, y: [1, 0], quadrature_order=6)
        np.testing.assert_allclose(errors, [np.sqrt(3/13), np.sqrt(3/16)], atol=2e-13)

    def test_nonaffine_geometry_uses_physical_measure(self):
        # Bernstein coefficients for x(u)=u+0.4*u^2.
        cps = [[x, v, 0] for v in (0, .5, 1) for x in (0, .5, 1.4)]
        space = self.space('THB', cps)
        for b in space.basis:
            b.scalar_value = b.cp[0]+1
        errors = compute_L2_H1_errors(space, lambda x, y: x, lambda x, y: [1, 0], quadrature_order=8)
        mean_x_squared = 1.4**2/3
        np.testing.assert_allclose(errors, [1/np.sqrt(mean_x_squared), 1/np.sqrt(mean_x_squared+1)], atol=2e-12)

    def test_absolute_displacement_L2_preserves_scalar_and_quadrature(self):
        cps = [[2*u, 3*v, 0] for v in (0, .5, 1) for u in (0, .5, 1)]
        for kind in ('THB', 'LR'):
            space = self.space(kind, cps)
            points = [el.gausspoints for el in space.elements]
            for b in space.basis:
                b.displacements = np.array([0., 0., b.cp[0]+1])
                b.scalar_value = 17.
            error, h1 = compute_L2_H1_errors(
                space, lambda x, y: x, quadrature_order=6,
                relative=False, field_component=2)
            self.assertAlmostEqual(error, np.sqrt(6), places=12)
            self.assertIsNone(h1)
            self.assertTrue(all(b.scalar_value == 17 for b in space.basis))
            for el, original_points in zip(space.elements, points):
                self.assertIs(el.gausspoints, original_points)


if __name__ == '__main__':
    unittest.main(verbosity=2)
