"""Characterize the bending-free shear-mode space used by the hierarchy."""
import unittest
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.postprocessing.projection import project_scalar_results
from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature
from tests.support.gradient_space_reference import gradient_space_diagnostic


class HierarchicalGradientSpaceTests(unittest.TestCase):
    def test_iga_projection_has_the_same_gradient_gap(self):
        diagnostic = gradient_space_diagnostic()
        target = diagnostic['target']
        for family in ('THB', 'LR'):
            with self.subTest(family=family):
                geo = get_geometry.surface(4, 4, 1, 1, 1, 1, 2, 3, distortion=0, n_waves=0)
                space = (HierarchicalSpace(*geo.knot_vectors, 3, geo.control_points) if family == 'THB'
                         else LRSpline(3, *geo.knot_vectors, geo.control_points))
                assign_gp_and_properties(space, full_quad=True)
                for cell in space.elements:
                    for gp in cell.gausspoints:
                        gp.gradient_probe = target(gp.xi)
                coefficients = project_scalar_results(space, 'gradient_probe')
                error = SurfaceL2Quadrature(space).error(coefficients, lambda x, y, z: target(x))
                self.assertAlmostEqual(error, diagnostic['errors']['same'], places=10)

    def test_gradient_is_not_in_same_maximum_continuity_space(self):
        for n in (4, 8):
            errors = gradient_space_diagnostic(n)['errors']
            self.assertGreater(errors['same'], .01)
            self.assertLess(errors['derivative'], 1e-12)
            self.assertLess(errors['lower_continuity'], 1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
