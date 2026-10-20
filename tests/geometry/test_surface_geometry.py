"""Surface geometry must use the supplied side of an interior knot line."""
import unittest
import numpy as np
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.differential_geometry.derivatives import (
    get_full_geometric_data, get_local_basis_and_variations,
)
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline


class SurfaceGeometryTests(unittest.TestCase):
    def test_rotated_c1_surface_has_correct_one_sided_curvature(self):
        ku = [0., 0., 0., .5, 1., 1., 1.]
        kv = [0., 0., 0., 1., 1., 1.]
        # X=(u,v,max(u-.5,0)^2): continuous tangents, discontinuous curvature.
        cps = np.array([[u, v, z] for v in (0., .5, 1.)
                        for u, z in zip((0., .25, .75, 1.), (0., 0., 0., .25))])
        for family in ('THB', 'LR'):
            for angle in (0., .6):
                R = rotation_matrix(angle)
                space = (HierarchicalSpace(ku, kv, 2, cps@R.T) if family == 'THB'
                         else LRSpline(2, ku, kv, cps@R.T))
                for cell in space.elements:
                    with self.subTest(family=family, angle=angle, left=cell.u_min == 0.):
                        curvature = 0. if cell.u_min == 0. else 2.
                        data = get_local_basis_and_variations(cell, .5, .3)
                        np.testing.assert_allclose(data[:3], R.T, atol=2e-14)
                        np.testing.assert_allclose(data[3], R@[0., 0., curvature], atol=2e-14)
                        np.testing.assert_allclose(data[7], R@[-curvature, 0., 0.], atol=2e-14)
                        _, _, _, metric, second_form, _ = get_full_geometric_data(cell, .5, .3)
                        np.testing.assert_allclose(metric, np.eye(2), atol=2e-14)
                        np.testing.assert_allclose(second_form, [[curvature, 0.], [0., 0.]], atol=2e-14)


if __name__ == '__main__':
    unittest.main(verbosity=2)
