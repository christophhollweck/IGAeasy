"""Physical surface L2 norms against independent area and field integrals."""
import unittest
import numpy as np
from scipy.integrate import dblquad
from shapely.geometry import box
from IGAeasy.trimming.Brep import BRep
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature


class SurfaceNormTests(unittest.TestCase):
    def space(self, curved=False, trimmed=False):
        geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
        rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
        angle = .6
        R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                      [-np.sin(angle), 0., np.cos(angle)]])
        for b in rs.basis:
            cp = np.asarray(b.cp).copy()
            if curved:
                cp[2] = .2*cp[0]*cp[1]
            else:
                cp *= [2., 3., 1.]
            b.cp = R@cp
        if trimmed:
            rs.BReps = [BRep(1, polygon=box(.1, .1, .9, .9), outerloop=True)]
        trim(rs)
        assign_gp_and_properties(rs, properties={'marker': 123}, full_quad=True)
        return rs

    def test_rotated_vector_norm_and_preserved_element_data(self):
        for trimmed in (False, True):
            rs = self.space(trimmed=trimmed)
            points = [cell.gausspoints for cell in rs.elements]
            norm = SurfaceL2Quadrature(rs, order=5)
            vector = np.tile([1., 2., 3.], (len(rs.basis), 1))
            area = 6*(.8**2 if trimmed else 1.)
            self.assertAlmostEqual(norm.error(vector), np.sqrt(14*area), places=11)
            self.assertAlmostEqual(norm.error(vector, vector), 0., places=12)
            for cell, original in zip(rs.elements, points):
                self.assertIs(cell.gausspoints, original)
                self.assertTrue(all(g.marker == 123 for g in cell.gausspoints))

    def test_curved_area_against_independent_integral(self):
        rs = self.space(curved=True)
        expected, _ = dblquad(lambda v, u: np.sqrt(1+.04*(u*u+v*v)),
                              0., 1., lambda u: 0., lambda u: 1.)
        norm = SurfaceL2Quadrature(rs, order=5)
        self.assertAlmostEqual(norm.error(np.ones(len(rs.basis)))**2, expected, places=11)

    def test_physical_callback_and_relative_error(self):
        rs = self.space()
        norm = SurfaceL2Quadrature(rs)
        coefficients = np.array([b.cp for b in sorted(rs.basis, key=lambda b: b.id)])
        self.assertLess(norm.error(coefficients, lambda x, y, z: [x, y, z], relative=True), 1e-13)
        self.assertAlmostEqual(norm.error(2*coefficients, coefficients, relative=True), 1., places=12)
        with self.assertRaises(ValueError):
            norm.error(coefficients, np.zeros_like(coefficients), relative=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
