"""Polyline quadrature preserves length and oriented tangents."""
import unittest
from types import SimpleNamespace
import numpy as np
from shapely.geometry import LineString
from IGAeasy.constraints.line_helpers import get_boundary_quadrature


class BoundaryQuadratureTests(unittest.TestCase):
    def test_diagonal_tensor_trace_with_explicit_order(self):
        cell = SimpleNamespace(void=False, trimmed=False, p=2,
                               u_min=0., u_max=1., v_min=0., v_max=1.)
        quadrature = get_boundary_quadrature(
            LineString([(0., 0.), (1., 1.)]), cell, quadrature_order=5)
        integral = sum(weight*point.x**4*point.y**4
                       for point, weight, _ in quadrature)
        self.assertAlmostEqual(integral, np.sqrt(2.)/9., places=14)

    def test_bends_repeated_vertices_and_reversed_orientation(self):
        cell = SimpleNamespace(void=False, trimmed=False, p=2,
                               u_min=0., u_max=1., v_min=0., v_max=1.)
        coordinates = [(0., .2), (.5, .2), (.5, .2), (.5, .8), (1., .8)]
        for vertices in (coordinates, coordinates[::-1]):
            line = LineString(vertices)
            quadrature = get_boundary_quadrature(line, cell)
            self.assertAlmostEqual(sum(weight for _, weight, _ in quadrature), line.length)
            integrated_tangent = sum(weight*tangent for _, weight, tangent in quadrature)
            np.testing.assert_allclose(integrated_tangent, np.subtract(vertices[-1], vertices[0]), atol=1e-14)
            for point, weight, tangent in quadrature:
                self.assertGreater(weight, 0.)
                self.assertAlmostEqual(np.linalg.norm(tangent), 1.)
                self.assertLess(line.distance(point), 1e-14)


if __name__ == '__main__':
    unittest.main(verbosity=2)
