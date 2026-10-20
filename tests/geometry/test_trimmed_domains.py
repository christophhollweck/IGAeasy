"""Outer polygons with interior rings must match separate inner BReps."""
import unittest
import numpy as np
from shapely.geometry import box
from shapely.affinity import rotate
from IGAeasy.trimming.Brep import BRep
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties


class TrimmedDomainTests(unittest.TestCase):
    def test_embedded_hole_equals_separate_breps(self):
        outer = box(0., 0., 1., 1.)
        inner = rotate(box(.31, .34, .69, .66), 27., origin=(.5, .5))
        domain = outer.difference(inner)
        meshes = []
        for embedded in (False, True):
            geo = get_geometry.surface(8, 8, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
            space = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
            space.BReps = ([BRep(1, polygon=domain, outerloop=True)] if embedded else
                [BRep(1, polygon=outer, outerloop=True), BRep(2, polygon=inner, outerloop=False)])
            trim(space)
            assign_gp_and_properties(space, full_quad=True)
            area = sum(g.weight_inplane for cell in space.elements for g in cell.gausspoints)
            self.assertAlmostEqual(area, domain.area, places=12)
            for cell in space.elements:
                rectangle = box(cell.u_min, cell.v_min, cell.u_max, cell.v_max)
                expected = rectangle.intersection(domain).area
                integrated = sum(g.weight_inplane for g in cell.gausspoints)
                self.assertAlmostEqual(integrated, expected, places=12)
            meshes.append([(cell.void, cell.trimmed) for cell in space.elements])
        self.assertEqual(meshes[0], meshes[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
