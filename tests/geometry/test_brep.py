"""Trim roles, hole preservation and material coverage of BRep polygons."""
import unittest
from shapely.geometry import box, Polygon
from shapely.geometry.polygon import orient
from IGAeasy.trimming.Brep import BRep
from IGAeasy.trimming import primitive_shapes
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.trimming.trim_elements import trim


class BRepTests(unittest.TestCase):
    def test_legacy_orientation_and_explicit_role(self):
        for outer in (True, False):
            polygon = orient(box(0, 0, 1, 1), sign=1 if outer else -1)
            self.assertEqual(BRep(1, polygon=polygon).outerloop, outer)
            boundary = BRep(1, polygon=polygon, outerloop=not outer)
            self.assertEqual(boundary.outerloop, not outer)
        boundary.polygon = orient(boundary.polygon, sign=-1)
        boundary.compute_area_and_orientation()
        self.assertFalse(boundary.outerloop)

    def test_holes_survive_construction_and_simplification(self):
        domain = box(0, 0, 1, 1).difference(box(.3, .3, .7, .7))
        boundary = BRep(1, polygon=domain, outerloop=True)
        self.assertAlmostEqual(boundary.area, .84)
        for polygon in (boundary.polygon, boundary.adaptive_sampling(domain),
                        boundary.simplify_polygon_douglas_peucker(domain)):
            self.assertEqual(len(polygon.interiors), 1)
            self.assertTrue(polygon.equals(domain))

    def test_trimming_combined_domain_matches_separate_loops(self):
        outer, hole = box(.1, .1, .9, .9), box(.35, .35, .65, .65)
        domains = []
        for boundaries in ([BRep(1, polygon=outer.difference(hole), outerloop=True)],
                           [BRep(1, polygon=outer, outerloop=True),
                            BRep(2, polygon=hole, outerloop=False)]):
            geometry = get_geometry.surface(4, 4, 1, 1, 1, 1, 1, 2)
            space = HierarchicalSpace(*geometry.knot_vectors, 2, geometry.control_points)
            space.BReps = boundaries
            trim(space)
            area = sum(sum(p.area for p in cell.material_polygons) if cell.trimmed
                       else (cell.u_max-cell.u_min)*(cell.v_max-cell.v_min)
                       for cell in space.elements if not cell.void)
            domains.append(area)
        for area in domains:
            self.assertAlmostEqual(area, outer.area-hole.area)

    def test_invalid_polygon_is_rejected(self):
        for polygon in (Polygon(), Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])):
            with self.assertRaises(ValueError):
                BRep(1, polygon=polygon)

    def test_primitive_shapes_keep_their_roles(self):
        for outer in (True, False):
            for boundary in (primitive_shapes.circle(.5, .5, .2, 1, outer),
                             primitive_shapes.rectangle(0, 0, 1, 1, 1, outer),
                             primitive_shapes.custom_geometry(box(0, 0, 1, 1), 1, outer)):
                self.assertEqual(boundary.outerloop, outer)


if __name__ == '__main__':
    unittest.main(verbosity=2)
