"""Parametric trace selection for rotated, refined and trimmed spline spaces."""
import unittest
import numpy as np
from shapely.geometry import box
from IGAeasy.trimming.Brep import BRep
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.constraints.boundary_selection import select_patch_boundary
from IGAeasy.trimming.trim_elements import trim


class BoundarySelectionTests(unittest.TestCase):
    def space(self, kind):
        knots = [0., 0., 0., .5, 1., 1., 1.]
        cps = [[u, v, 0.] for v in (0., .25, .75, 1.) for u in (0., .25, .75, 1.)]
        return (HierarchicalSpace(knots, knots, 2, cps, 2) if kind == 'THB'
                else LRSpline(2, knots, knots, cps))

    def test_rotation_does_not_change_trace_ids(self):
        for kind in ('THB', 'LR'):
            rs = self.space(kind)
            original = {edge: select_patch_boundary(rs, edge) for edge in ('umin', 'umax', 'vmin', 'vmax')}
            self.assertTrue(all(len(ids) == 4 for ids in original.values()))
            for b in rs.basis:
                x, y, z = b.cp
                b.cp = np.array([y, z, x])
            for edge, ids in original.items():
                self.assertEqual(select_patch_boundary(rs, edge), ids)

    def test_trace_partition_after_refinement(self):
        for kind in ('THB', 'LR'):
            rs = self.space(kind)
            if kind == 'THB':
                rs.refine({0: rs.get_funcs_in_rectangle(0, [[[0., .5], [0., .5]]])})
            else:
                rs.refine_by_structured_mesh([rs.basis[0]])
                rs.get_connectivity()
            selected = select_patch_boundary(rs, 'vmin')
            for u in np.linspace(.01, .99, 11):
                self.assertAlmostEqual(sum(b(u, 0.) for b in rs.basis if b.id in selected), 1., places=11)

    def test_trimmed_patch_edge_has_no_void_trace(self):
        rs = self.space('THB')
        rs.BReps = [BRep(1, polygon=box(-.1, -.1, 1.1, .1), outerloop=False)]
        trim(rs)
        self.assertEqual(select_patch_boundary(rs, 'vmin'), set())
        self.assertTrue(select_patch_boundary(rs, 'vmax'))
        with self.assertRaises(ValueError):
            select_patch_boundary(rs, 'xmin')


if __name__ == '__main__':
    unittest.main(verbosity=2)
