"""Physical director constraints and derivative/rotation coupling."""
import unittest
import numpy as np
from IGAeasy.constraints.linear_constraints import LinearConstraints
from IGAeasy.constraints.shell_constraints import constrain_director_at_point, constrain_affine_director_boundary
from IGAeasy.constraints.linear_constraints import solve_linear_constraints
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace


class ShellConstraintTests(unittest.TestCase):
    def test_RM5_hier_clamped_plate_has_zero_boundary_director(self):
        from IGAeasy.geometry import get_geometry
        from tests.support import plate_hier_reference as ex1_plate_RM5_hier
        geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
        rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
        ex1_plate_RM5_hier.solve(rs, use_sparse=True, compute_stress=False,
                               verbose=False, boundary='clamped')
        values = np.concatenate([b.dof_values for b in sorted(rs.basis, key=lambda b: b.id)])
        self.assertLess(rs.eval_displacement(.5, .5)[2], 0.)
        for t in np.linspace(.01, .99, 7):
            for u, v in ((0., t), (1., t), (t, 0.), (t, 1.)):
                np.testing.assert_allclose(rs.eval_displacement(u, v), 0., atol=1e-11)
                for direction in ([1., 0., 0.], [0., 1., 0.]):
                    point = LinearConstraints.for_control_points(
                        ('x', 'y', 'z', 'r_a1', 'r_a2'), len(rs.basis))
                    constrain_director_at_point(rs, point, u, v, direction, 'RM5_hier')
                    np.testing.assert_allclose(point.matrices()[0]@values, 0., atol=1e-11)

    def test_full_affine_edge_director_trace(self):
        rs = self.space()
        original = LinearConstraints.for_control_points(('x', 'y', 'z'), len(rs.basis))
        conditions = constrain_affine_director_boundary(rs, original, 'umin', [1., 0., 0.], 'KL')
        C, _ = conditions.matrices()
        self.assertEqual(C.shape[0], 3)
        u, _ = solve_linear_constraints(np.eye(3*len(rs.basis)),
                                        np.arange(3*len(rs.basis), dtype=float), conditions)
        for v in np.linspace(0., 1., 17):
            point = LinearConstraints.for_control_points(('x', 'y', 'z'), len(rs.basis))
            constrain_director_at_point(rs, point, 0., v, [1., 0., 0.], 'KL')
            np.testing.assert_allclose(point.matrices()[0]@u, 0., atol=1e-11)
        self.assertEqual(original.matrices()[0].shape[0], 0)

    def space(self):
        knots = [0., 0., 0., 1., 1., 1.]
        cps = [[2*u, 3*v, 0.] for v in (0., .5, 1.) for u in (0., .5, 1.)]
        return HierarchicalSpace(knots, knots, 2, cps)

    def test_physical_director_for_all_formulations(self):
        layouts = {'KL': ('x', 'y', 'z'), 'RM6': ('x', 'y', 'z', 'rx', 'ry', 'rz')}
        for formulation in ('KL', 'RM6', 'RM5', 'RM5_v2', 'RM5_hier'):
            rs = self.space()
            layout = layouts.get(formulation, ('x', 'y', 'z', 'r_a1', 'r_a2'))
            conditions = LinearConstraints.for_control_points(layout, len(rs.basis))
            constrain_director_at_point(rs, conditions, 0., .4, [1., 0., 0.], formulation)
            coefficients = []
            for basis in rs.basis:
                # w=.1*x; physical director correction has x component .2.
                extras = ([] if formulation == 'KL' else
                          [0., .2, 0.] if formulation == 'RM6' else
                          [.4, 0.] if formulation == 'RM5' else [.1, 0.])
                coefficients.extend([0., 0., .1*basis.cp[0], *extras])
            C, _ = conditions.matrices()
            expected = -.1 if formulation == 'KL' else .1 if formulation == 'RM5_hier' else .2
            np.testing.assert_allclose(C@coefficients, [expected], atol=1e-13)

    def test_normal_direction_is_rejected(self):
        rs = self.space()
        conditions = LinearConstraints.for_control_points(('x', 'y', 'z'), len(rs.basis))
        with self.assertRaisesRegex(ValueError, 'tangential'):
            constrain_director_at_point(rs, conditions, .2, .3, [0., 0., 1.], 'KL')
        self.assertEqual(conditions.matrices()[0].shape[0], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
