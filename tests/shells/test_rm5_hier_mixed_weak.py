"""Independent edge moments for selected curved hierarchical RM5 traces."""
import unittest
import numpy as np
from shapely.geometry import box, LineString
from shapely.affinity import rotate
from IGAeasy.manufactured_solutions.shell_verification import solve_gradient_patch as solve_level
from IGAeasy.manufactured_solutions.kl_parabolic_reference import ParabolicRM5HierReference
from IGAeasy.constraints.rm5_hier_weak import WeakRM5HierBoundary
from IGAeasy.elements.RMshell5p_hier import RMShell


class RM5HierMixedTests(unittest.TestCase):
    def test_partial_director_data_can_be_completed_tangentially(self):
        rs, _, _ = solve_level(2, formulation='RM5_hier', power=3)
        cell = next(c for c in rs.elements if c.u_min == 0. and c.v_min == 0.)
        RMShell().element_routine(cell)
        line = LineString([(0.,0.),(1.,0.)])
        field = lambda x,y,z: np.array([0.,0.,0.,.1,0.,0.])
        boundary = WeakRM5HierBoundary(line, field, components=('dx',))
        matrix, rhs, gamma = boundary.integrate_contribution(cell)
        self.assertTrue(np.isfinite(matrix).all() and np.isfinite(rhs).all())
        self.assertGreater(gamma, 0.)
        def normal_direction(x,y,z):
            normal = np.array([-.4*x,0.,1.])
            return np.r_[np.zeros(3), normal/np.linalg.norm(normal)][None,:]
        boundary = WeakRM5HierBoundary(line, field, directions=normal_direction)
        with self.assertRaisesRegex(ValueError, 'not attainable'):
            boundary.integrate_contribution(cell)

    def test_nonzero_patch_on_locally_refined_thb_lr_with_hole(self):
        domain = box(.07,.09,.93,.91).difference(rotate(box(.37,.39,.63,.61),27.,origin=(.5,.5)))
        for family in ('THB','LR'):
            with self.subTest(family=family):
                _, _, row = solve_level(4, formulation='RM5_hier', boundary_kind='mixed_hier',
                    family=family, power=3, angle=.6, local_refinement=True, domain=domain)
                self.assertLess(row['l2'], 1e-8)
                self.assertLess(row['director_l2'], 1e-8)
                self.assertLess(row['patch_residual'], 1e-8)
                self.assertEqual(row['gamma_d_max'], 0.)

    def test_curved_quartic_convergence(self):
        rows = [solve_level(n, formulation='RM5_hier', boundary_kind='mixed_hier',
                            angle=.6, trimmed=True)[2] for n in (4,8)]
        self.assertGreater(np.log2(rows[0]['l2']/rows[1]['l2']), 3.5)
        self.assertGreater(np.log2(rows[0]['director_l2']/rows[1]['director_l2']), 2.7)

    def test_moving_director_direction_and_complementary_moment(self):
        def directions(x,y,z):
            a1 = np.array([1.,0.,.4*x])
            a1 /= np.linalg.norm(a1)
            rows = np.zeros((4,6))
            rows[:3,:3] = np.eye(3)
            rows[3,3:] = a1
            return rows
        _, _, row = solve_level(4, formulation='RM5_hier', boundary_kind='mixed_hier',
            power=3, trimmed=True, hier_directions=directions)
        self.assertLess(row['l2'], 1e-8)
        self.assertLess(row['director_l2'], 1e-8)

    def test_analytic_moment_flat_limit(self):
        reference = ParabolicRM5HierReference(power=3, curvature=0.)
        D = reference.E*reference.thickness**3/(12*(1-reference.nu**2))
        for mu, expected in (([1.,0.,0.],[-6*D*reference.amplitude*.4,0.,0.]),
                             ([0.,1.,0.],[0.,-6*D*reference.nu*reference.amplitude*.4,0.])):
            np.testing.assert_allclose(reference.moment_vector(.4,0.,0.,np.array(mu)), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
