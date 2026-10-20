"""Mixed curved KL boundaries against independent field and moment data."""
import unittest
import numpy as np
from shapely.geometry import box
from shapely.affinity import rotate
from IGAeasy.manufactured_solutions.shell_verification import solve_gradient_patch as solve_level
from IGAeasy.manufactured_solutions.kl_parabolic_reference import ParabolicKLReference
from IGAeasy.constraints.kl_mixed_weak import lift_boundary_work


class KLMixedTests(unittest.TestCase):
    def test_curved_diaphragm_with_complementary_traction(self):
        axis = np.array([1.,2.,3.])/np.sqrt(14.)
        cross = np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
        R = np.eye(3)+np.sin(.6)*cross+(1-np.cos(.6))*cross@cross
        domain = box(.07,.09,.93,.91).difference(
            rotate(box(.37,.39,.63,.61),27.,origin=(.5,.5)))
        for family in ('THB','LR'):
            _, _, row = solve_level(4, family=family, angle=.6, power=3, local_refinement=True,
                domain=domain, boundary_kind='mixed', kl_displacement_directions=R[:,[0,2]].T)
            self.assertLess(row['l2'], 1e-8)
            self.assertLess(row['director_l2'], 1e-8)
            self.assertLess(row['patch_residual'], 1e-8)

    def test_rejects_uncontrolled_normal_displacement(self):
        with self.assertRaisesRegex(ValueError, 'contain the surface normal'):
            solve_level(2, boundary_kind='mixed', kl_displacement_directions=[[1.,0.,0.]])

    def test_trimmed_cubic_patch_on_both_refined_spaces(self):
        domain = box(.07, .09, .93, .91).difference(
            rotate(box(.37, .39, .63, .61), 27., origin=(.5, .5)))
        for family in ('THB', 'LR'):
            for slope in (False, True):
                with self.subTest(family=family, slope=slope):
                    _, _, row = solve_level(4, family=family, angle=.6, power=3,
                        local_refinement=True, domain=domain, boundary_kind='mixed',
                        normal_rotation=slope)
                    self.assertLess(row['l2'], 1e-8)
                    self.assertLess(row['director_l2'], 1e-7)
                    self.assertLess(row['patch_residual'], 1e-7)

    def test_quartic_convergence(self):
        rows = [solve_level(n, angle=.6, trimmed=True, boundary_kind='mixed')[2]
                for n in (4, 8)]
        self.assertGreater(np.log2(rows[0]['l2']/rows[1]['l2']), 3.6)
        self.assertGreater(np.log2(rows[0]['director_l2']/rows[1]['director_l2']), 2.6)

    def test_small_cut_patch_with_physical_norms(self):
        for fraction in (1e-2, 1e-3):
            with self.subTest(fraction=fraction):
                _, _, row = solve_level(4, angle=.6, power=3, boundary_kind='mixed',
                    domain=box(0., 0., .25+.25*fraction, 1.))
                self.assertLess(row['l2'], 1e-9)
                self.assertLess(row['director_l2'], 1e-8)
                self.assertLess(row['patch_residual'], 1e-9)

    def test_rotation_covariance(self):
        _, flat, a = solve_level(2, boundary_kind='mixed', power=3)
        _, rotated, b = solve_level(2, boundary_kind='mixed', power=3, angle=.6)
        np.testing.assert_allclose(np.linalg.norm(flat.reshape(-1, 3), axis=1),
            np.linalg.norm(rotated.reshape(-1, 3), axis=1), atol=1e-10)
        self.assertAlmostEqual(a['gamma_max']/b['gamma_max'], 1., places=7)

    def test_penalty_bias_decreases(self):
        rows = [solve_level(4, angle=.6, trimmed=True, boundary_kind='mixed',
                    power=3, method='penalty', safety_factor=factor)[2]
                for factor in (40., 400.)]
        self.assertLess(rows[1]['l2'], rows[0]['l2']/5)

    def test_moment_flat_limit(self):
        reference = ParabolicKLReference(curvature=0., power=3)
        u = .4
        D = reference.E*reference.thickness**3/(12*(1-reference.nu**2))
        bending = -6*reference.amplitude*u
        self.assertAlmostEqual(reference.normal_moment(u, 0., 0., np.array([1., 0., 0.])), D*bending)
        self.assertAlmostEqual(reference.normal_moment(u, 0., 0., np.array([0., 1., 0.])), reference.nu*D*bending)

    def test_lifting_preserves_work_and_rejects_uncontrolled_modes(self):
        trace = np.array([[1., .2], [.3, 1.], [.5, .6]])
        energy = np.array([[1., 1.]])
        bulk = energy.T@energy
        consistency = np.array([[2., 2.], [-1., -1.]])
        flux = lift_boundary_work(trace, consistency, bulk, energy)
        np.testing.assert_allclose(trace.T@flux, consistency, atol=1e-14)
        np.testing.assert_allclose(flux@np.array([1., -1.]), 0., atol=1e-14)
        factored = lift_boundary_work(trace, consistency, bulk, energy,
                                     work_factors=(np.eye(2), consistency))
        np.testing.assert_allclose(factored, flux, atol=1e-14)
        with self.assertRaisesRegex(ValueError, 'uncontrolled'):
            lift_boundary_work(trace, np.eye(2), bulk, energy)

    def test_extreme_cut_has_no_spurious_lifting_penalty(self):
        _, _, row = solve_level(4, angle=.6, power=3, boundary_kind='mixed',
                                domain=box(0., 0., .2500025, 1.))
        # The physical cut bound is still large; the old rounded cross product
        # inflated it to 8e24. This guards the lifting, not general cut robustness.
        self.assertGreater(row['gamma_max'], 1e17)
        self.assertLess(row['gamma_max'], 1e20)
        self.assertLess(row['l2'], 1e-3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
