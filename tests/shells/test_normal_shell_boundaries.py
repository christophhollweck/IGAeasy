"""Independent traction limits and moving normal supports on doubly curved shells."""
import unittest
import numpy as np

from IGAeasy.examples.shells.verification.manufactured.paraboloid_convergence import solve_level
from IGAeasy.examples.shells.verification.trimmed.all_formulations_convergence import domain
from IGAeasy.manufactured_solutions.rm_paraboloid_reference import ParaboloidRMReference
from IGAeasy.manufactured_solutions.gradient_paraboloid_reference import GradientParaboloidReference


class NormalShellBoundaryTests(unittest.TestCase):
    def test_flat_traction_matches_closed_continuum_resultants(self):
        E, nu, t, A, Q = 100., .3, .1, .001, .0003
        u, v = .23, .41
        mu = np.array([.6, .8, 0.])
        gradient = A*np.pi*np.array([np.cos(np.pi*u)*np.sin(np.pi*v),
                                    np.sin(np.pi*u)*np.cos(np.pi*v)])
        C = E*t/(1-nu**2)
        N = np.array([[C*A*v/10, E*t*A*u/(4*(1+nu))],
                      [E*t*A*u/(4*(1+nu)), nu*C*A*v/10]])
        rm = ParaboloidRMReference((0., 0.), centered=True, enriched=True, director_amplitude=Q)
        shear = (5/6)*E*t/(2*(1+nu))*(gradient+Q*np.array([1+u, .5+v]))
        expected = np.r_[N@mu[:2], shear@mu[:2]]
        np.testing.assert_allclose(rm.traction(u-.5, v-.5, 0., mu), expected, atol=1e-13)
        kl = GradientParaboloidReference('KL', curvature=(0., 0.))
        D = E*t**3/(12*(1-nu**2))
        expected[2] = 2*D*np.pi**2*(gradient@mu[:2])
        np.testing.assert_allclose(kl.traction(u-.5, v-.5, 0., mu), expected, atol=1e-13)

    def test_normal_support_converges_and_rotates_with_geometry(self):
        settings = dict(boundary_kind='normal', surface_curvature=(.4, -.2), mixed=(1.2, -.8),
                        director_amplitude=.0003, enriched=True)
        for formulation in ('RM5', 'RM5_hier', 'KL'):
            with self.subTest(formulation=formulation):
                rows = [solve_level(n, formulation, **settings)[1] for n in (4, 8)]
                original = solve_level(4, formulation, rotation_angle=0., **settings)[1]
                self.assertGreater(np.log2(rows[0]['l2']/rows[1]['l2']), 3.)
                self.assertLess(rows[1]['residual'], 1e-9)
                for key in ('l2', 'director_l2'):
                    np.testing.assert_allclose(original[key], rows[0][key], rtol=2e-6, atol=1e-12)

    def test_normal_support_with_local_trimmed_thb_and_lr(self):
        for formulation in ('RM5', 'RM5_hier', 'KL'):
            rows = [solve_level(4, formulation, boundary_kind='normal', domain=domain,
                local_refinements=1, refinement_region=(0., 0., .2, .2), family=family,
                surface_curvature=(.4, -.2), mixed=(1.2, -.8),
                director_amplitude=.0003, enriched=True)[1] for family in ('THB', 'LR')]
            with self.subTest(formulation=formulation):
                for key in ('l2', 'director_l2'):
                    np.testing.assert_allclose(rows[0][key], rows[1][key], rtol=2e-6, atol=1e-12)
                fine = solve_level(8, formulation, boundary_kind='normal', domain=domain,
                    local_refinements=1, refinement_region=(0., 0., .2, .2),
                    surface_curvature=(.4, -.2), mixed=(1.2, -.8),
                    director_amplitude=.0003, enriched=True)[1]
                self.assertGreater(np.log2(rows[0]['l2']/fine['l2']), 3.)
                self.assertLess(fine['l2'], 5e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
