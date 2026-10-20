"""Independent natural moment loads on doubly curved mixed shell boundaries."""
import unittest
import numpy as np

from IGAeasy.examples.shells.verification.manufactured.paraboloid_convergence import solve_level
from IGAeasy.examples.shells.verification.trimmed.all_formulations_convergence import domain
from IGAeasy.manufactured_solutions.gradient_paraboloid_reference import GradientParaboloidReference


class DoublyCurvedMixedShellTests(unittest.TestCase):
    def test_flat_kl_moment_matches_closed_plate_resultant(self):
        reference = GradientParaboloidReference('KL', curvature=(0., 0.))
        u, v = .23, .41
        D = 100.*.1**3/(12*(1-.3**2))
        w = .001*np.sin(np.pi*u)*np.sin(np.pi*v)
        M = D*np.array([[(1+.3)*np.pi**2*w,
                         -(1-.3)*.001*np.pi**2*np.cos(np.pi*u)*np.cos(np.pi*v)],
                        [-(1-.3)*.001*np.pi**2*np.cos(np.pi*u)*np.cos(np.pi*v),
                         (1+.3)*np.pi**2*w]])
        mu = np.array([.6, .8, 0.])
        np.testing.assert_allclose(reference.moment(u-.5, v-.5, 0., mu),
                                   np.r_[M@mu[:2], 0.], atol=1e-14)

    def test_mixed_nitsche_convergence_and_trimmed_local_consistency(self):
        for formulation in ('RM5', 'RM5_hier', 'KL'):
            options = dict(boundary_kind='mixed', surface_curvature=(.4, -.2), mixed=(1.2, -.8),
                           director_amplitude=.0003, enriched=True)
            rows = [solve_level(n, formulation, **options)[1] for n in (4, 8)]
            with self.subTest(formulation=formulation):
                self.assertGreater(np.log2(rows[0]['l2']/rows[1]['l2']), 3.)
                self.assertLess(rows[1]['residual'], 1e-10)
                thb = solve_level(4, formulation, domain=domain, local_refinements=1,
                    refinement_region=(0., 0., .2, .2), **options)[1]
                lr = solve_level(4, formulation, domain=domain, local_refinements=1, family='LR',
                    refinement_region=(0., 0., .2, .2), **options)[1]
                for key in ('l2', 'director_l2'):
                    np.testing.assert_allclose(thb[key], lr[key], rtol=2e-7, atol=1e-12)
                self.assertLess(thb['l2'], 2e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
