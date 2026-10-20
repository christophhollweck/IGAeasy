"""Independent flat-limit loads and doubly curved KL/hierarchical RM convergence."""
import unittest
import numpy as np
from IGAeasy.manufactured_solutions.gradient_paraboloid_reference import GradientParaboloidReference
from IGAeasy.examples.shells.verification.manufactured.paraboloid_convergence import solve_level


class GradientParaboloidTests(unittest.TestCase):
    def test_flat_limit_matches_plate_equilibrium(self):
        E, nu, t, A = 100., .3, .1, .001
        D = E*t**3/(12*(1-nu**2))
        shear = (5/6)*E*t/(2*(1+nu))
        for formulation in ('KL', 'RM5_hier'):
            reference = GradientParaboloidReference(formulation, curvature=(0., 0.))
            for u, v in ((.23, .41), (.61, .72)):
                w = A*np.sin(np.pi*u)*np.sin(np.pi*v)
                fy = -E*t*A/(4*(1+nu))-nu*E*t*A/(10*(1-nu**2))
                expected = [0., fy, 4*D*np.pi**4*w]
                if formulation == 'RM5_hier':
                    expected.extend([
                        -2*D*A*np.pi**3*np.cos(np.pi*u)*np.sin(np.pi*v)+shear*reference.eta[0],
                        -2*D*A*np.pi**3*np.sin(np.pi*u)*np.cos(np.pi*v)+shear*reference.eta[1]])
                np.testing.assert_allclose(reference.body(u, v), expected, rtol=1e-12, atol=1e-14)

    def test_saddle_convergence_and_general_rotation(self):
        for formulation in ('KL', 'RM5_hier'):
            rows = [solve_level(n, formulation, surface_curvature=(1., -1.))[1] for n in (4, 8)]
            original = solve_level(4, formulation, rotation_angle=0., surface_curvature=(1., -1.))[1]
            with self.subTest(formulation=formulation):
                self.assertGreater(np.log2(rows[0]['l2']/rows[1]['l2']), 3.3)
                self.assertGreater(np.log2(rows[0]['director_l2']/rows[1]['director_l2']), 2.)
                self.assertLess(rows[1]['l2'], 1e-7)
                self.assertLess(rows[1]['residual'], 1e-11)
                np.testing.assert_allclose(original['l2'], rows[0]['l2'], rtol=2e-7, atol=1e-13)
                np.testing.assert_allclose(original['director_l2'], rows[0]['director_l2'], rtol=2e-7, atol=1e-13)


if __name__ == '__main__':
    unittest.main(verbosity=2)
