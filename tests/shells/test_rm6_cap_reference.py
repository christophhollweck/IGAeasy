"""Independent RM6 double-curvature equilibrium and zero-drilling field."""
import unittest
import numpy as np
from shapely.geometry import box
from IGAeasy.manufactured_solutions.rm6_cap_reference import RM6CapReference
from IGAeasy.examples.shells.verification.manufactured.rm6_cap_convergence import solve_level


class RM6CapReferenceTests(unittest.TestCase):
    def test_loads_match_flux_divergence_and_normal_spin_is_zero(self):
        reference = RM6CapReference()
        a, b = reference.curvature
        for u, v in ((.21, .37), (.63, .81)):
            J, _, _, moment_value = reference.fluxes(u, v)
            divergence = np.zeros((2, 3))
            step = 1e-5
            for axis in range(2):
                delta = np.eye(2)[axis]*step
                plus = reference.fluxes(*(np.array([u, v])+delta))
                minus = reference.fluxes(*(np.array([u, v])-delta))
                divergence[0] += (plus[0]*plus[1][axis]-minus[0]*minus[1][axis])/(2*step)
                divergence[1] += (plus[0]*plus[2][axis]-minus[0]*minus[2][axis])/(2*step)
            expected = np.r_[-divergence[0]/J, moment_value-divergence[1]/J]
            np.testing.assert_allclose(reference.body(u, v), expected, rtol=1e-7, atol=1e-10)
            normal = np.array([-2*a*u, -2*b*v, 1.])/J
            # grad_s(A*h^2) = 2*A*h*(I-nn.T)*e_z, hence n.curl_s(u)=0.
            gradient = 2*reference.amplitude*(a*u*u+b*v*v)*(np.eye(3)-np.outer(normal, normal))[:, 2]
            self.assertAlmostEqual(normal@np.cross(gradient, [0., 0., 1.]), 0., places=14)
            self.assertGreater(np.linalg.norm(reference.body(u, v)[3:]), 1e-5)

    def test_loaded_saddle_converges_and_drilling_energy_vanishes(self):
        rows = [solve_level(n)[1] for n in (4, 8)]
        self.assertGreater(np.log2(rows[0]['l2']/rows[1]['l2']), 3.5)
        self.assertGreater(np.log2(rows[0]['director_l2']/rows[1]['director_l2']), 3.5)
        self.assertLess(rows[1]['drilling_energy'], rows[0]['drilling_energy']/20)
        self.assertLess(rows[1]['drilling_fraction'], 1e-8)
        self.assertLess(rows[1]['residual'], 1e-10)

    def test_rotation_and_moderately_trimmed_thb_lr(self):
        settings = dict(domain=box(.1, .1, .9, .9))
        rows = [solve_level(4, family, **settings)[1] for family in ('THB', 'LR')]
        original = solve_level(4, rotation_angle=0., **settings)[1]
        for row in rows:
            for key in ('l2', 'director_l2', 'drilling_energy'):
                np.testing.assert_allclose(row[key], original[key], rtol=1e-6, atol=1e-13)


if __name__ == '__main__':
    unittest.main(verbosity=2)
