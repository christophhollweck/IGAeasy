"""Independent loads and exact continuum strains for the curved RM study."""
import unittest
import numpy as np
from IGAeasy.manufactured_solutions.rm_parabolic_reference import ParabolicRMShearReference
from IGAeasy.elements.RMshell6p import RMShell
from IGAeasy.material.linear_elastic import LinearElasticMaterial


class CurvedRMReferenceTests(unittest.TestCase):
    def test_loads_are_resultant_derivatives_and_have_the_flat_limit(self):
        for formulation in ('RM6', 'RM5', 'RM5_v2'):
            reference = ParabolicRMShearReference(formulation)
            for u in (.17, .43, .89):
                g = 1+4*reference.curvature**2*u*u
                loads = -np.imag(reference.resultants(u+1e-30j))/1e-30/np.sqrt(g)
                loads[1] += reference.kappa*reference.E/(2*(1+reference.nu))*reference.thickness*reference.director_amplitude*u**reference.power
                body = reference.body(u, .3)
                self.assertAlmostEqual(body[1], loads[0], places=14)
                if formulation == 'RM6':
                    tangent = np.array([1., 0., 2*reference.curvature*u])/np.sqrt(g)
                    np.testing.assert_allclose(body[3:], -loads[1]*tangent, rtol=1e-12, atol=1e-15)
                else:
                    self.assertEqual(body[3], 0.)
                    self.assertAlmostEqual(body[4], loads[1], places=14)
            flat = ParabolicRMShearReference(formulation, curvature=0., director_amplitude=0.)
            expected = -flat.E/(2*(1+flat.nu))*flat.thickness*flat.amplitude*12*.4**2
            self.assertAlmostEqual(flat.body(.4, .3)[1], expected, places=14)
            np.testing.assert_allclose(flat.body(.4, .3)[3:], 0., atol=1e-16)

    def test_rm6_continuum_strain_stress_and_zero_drilling(self):
        reference = ParabolicRMShearReference()
        shell = RMShell()
        material = LinearElasticMaterial(reference.E, reference.nu, 1.)
        for u in (.17, .43, .89):
            c, t = reference.curvature, reference.thickness
            s, g = 2*c*u, 1+4*c*c*u*u
            tangent = np.array([1., 0., s])/np.sqrt(g)
            normal = np.array([-s, 0., 1.])/np.sqrt(g)
            normal_u = -2*c/g*tangent
            field = lambda u: reference.field(u, .3, c*u*u)
            value = field(u)
            derivative = np.imag(field(u+1e-30j))/1e-30
            for zeta in (-.8, 0., .8):
                stretch = 1-zeta*t/2*2*c/g**1.5
                J = np.array([stretch*np.sqrt(g)*tangent, [0., 1., 0.], t/2*normal])
                inverse = np.linalg.inv(J)
                normal_derivatives = (inverse@np.array([normal_u, np.zeros(3), np.zeros(3)])).T.ravel()
                strain = np.zeros(6)
                for N, gradient, coefficients in ((1., np.zeros(3), value),
                                                   (0., inverse[:, 0], derivative)):
                    B = shell.compute_B_matrix(1, 6, *[np.array([x]) for x in gradient],
                                               np.array([N]), *normal_derivatives,
                                               *inverse[:, 2], *normal, zeta, t)
                    strain += B@coefficients
                q = reference.director_amplitude*u**reference.power
                q_slope = reference.director_amplitude*reference.power*u**(reference.power-1)
                shear = (reference.amplitude*reference.power*u**(reference.power-1)+zeta*t/2*q_slope)/(np.sqrt(g)*stretch)
                expected = np.array([0., 0., 0., shear*tangent[0]+q*normal[0],
                                     shear*tangent[2]+q*normal[2], 0.])
                np.testing.assert_allclose(strain, expected, rtol=1e-12, atol=1e-15)
                rotation = shell.get_rotationsmatrix(tangent, [0., 1., 0.], normal)
                stress = rotation@material.get_C_voigt_cart_3D()@rotation.T@strain
                expected_stress = np.array([0., 0., 0.,
                    shear*tangent[0]+reference.kappa*q*normal[0],
                    shear*tangent[2]+reference.kappa*q*normal[2], 0.])
                np.testing.assert_allclose(stress, reference.E/(2*(1+reference.nu))*expected_stress,
                                           rtol=1e-12, atol=1e-14)
            spin = .5*np.cross(normal, tangent/np.sqrt(g))@derivative[:3]
            self.assertAlmostEqual(spin-normal@value[3:], 0., places=14)


if __name__ == '__main__':
    unittest.main(verbosity=2)
