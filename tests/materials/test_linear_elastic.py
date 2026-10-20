"""Physical constitutive responses and equivalent tensor/Voigt representations."""
import unittest
import numpy as np
from IGAeasy.material.linear_elastic import LinearElasticMaterial


class LinearElasticTests(unittest.TestCase):
    def test_plane_stress_response_is_independent_of_transverse_shear_factor(self):
        E, nu, gamma = 210., .3, .02
        for kappa in (1., 5/6, .4):
            material = LinearElasticMaterial(E, nu, 1., kappa)
            C = material.get_C_voigt_cart_2D()
            np.testing.assert_allclose(C @ [0., 0., gamma],
                                       [0., 0., E/(2*(1+nu))*gamma], atol=1e-14)
            # Uniaxial plane stress: the lateral contraction cancels sigma_yy.
            np.testing.assert_allclose(C @ [.01, -nu*.01, 0.], [E*.01, 0., 0.], atol=1e-14)

    def test_shell_plane_stress_order_and_transverse_shear(self):
        material = LinearElasticMaterial(210., .3, 1., .4)
        C = material.get_C_voigt_cart_3D()
        indices = [0, 1, 3]
        np.testing.assert_allclose(C[np.ix_(indices, indices)], material.get_C_voigt_cart_2D())
        np.testing.assert_array_equal(C[2], 0.)
        np.testing.assert_array_equal(C[:, 2], 0.)
        G = material.E/(2*(1+material.nu))
        np.testing.assert_allclose(np.diag(C)[[4, 5]], material.kappa*G)

    def test_tensor_voigt_response_and_energy_on_skew_metric(self):
        material = LinearElasticMaterial(210., .3, 1.)
        metric = np.linalg.inv([[2., .4], [.4, .8]])
        t = .17
        tensors = material.compute_C_tensors_curv_2D(metric, t)
        matrices = material.compute_C_voigt_curv_2D(metric, t)
        strain = np.array([[.02, -.007], [-.007, .01]])
        engineering = np.array([strain[0, 0], strain[1, 1], 2*strain[0, 1]])
        for tensor, matrix in zip(tensors[:2], matrices[:2]):
            stress = np.einsum('ijkl,kl->ij', tensor, strain)
            np.testing.assert_allclose(matrix@engineering, [stress[0, 0], stress[1, 1], stress[0, 1]])
            self.assertAlmostEqual(engineering@matrix@engineering, np.sum(strain*stress), places=13)
            np.testing.assert_allclose(material.tensor4_to_voigt2D(tensor), matrix)
        np.testing.assert_allclose(tensors[2], matrices[2])

    def test_basis_change_preserves_membrane_energy(self):
        material = LinearElasticMaterial(210., .3, 1.)
        basis = np.array([[1.4, .3], [.2, .9]])
        inverse_metric = np.linalg.inv(basis.T@basis)
        cartesian_strain = np.array([[.02, -.007], [-.007, .01]])
        covariant_strain = basis.T@cartesian_strain@basis
        voigt = np.array([.02, .01, -.014])
        tensor, _, _ = material.compute_C_tensors_curv_2D(inverse_metric, 1.)
        energy = np.einsum('ij,ijkl,kl', covariant_strain, tensor, covariant_strain)
        self.assertAlmostEqual(energy, voigt@material.get_C_voigt_cart_2D()@voigt, places=13)


if __name__ == '__main__':
    unittest.main(verbosity=2)
