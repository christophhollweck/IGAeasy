"""Small positive strain modes and true null traction modes are distinct."""
import unittest
from types import SimpleNamespace
import numpy as np
from shapely.geometry import LineString
from IGAeasy.constraints.shell_weak import WeakRM6Boundary
from IGAeasy.elements.RMshell6p import RMShell
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from tests.shells.test_shell_batch_evaluation import space


class ShellEnergyRankTests(unittest.TestCase):
    def test_rm6_factor_reproduces_bulk_energy_and_checks_thickness_rule(self):
        rule = np.polynomial.legendre.leggauss(3)
        for family in ('THB', 'LR'):
            with self.subTest(family=family):
                rs = space(family)
                assign_gp_and_properties(rs, material_model=LinearElasticMaterial(100., .3, 1.),
                                         properties={'t': .1, 'rho': 1.}, full_quad=True)
                cell = max(rs.elements, key=lambda c: len(c.supported_b_splines))
                RMShell().element_routine(cell, rule)
                boundary = WeakRM6Boundary(LineString([(0., 0.), (1., 0.)]),
                                           lambda x, y, z: np.zeros(6), rule)
                factor = boundary._bulk_energy_factor(cell)
                np.testing.assert_allclose(factor.T@factor, cell.K_el, rtol=1e-11, atol=1e-11)
                mismatched = WeakRM6Boundary(LineString([(0., 0.), (1., 0.)]),
                                             lambda x, y, z: np.zeros(6),
                                             np.polynomial.legendre.leggauss(2))
                with self.assertRaisesRegex(ValueError, 'thickness rule'):
                    mismatched._bulk_energy_factor(cell)

    def boundary(self):
        return WeakRM6Boundary(LineString([(0., 0.), (1., 0.)]),
                               lambda x, y, z: np.zeros(6), ([0.], [2.]))

    def test_small_positive_mode_uses_factor_bound(self):
        # Nearly dependent columns have order-one diagonal energies, while
        # their difference has energy too small to recover from A.T @ A.
        factor = np.array([[1., 1.], [0., 1e-8]])
        flux = np.array([[1., -1.]])
        cell = SimpleNamespace(K_el=factor.T@factor)
        _, _, gamma = self.boundary()._assemble(
            cell, [(1., np.array([[1., 0.]]), flux, np.zeros(1))],
            2, energy_factor=factor)
        expected = 4.*np.linalg.norm(flux@np.linalg.inv(factor))**2
        np.testing.assert_allclose(gamma, expected, rtol=1e-12)

    def test_true_null_mode_with_nonzero_flux_is_rejected(self):
        factor = np.array([[1., 1.], [0., 0.]])
        flux = np.array([[1., -1.]])
        cell = SimpleNamespace(K_el=factor.T@factor)
        with self.assertRaisesRegex(ValueError, 'Uncontrolled shell traction'):
            self.boundary()._assemble(
                cell, [(1., np.array([[1., 0.]]), flux, np.zeros(1))],
                2, energy_factor=factor)

    def test_small_null_flux_is_not_hidden_by_its_gram_matrix(self):
        factor = np.array([[1., 1.], [0., 0.]])
        flux = np.array([[1., 1.], [1e-5, -1e-5]])
        cell = SimpleNamespace(K_el=factor.T@factor)
        with self.assertRaisesRegex(ValueError, 'Uncontrolled shell traction'):
            self.boundary()._assemble(
                cell, [(1., np.eye(2), flux, np.zeros(2))],
                2, energy_factor=factor)


if __name__ == '__main__':
    unittest.main(verbosity=2)
