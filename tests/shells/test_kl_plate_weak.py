"""Mixed KL traces: corner work, refinement and approximation order."""
import unittest
import numpy as np
from shapely.geometry import box, LineString
from shapely.affinity import rotate
from tests.support.kl_plate_reference import solve_level
from IGAeasy.manufactured_solutions.kl_plate_reference import KLPlateReference
from IGAeasy.elements.KLshell3p import KLShell
from IGAeasy.assembler import MatrixAssembler
from IGAeasy.constraints.kl_plate_weak import WeakKLPlateBoundary


class KLPlateWeakTests(unittest.TestCase):
    def test_rejects_curved_geometry(self):
        rs, _, _ = solve_level(2, degree=2, reference_kind='twist', angle=0.)
        for basis in rs.basis:
            basis.cp[2] += .1*basis.cp[0]**2
        boundary = WeakKLPlateBoundary(LineString([(0., 0.), (1., 0.)]),
                                       lambda x, y, z: np.zeros(6))
        cell = next(cell for cell in rs.elements if cell.v_min == 0.)
        with self.assertRaisesRegex(ValueError, 'affine'):
            boundary.integrate_contribution(cell)

    def test_twist_corner_reactions(self):
        rs, coefficients, _ = solve_level(2, degree=2, reference_kind='twist', angle=0.)
        for cell in rs.elements:
            KLShell().element_routine(cell)
        bulk = MatrixAssembler(dof_per_cp=3).assemble(rs, 'K_el', use_sparse=True)
        expected = np.zeros_like(coefficients)
        reference = KLPlateReference('twist')
        force = 2*reference.rigidity*(1-reference.nu)*.01
        for u, v, sign in ((0., 0., 1), (1., 0., -1), (1., 1., 1), (0., 1., -1)):
            for basis in rs.basis:
                expected[basis.id, 2] += sign*force*basis(u, v)
        np.testing.assert_allclose(bulk@coefficients.ravel(), expected.ravel(), atol=1e-12)

    def test_twisting_patch_on_rotated_trimmed_refined_spaces(self):
        domain = box(.07, .09, .93, .91).difference(
            rotate(box(.37, .39, .63, .61), 27, origin=(.5, .5)))
        for family in ('THB', 'LR'):
            for slope in (False, True):
                with self.subTest(family=family, slope=slope):
                    _, _, row = solve_level(2, family=family, degree=2,
                        reference_kind='twist', normal_rotation=slope,
                        domain=domain, local_refinement=True)
                    self.assertLess(row['l2'], 1e-11)
                    self.assertLess(row['director_l2'], 1e-10)
                    self.assertTrue(np.isnan(row['center']))

    def test_sine_convergence(self):
        for degree, expected in ((2, 2), (3, 4)):
            with self.subTest(degree=degree):
                rows = [solve_level(n, degree=degree, reference_kind='sine')[2]
                        for n in (4, 8)]
                rate = np.log2(rows[0]['l2']/rows[1]['l2'])
                director_rate = np.log2(rows[0]['director_l2']/rows[1]['director_l2'])
                self.assertGreater(rate, expected-.4)
                self.assertLess(rate, expected+.6)
                self.assertGreater(director_rate, degree-.4)

    def test_rotation_covariance(self):
        _, flat, a = solve_level(2, degree=2, reference_kind='sine', angle=0.)
        _, rotated, b = solve_level(2, degree=2, reference_kind='sine', angle=.6)
        np.testing.assert_allclose(np.linalg.norm(flat, axis=1),
                                   np.linalg.norm(rotated, axis=1), atol=1e-12)
        self.assertAlmostEqual(a['l2'], b['l2'], places=12)
        self.assertAlmostEqual(a['gamma_max']/b['gamma_max'], 1., places=8)

    def test_navier_series_resolution(self):
        a = KLPlateReference(n_terms=64).values(.5, .5)[0]
        b = KLPlateReference(n_terms=128).values(.5, .5)[0]
        self.assertLess(abs(a-b), 1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
