"""Homogeneous clamping isolates boundary-data approximation from bulk errors."""
import unittest
import numpy as np
from IGAeasy.manufactured_solutions.kl_parabolic_reference import PolynomialClampedPlateReference
from IGAeasy.manufactured_solutions.shell_verification import solve_gradient_patch as solve_level


class HomogeneousShellReferenceTests(unittest.TestCase):
    def test_trace_is_zero_on_every_unit_square_edge(self):
        reference = PolynomialClampedPlateReference()
        for coordinate in (.0, .13, .5, .87, 1.):
            for x, y in ((0., coordinate), (1., coordinate), (coordinate, 0.), (coordinate, 1.)):
                np.testing.assert_allclose(reference.field(x, y, 0.), 0., atol=1e-15)

    def test_exact_biquartic_field_in_both_formulations(self):
        for formulation in ('KL', 'RM5_hier'):
            with self.subTest(formulation=formulation):
                _, _, row = solve_level(2, degree=4, angle=.6, formulation=formulation,
                    curvature=0., reference_kind='clamped_plate')
                self.assertLess(row['l2'], 1e-10)
                self.assertLess(row['director_l2'], 1e-9)
                self.assertLess(row['patch_residual'], 1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
