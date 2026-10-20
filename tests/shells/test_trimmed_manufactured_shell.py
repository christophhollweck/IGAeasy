"""Moderately cut curved shells with an analytic solution on outer and hole boundaries."""
import unittest
import numpy as np
from IGAeasy.examples.shells.verification.manufactured.paraboloid_convergence import solve_level
from IGAeasy.examples.shells.verification.trimmed.all_formulations_convergence import domain


class TrimmedManufacturedShellTests(unittest.TestCase):
    def solve(self, n, formulation, family='THB'):
        return solve_level(n, formulation, surface_curvature=(.4, -.2), mixed=(1.2, -.8),
                           director_amplitude=.0003, enriched=True, domain=domain, family=family,
                           stabilization='split')

    def test_all_formulations_converge_without_tiny_cut_fractions(self):
        for formulation in ('RM5', 'RM5_hier', 'KL'):
            rows = [self.solve(n, formulation)[1] for n in (4, 8)]
            with self.subTest(formulation=formulation):
                self.assertGreater(np.log2(rows[0]['l2']/rows[1]['l2']), 3.)
                self.assertLess(rows[1]['l2'], 1.5e-7)
                self.assertTrue(all(1/9-1e-12 <= row['min_parametric_fraction'] < 1. for row in rows))
                self.assertTrue(all(row['residual'] < 1e-10 for row in rows))

    def test_lr_and_thb_agree_on_the_same_trimmed_tensor_product_space(self):
        for formulation in ('RM5', 'RM5_hier', 'KL'):
            thb, thb_row = self.solve(4, formulation)
            lr, lr_row = self.solve(4, formulation, family='LR')
            with self.subTest(formulation=formulation):
                self.assertEqual(len(thb.boundary_conditions), 2)
                self.assertEqual(len(lr.boundary_conditions), 2)
                for name in ('l2', 'director_l2'):
                    np.testing.assert_allclose(thb_row[name], lr_row[name], rtol=2e-7, atol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
