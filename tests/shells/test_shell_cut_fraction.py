"""Physical patch errors on small-cut shell cells with diagonal scaling."""
import unittest
import numpy as np
from tests.support.shell_cut_reference import run_case


class ShellCutFractionTests(unittest.TestCase):
    def test_curved_rotated_rm6_rigid_motion_with_small_cuts(self):
        for family in ('THB', 'LR'):
            for method in ('nitsche', 'penalty'):
                with self.subTest(family=family, method=method):
                    _, _, row = run_case(1e-5, 'RM6', method, family,
                                         local_refinement=True, curvature=.2)
                    self.assertLess(row['patch_residual'], 1e-11)
                    self.assertLess(row['l2'], 1e-8)
                    self.assertLess(row['director_l2'], 1e-7)
                    self.assertEqual(row['iterative_info'], 0)
                    self.assertLess(row['solution_difference'], 1e-8)
                    self.assertLess(row['director_difference'], 1e-7)

    def test_small_cut_rotated_locally_refined_patch(self):
        for family in ('THB', 'LR'):
            for formulation in ('RM6', 'RM5', 'RM5_v2'):
                with self.subTest(family=family, formulation=formulation):
                    _, _, row = run_case(1e-5, formulation, family=family, local_refinement=True)
                    self.assertTrue(np.isfinite(row['gamma_max']))
                    self.assertGreater(row['gamma_max'], 0.)
                    self.assertLess(row['l2'], 1e-9)
                    self.assertLess(row['director_l2'], 1e-9)
                    self.assertLess(row['patch_residual'], 1e-11)
                    self.assertLess(row['scaled_residual'], 1e-11)
                    self.assertEqual(row['iterative_info'], 0)
                    self.assertLess(row['solution_difference'], 1e-9)
                    self.assertLess(row['director_difference'], 1e-9)

    def test_cubic_patch_and_penalty_error(self):
        for formulation in ('RM6', 'RM5', 'RM5_v2'):
            with self.subTest(formulation=formulation):
                _, _, nitsche = run_case(1e-5, formulation, degree=3)
                _, _, penalty = run_case(1e-5, formulation, 'penalty', degree=3)
                self.assertLess(nitsche['l2'], 1e-9)
                self.assertLess(nitsche['director_l2'], 1e-9)
                self.assertLess(nitsche['patch_residual'], 1e-11)
                self.assertLess(penalty['l2'], 1e-5)
                self.assertGreater(penalty['l2'], 100*nitsche['l2'])
                self.assertLess(penalty['scaled_residual'], 1e-10)
                for row in (nitsche, penalty):
                    self.assertEqual(row['iterative_info'], 0)
                    self.assertLess(row['solution_difference'], 1e-9)
                    self.assertLess(row['director_difference'], 1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
