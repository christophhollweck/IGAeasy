"""Loaded curved-shell convergence, trimming and physical rotation covariance."""
import unittest
import numpy as np
from shapely.geometry import box
from shapely.affinity import rotate
from IGAeasy.manufactured_solutions.shell_verification import solve_rm_patch as solve_level
from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature, SurfaceDirectorL2Quadrature
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.solver import iterative_solver


class CurvedRMConvergenceTests(unittest.TestCase):
    def test_cubic_patch_on_locally_refined_thb_lr_with_rotated_hole(self):
        domain = box(.07, .09, .93, .91).difference(
            rotate(box(.37, .39, .63, .61), 27., origin=(.5, .5)))
        for family in ('THB', 'LR'):
            for formulation in ('RM5', 'RM5_v2'):
                with self.subTest(family=family, formulation=formulation):
                    _, _, row = solve_level(2, formulation, family=family, power=3,
                                             local_refinement=True, domain=domain)
                    self.assertLess(row['l2'], 1e-9)
                    self.assertLess(row['director_l2'], 1e-9)
                    self.assertLess(row['patch_residual'], 1e-9)

    def test_cubic_basis_has_fourth_order_displacement_convergence(self):
        for formulation in ('RM6', 'RM5', 'RM5_v2'):
            with self.subTest(formulation=formulation):
                rows = [solve_level(n, formulation)[2] for n in (4, 8)]
                rate = np.log2(rows[0]['l2']/rows[1]['l2'])
                self.assertGreater(rate, 3.5)
                self.assertLess(rate, 4.5)
                director_rate = np.log2(rows[0]['director_l2']/rows[1]['director_l2'])
                self.assertGreater(director_rate, 3.5)
                self.assertLess(director_rate, 4.5)
                self.assertLess(rows[1]['residual'], 1e-11)

    def test_loaded_solution_is_covariant_under_spatial_rotation(self):
        angle = .6
        axis = np.array([1., 2., 3.])/np.sqrt(14.)
        cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
                          [-axis[1], axis[0], 0.]])
        rotation = np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*cross@cross
        for formulation in ('RM6', 'RM5', 'RM5_v2'):
            with self.subTest(formulation=formulation):
                _, original, _ = solve_level(2, formulation, angle=0.)
                rs, actual, _ = solve_level(2, formulation, angle=angle)
                expected = original.copy()
                expected[:, :3] = original[:, :3]@rotation.T
                if formulation == 'RM6':
                    expected[:, 3:] = original[:, 3:]@rotation.T
                norm = SurfaceL2Quadrature(rs, order=6)
                director_norm = SurfaceDirectorL2Quadrature(rs, formulation, order=6)
                self.assertLess(norm.error(actual[:, :3], expected[:, :3]), 1e-9)
                self.assertLess(director_norm.error(actual, expected), 1e-9)

    def test_flat_rm6_polynomial_limit_is_exact(self):
        _, _, row = solve_level(2, 'RM6', curvature=0., power=3)
        self.assertLess(row['l2'], 1e-12)
        self.assertLess(row['director_l2'], 1e-12)
        self.assertLess(row['rotation_l2'], 1e-12)

    def test_penalty_error_decreases_but_is_not_an_exact_patch(self):
        for formulation in ('RM6', 'RM5', 'RM5_v2'):
            with self.subTest(formulation=formulation):
                rows = [solve_level(n, formulation, 'penalty')[2] for n in (2, 4)]
                self.assertLess(rows[1]['l2'], rows[0]['l2'])
                self.assertLess(rows[1]['director_l2'], rows[0]['director_l2'])
                self.assertGreater(rows[1]['l2'], 1e-10)
                self.assertLess(rows[1]['residual'], 1e-11)

    def test_loaded_small_cuts_do_not_amplify_the_physical_error(self):
        for formulation in ('RM6', 'RM5', 'RM5_v2'):
            with self.subTest(formulation=formulation):
                rows = []
                for fraction in (.01, .00001):
                    rs, direct, row = solve_level(
                        2, formulation, domain=box(.5-fraction/2, 0., 1., 1.))
                    rows.append(row)
                self.assertLess(rows[1]['l2'], 2*rows[0]['l2'])
                self.assertLess(rows[1]['director_l2'], 2*rows[0]['director_l2'])
                self.assertLess(rows[1]['residual'], 1e-11)
                self.assertGreater(rows[1]['gamma_max'], rows[0]['gamma_max'])
                dof = direct.shape[1]
                K = MatrixAssembler(dof_per_cp=dof).assemble(rs, 'K_el', use_sparse=True)
                f = VectorAssembler(dof_per_cp=dof).assemble(rs, 'f_el')
                iterative = iterative_solver.solve(K, f, preconditioner='ILU', scaling=True,
                                                    rtol=1e-10, atol=0.).reshape(direct.shape)
                norm = SurfaceL2Quadrature(rs, order=6)
                director_norm = SurfaceDirectorL2Quadrature(rs, formulation, order=6)
                self.assertLess(norm.error(iterative[:, :3], direct[:, :3]), 1e-8)
                self.assertLess(director_norm.error(iterative, direct), 1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
