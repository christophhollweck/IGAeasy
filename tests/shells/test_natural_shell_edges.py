"""Independent natural shell loads and genuinely unconstrained curved edges."""
import unittest
import numpy as np
from shapely.geometry import LineString, box

from IGAeasy.geometry.get_geometry import polynomial_shell
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.loads import ShellEdgeLoad
from IGAeasy.assembler import VectorAssembler
from IGAeasy.manufactured_solutions.kl_parabolic_reference import ParabolicKLReference
from IGAeasy.manufactured_solutions.rm_parabolic_reference import ParabolicRMShearReference
from IGAeasy.examples.shells.verification.boundary_conditions.free_edge_convergence import (
    solve_level, trimmed_domain,
)


class NaturalShellEdgeTests(unittest.TestCase):
    def test_analytic_edge_data_have_closed_flat_limits(self):
        u, mu = .4, np.array([.6, .8, 0.])
        kl = ParabolicKLReference(power=3, curvature=0.)
        D = kl.E*kl.thickness**3/(12*(1-kl.nu**2))
        np.testing.assert_allclose(kl.edge_data(u, .3, 0., mu), np.r_[
            0., 0., -6*D*kl.amplitude*mu[0],
            -6*D*kl.amplitude*u*mu[0], -6*kl.nu*D*kl.amplitude*u*mu[1], 0.], atol=1e-14)
        rm = ParabolicRMShearReference('RM5', power=3, curvature=0.)
        G = rm.E/(2*(1+rm.nu))
        N = G*rm.thickness*3*rm.amplitude*u*u
        M = G*rm.thickness**3/12*3*rm.director_amplitude*u*u
        Q = rm.kappa*G*rm.thickness*rm.director_amplitude*u**3
        np.testing.assert_allclose(rm.edge_data(u, .3, 0., mu),
                                   [N*mu[1], N*mu[0], Q*mu[1], M*mu[1], M*mu[0], 0.], atol=1e-14)

    def test_kl_natural_twisting_moment_contains_the_four_corner_forces(self):
        geometry = polynomial_shell(2, 2, curvature=(0., 0.), centered=False)
        space = HierarchicalSpace(*geometry.knot_vectors, 3, geometry.control_points)
        # D=1, nu=.3, w=.01*u*v: M12=-.007, no raw line force.
        def data(x, y, z, mu):
            return np.r_[np.zeros(3), -.007*mu[1], -.007*mu[0], 0.]

        load = ShellEdgeLoad(LineString(box(0., 0., 1., 1.).exterior.coords), data)
        for cell in space.elements:
            cell.bfnums = np.array(sorted(b.id for b in cell.supported_b_splines))
            cell.f_el = load.integrate_contribution(cell)
        force = VectorAssembler(3).assemble(space, 'f_el').reshape(-1, 3)
        expected = np.zeros_like(force)
        for u, v, sign in ((0., 0., 1.), (1., 0., -1.), (1., 1., 1.), (0., 1., -1.)):
            for b in space.basis:
                expected[b.id, 2] += sign*.014*b(u, v)
        np.testing.assert_allclose(force, expected, atol=1e-14)
        self.assertGreater(np.linalg.norm(force), .02)

    def test_curved_free_edges_converge_with_physical_norms(self):
        for formulation in ('RM5', 'KL'):
            with self.subTest(formulation=formulation):
                rows = [solve_level(n, formulation)[1] for n in (4, 8)]
                self.assertGreater(np.log2(rows[0]['l2']/rows[1]['l2']), 3.6)
                director_rate = np.log2(rows[0]['director_l2']/rows[1]['director_l2'])
                self.assertGreater(director_rate, 3.5 if formulation == 'RM5' else 2.8)
                self.assertLess(rows[1]['residual'], 1e-10)
                self.assertLess(rows[1]['energy_error'], rows[0]['energy_error']/8)
                for row in rows:
                    self.assertLess(row['energy_balance'], 1e-10)
                    self.assertLess(row['physical_force_balance'], 1e-10)
                    self.assertLess(row['physical_moment_balance'], 1e-10)
                    self.assertLess(row['reaction_force_error'], 1e-8)
                original = solve_level(4, formulation, rotation_angle=0.)[1]
                for key in ('l2', 'director_l2'):
                    np.testing.assert_allclose(rows[0][key], original[key], rtol=1e-6, atol=1e-12)

    def test_slanted_trimmed_free_edges_on_locally_refined_thb_lr(self):
        for formulation in ('RM5', 'KL'):
            rows = []
            for family in ('THB', 'LR'):
                with self.subTest(formulation=formulation, family=family):
                    space, row = solve_level(3, formulation, family, power=3,
                        domain=trimmed_domain, local_refinement=True)
                    self.assertEqual(len(space.boundary_conditions), 1)
                    self.assertTrue(any(c.trimmed and not c.void for c in space.elements))
                    self.assertGreater(row['min_parametric_fraction'], .05)
                    self.assertLess(row['l2'], 1e-9)
                    self.assertLess(row['director_l2'], 1e-8)
                    self.assertLess(row['patch_residual'], 1e-8)
                    rows.append(row)
            np.testing.assert_allclose(rows[0]['l2'], rows[1]['l2'], rtol=1e-4, atol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
