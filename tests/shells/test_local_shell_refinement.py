"""Curved shell accuracy and geometry across genuine local refinement transitions."""
import unittest
import numpy as np

from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.examples.shells.verification.manufactured.paraboloid_convergence import solve_level
from IGAeasy.examples.shells.verification.trimmed.all_formulations_convergence import domain


class LocalShellRefinementTests(unittest.TestCase):
    def test_locally_refined_mesh_sequence_converges_and_preserves_geometry(self):
        rotation = rotation_matrix(.6)
        for formulation in ('RM5', 'RM5_hier', 'KL'):
            settings = dict(surface_curvature=(.4, -.2), mixed=(1.2, -.8),
                            director_amplitude=.0003, enriched=True, stabilization='split')
            for family in ('THB', 'LR'):
                with self.subTest(formulation=formulation, family=family):
                    # Shrink the whole background mesh as well as the local cells.
                    # Refining only a low-error corner need not reduce a global L2 norm.
                    settings.update(family=family, local_refinements=1,
                                    refinement_region=(0., 0., .2, .2))
                    _, coarse = solve_level(4, formulation, **settings)
                    space, fine = solve_level(8, formulation, **settings)
                    widths = {round(cell.u_max-cell.u_min, 12) for cell in space.elements}
                    self.assertGreater(len(widths), 1)
                    self.assertGreater(fine['n_basis'], coarse['n_basis'])
                    self.assertGreater(np.log2(coarse['l2']/fine['l2']), 3.)
                    self.assertLess(fine['residual'], 1e-10)
                    if family == 'THB':
                        self.assertTrue(any(basis.trunc for basis in space.basis))

                    # Test exact geometry separately from the solved displacement field.
                    for cell in space.elements:
                        u, v = (cell.u_min+cell.u_max)/2, (cell.v_min+cell.v_max)/2
                        x, y = u-.5, v-.5
                        expected = rotation@[x, y, -.4*x*x+.2*y*y+1.2*x*x*y-.8*x*y*y]
                        position = sum((basis(u, v, cell=cell)*basis.cp
                                        for basis in cell.supported_b_splines), np.zeros(3))
                        np.testing.assert_allclose(position, expected, atol=2e-14)

    def test_trimmed_local_spaces_agree_with_both_boundaries_present(self):
        for formulation in ('RM5', 'RM5_hier', 'KL'):
            rows = []
            for family in ('THB', 'LR'):
                space, row = solve_level(
                    4, formulation, family=family, domain=domain,
                    surface_curvature=(.4, -.2), mixed=(1.2, -.8),
                    director_amplitude=.0003, enriched=True, stabilization='split',
                    local_refinements=1, refinement_region=(0., 0., .2, .2)
                )
                rows.append(row)
                with self.subTest(formulation=formulation, family=family):
                    self.assertEqual(len(space.boundary_conditions), 2)
                    self.assertTrue(any(cell.trimmed and not cell.void for cell in space.elements))
                    self.assertGreater(len({cell.u_max-cell.u_min for cell in space.elements}), 1)
                    self.assertLess(row['residual'], 1e-10)
                    self.assertLess(row['l2'], 2e-6)
            with self.subTest(formulation=formulation):
                for key in ('l2', 'director_l2'):
                    np.testing.assert_allclose(rows[0][key], rows[1][key], rtol=2e-7, atol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
