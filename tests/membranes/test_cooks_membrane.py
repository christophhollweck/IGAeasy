"""Physical equivalence, load normalization and supports of both Cook discretizations."""
import unittest
import numpy as np
from IGAeasy.examples.membranes.benchmarks import cooks_membrane as fitted
from IGAeasy.examples.membranes.trimmed import cooks_membrane as trimmed
from IGAeasy.assembler import VectorAssembler
from IGAeasy.postprocessing.boundary_visualization import build_boundary_layers


class CookMembraneTests(unittest.TestCase):
    def test_same_panel_area_resultant_and_support_on_both_meshes(self):
        for setup, is_trimmed in ((fitted, False), (trimmed, True)):
            space, row = setup.run(4, show=False)
            self.assertEqual(any(cell.trimmed for cell in space.elements), is_trimmed)
            area = sum(gp.weight_inplane*np.linalg.norm(np.cross(*gp.surface_tangents))
                       for cell in space.elements if not cell.void for gp in cell.gausspoints)
            self.assertAlmostEqual(area, 1440., places=8)
            force = VectorAssembler(2).assemble(space, 'f_el').reshape(-1, 2).sum(axis=0)
            np.testing.assert_allclose(force, [0., 1.], atol=1e-12)
            self.assertLess(row['residual'], 1e-10)
            self.assertIn('Supports', build_boundary_layers(space, space.boundary_conditions))

    def test_refined_fitted_and_trimmed_midpoint_responses_agree(self):
        rows = [setup.run(16, show=False)[1] for setup in (fitted, trimmed)]
        self.assertLess(abs(rows[0]['displacement']-rows[1]['displacement']), 2e-6)
        for row in rows:
            self.assertGreater(row['normalized_displacement'], 23.90)
            self.assertLess(row['normalized_displacement'], 23.96)


if __name__ == '__main__':
    unittest.main(verbosity=2)
