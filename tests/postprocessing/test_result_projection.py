"""Physical projection orthogonality and material-aware plotting geometry."""
import unittest
import numpy as np
from shapely.geometry import box, Polygon
from IGAeasy.trimming.Brep import BRep
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.numerical_integration.map_gp_to_element import get_untrimmed_element_gp
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.postprocessing.projection import project_scalar_results
from IGAeasy.postprocessing.visualization import build_surface_data, plot_surface_results


class ProjectionTests(unittest.TestCase):
    def test_weighted_projection_orthogonality(self):
        knots = [0., 0., 0., 1., 1., 1.]
        cps = [[x, v, 0.] for v in (0., .5, 1.) for x in (0., .5, 1.4)]
        rs = HierarchicalSpace(knots, knots, 2, cps)
        assign_gp_and_properties(rs, full_quad=True)
        cell = rs.elements[0]
        # More samples than coefficients make the weighting observable.
        cell.p = 4
        get_untrimmed_element_gp(cell, full_quad=True)
        cell.p = 2
        for gp in cell.gausspoints:
            gp.sigma_vM = np.exp(gp.xi+.4*gp.eta)
        coefficients = project_scalar_results(rs)
        residual = np.zeros(len(rs.basis))
        for gp in cell.gausspoints:
            values = np.array([b(gp.xi, gp.eta) for b in rs.basis])
            tangents = np.array([b.grad(gp.xi, gp.eta) for b in rs.basis]).T@np.array(cps)
            weight = gp.weight_inplane*np.linalg.norm(np.cross(*tangents))
            residual += weight*values*(values@coefficients-gp.sigma_vM)
        np.testing.assert_allclose(residual, 0., atol=1e-10)

    def test_plot_mesh_excludes_hole_after_local_refinement(self):
        knots = [0., 0., 0., .5, 1., 1., 1.]
        cps = [[u, v, 0.] for v in (0., .25, .75, 1.) for u in (0., .25, .75, 1.)]
        rs = HierarchicalSpace(knots, knots, 2, cps, 2)
        funcs = rs.get_funcs_in_rectangle(0, [[[0., .5], [0., .5]]])
        rs.refine({0: funcs})
        hole = box(.35, .35, .65, .65)
        rs.BReps = [BRep(2, polygon=hole, outerloop=False)]
        trim(rs)
        mesh, boundaries = build_surface_data(rs, samples_per_element=3,
            projected_fields={'stress_von_mises': np.ones(len(rs.basis))})
        domain = box(0, 0, 1, 1).difference(hole)
        for face in mesh.faces.reshape(-1, 4):
            triangle = Polygon(mesh.points[face[1:], :2])
            self.assertLess(triangle.difference(domain).area, 1e-12)
        self.assertGreater(boundaries.n_cells, 0)
        self.assertEqual(boundaries.n_faces_strict, 0)
        self.assertEqual(boundaries.n_verts, 0)
        np.testing.assert_allclose(mesh['stress_von_mises'], 1., atol=1e-12)

    def test_subplot_scalar_fields_are_independent(self):
        import pyvista as pv
        mesh = pv.Plane()
        mesh['uz'] = np.linspace(-.3, 0., mesh.n_points)
        mesh['stress_von_mises'] = np.linspace(100., 1000., mesh.n_points)
        plotter = plot_surface_results(mesh, fields=('uz', 'stress_von_mises'),
                                       show=False, off_screen=True)
        try:
            ranges = []
            arrays = []
            for renderer in plotter.renderers:
                actors = [a for a in renderer.actors.values() if hasattr(a, 'mapper')
                          and hasattr(a.mapper, 'dataset')]
                actor = actors[0]
                ranges.append(actor.mapper.scalar_range)
                arrays.append(actor.mapper.dataset.active_scalars_name)
            np.testing.assert_allclose(ranges, [(-.3, 0.), (100., 1000.)])
            self.assertEqual(arrays, ['uz', 'stress_von_mises'])
        finally:
            plotter.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
