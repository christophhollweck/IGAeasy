"""Boundary clipping, support mapping and interactive visibility controls."""
import unittest
import numpy as np
from shapely.geometry import box, LineString, Point
from shapely.ops import unary_union
from IGAeasy.trimming.Brep import BRep
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.constraints.shell_weak import WeakRM5Boundary
from IGAeasy.postprocessing.boundary_visualization import build_boundary_layers
from IGAeasy.postprocessing.visualization import build_surface_data, plot_surface_results, plot_solution


class BoundaryVisualizationTests(unittest.TestCase):
    def test_polygonal_hole_support_has_no_missing_segments(self):
        space = self.space(n_elements=8)
        hole = Point(.5, .5).buffer(.205, quad_segs=12)
        domain = box(1/12, 1/12, 11/12, 11/12).difference(hole)
        space.BReps = [BRep(1, polygon=domain, outerloop=True)]
        trim(space)
        line = LineString(hole.exterior.coords)
        condition = WeakRM5Boundary(line, lambda x, y, z: np.zeros(6))
        data, _, _ = build_boundary_layers(space, [condition], deformation_scale=0.)['Supports'][0]
        curves = []
        offset = 0
        while offset < len(data.lines):
            count = data.lines[offset]
            curves.append(LineString(data.points[data.lines[offset+1:offset+1+count], :2]))
            offset += count+1
        coverage = unary_union(curves)
        self.assertLess(line.difference(coverage.buffer(1e-10)).length, 1e-10)
        self.assertAlmostEqual(coverage.length, line.length, places=9)

    def test_support_line_does_not_bridge_a_trimmed_hole(self):
        space = self.space(hole=True)
        condition = WeakRM5Boundary(LineString([(.5, 0.), (.5, 1.)]), lambda x, y, z: np.zeros(6))
        data, _, _ = build_boundary_layers(space, [condition], deformation_scale=0.)['Supports'][0]
        offset = 0
        while offset < len(data.lines):
            count = data.lines[offset]
            y = data.points[data.lines[offset+1:offset+1+count], 1]
            self.assertTrue(y.max() <= .3+1e-12 or y.min() >= .7-1e-12)
            offset += count+1

    def test_support_line_covers_the_entire_edge(self):
        space = self.space()
        condition = WeakRM5Boundary(LineString([(0., 1.), (0., 0.)]), lambda x, y, z: np.zeros(6))
        data, _, label = build_boundary_layers(space, [condition], deformation_scale=0.)['Supports'][0]
        self.assertIn('Clamped', label)
        self.assertEqual(data.n_faces_strict, 0)
        np.testing.assert_allclose(data.bounds[:4], [0., 0., 0., 1.], atol=1e-12)

    def test_deformation_checkbox_moves_surface_edges_and_supports_together(self):
        space = self.space(hole=True)
        for basis in space.basis:
            basis.displacements = np.array([.01, 0., .02])
        condition = WeakRM5Boundary(LineString([(0., 1.), (0., 0.)]), lambda x, y, z: np.zeros(6))
        plotter = plot_solution(space, fields=('uz', 'displacement_magnitude'), deformation_scale=3.,
                                boundary_conditions=[condition], show=False, off_screen=True)
        try:
            widget = plotter.button_widgets[-1]
            for state, z in ((0, 0.), (1, .06), (0, 0.)):
                widget.GetRepresentation().SetState(state)
                widget.InvokeEvent('StateChangedEvent')
                for renderer, field in zip(plotter.renderers, ('uz', 'displacement_magnitude')):
                    for name in ('result_surface', 'overlay:Element edges:0', 'overlay:Supports:0',
                                 'overlay:Trim boundaries:0'):
                        np.testing.assert_allclose(renderer.actors[name].mapper.dataset.points[:, 2], z, atol=1e-12)
                    np.testing.assert_allclose(renderer.actors['result_surface'].mapper.dataset['uz'], .02)
                    self.assertEqual(renderer.actors['result_surface'].mapper.GetArrayName(), field)
        finally:
            plotter.close()

    def test_moving_normal_and_kl_reflection_supports_have_distinct_labels(self):
        from IGAeasy.constraints.kl_mixed_weak import WeakKLMixedBoundary
        from IGAeasy.constraints.kl_symmetry_weak import WeakKLSymmetryBoundary
        space = self.space()
        line = LineString([(0., 1.), (0., 0.)])
        normal = WeakKLMixedBoundary(line, lambda x, y, z: np.zeros(6), displacement_directions='normal')
        reflection = WeakKLSymmetryBoundary(line, [1., 0., 0.])
        layers = build_boundary_layers(space, [normal, reflection])
        labels = [entry[2] for entry in layers['Supports']]
        self.assertTrue(any('Normal support' in label for label in labels))
        self.assertTrue(any('Reflection symmetry' in label for label in labels))

    def test_shared_edges_are_drawn_once_including_hanging_nodes_and_trimming(self):
        for family in ('THB', 'LR'):
            for refined in (False, True):
                space = self.space(family, hole=refined, refined=refined)
                _, edges = build_surface_data(space, fields=('uz',))
                curves = []
                for cell in space.elements:
                    if cell.void or not getattr(cell, 'active', True):
                        continue
                    polygons = (cell.material_polygons if cell.trimmed else
                                [box(cell.u_min, cell.v_min, cell.u_max, cell.v_max)])
                    curves.extend(polygon.boundary for polygon in polygons)
                expected = unary_union(curves).length
                length = 0.
                offset = 0
                while offset < len(edges.lines):
                    count = edges.lines[offset]
                    points = edges.points[edges.lines[offset+1:offset+1+count]]
                    length += np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
                    offset += count+1
                with self.subTest(family=family, refined=refined):
                    self.assertAlmostEqual(length, expected, places=11)

    def space(self, family='THB', hole=False, refined=False, n_elements=2):
        geometry = get_geometry.surface(n_elements, n_elements, 1, 1, 1, 1, 1, 2)
        space = (HierarchicalSpace(*geometry.knot_vectors, 2, geometry.control_points)
                 if family == 'THB' else LRSpline(2, *geometry.knot_vectors, geometry.control_points))
        if refined:
            selected = min(space.basis, key=lambda b: np.linalg.norm(np.asarray(b.cp)-[.25, .25, 0.]))
            if family == 'THB':
                space.refine({0: [selected.loc_id]})
            else:
                space.refine_by_structured_mesh([selected])
                space.get_connectivity()
        for basis in space.basis:
            basis.displacements = np.array([0., 0., .02])
        if hole:
            space.BReps = [BRep(1, polygon=box(.3, .3, .7, .7), outerloop=False)]
            trim(space)
        return space

    def test_trim_curves_follow_material_boundary_and_displayed_displacement(self):
        material = box(0., 0., 1., 1.).difference(box(.3, .3, .7, .7))
        for family in ('THB', 'LR'):
            space = self.space(family, hole=True, refined=True)
            layers = build_boundary_layers(space, deformation_scale=3.)
            curves = layers['Trim boundaries'][0][0]
            self.assertGreater(curves.n_lines, 0)
            np.testing.assert_allclose(curves.points[:, 2], .06, atol=1e-14)
            self.assertTrue(all(material.boundary.distance(Point(*p[:2])) < 1e-12 for p in curves.points))
            self.assertTrue(any(abs(p[0]-.3) < 1e-12 for p in curves.points))

    def test_translation_support_does_not_become_a_clamp(self):
        space = self.space()
        condition = WeakRM5Boundary(LineString([(0., 0.), (1., 0.)]),
            lambda x, y, z: np.zeros(6), components=('uz',))
        layers = build_boundary_layers(space, [condition], deformation_scale=0.)
        data, _, label = layers['Supports'][0]
        self.assertIn('Translation support', label)
        self.assertGreater(data.n_lines, 0)
        np.testing.assert_allclose(data.points[:, 1:], 0., atol=1e-12)

    def test_checkbox_changes_every_subplot_without_rebuilding_data(self):
        space = self.space()
        condition = WeakRM5Boundary(LineString(box(0., 0., 1., 1.).exterior.coords),
            lambda x, y, z: np.zeros(6))
        layers = build_boundary_layers(space, [condition])
        mesh, edges = build_surface_data(space, fields=('uz', 'displacement_magnitude'))
        plotter = plot_surface_results(mesh, edges, overlays=layers, show=False, off_screen=True)
        try:
            self.assertEqual(len(plotter.button_widgets), 4)
            actors = [renderer.actors['overlay:Supports:0'] for renderer in plotter.renderers]
            self.assertTrue(all(not actor.GetVisibility() for actor in actors))
            widget = plotter.button_widgets[2]
            widget.GetRepresentation().SetState(1)
            widget.InvokeEvent('StateChangedEvent')
            self.assertTrue(all(actor.GetVisibility() for actor in actors))
            widget.GetRepresentation().SetState(0)
            widget.InvokeEvent('StateChangedEvent')
            self.assertTrue(all(not actor.GetVisibility() for actor in actors))

            color, surface = plotter.button_widgets[:2]
            color.GetRepresentation().SetState(0)
            color.InvokeEvent('StateChangedEvent')
            surfaces = [renderer.actors['result_surface'] for renderer in plotter.renderers]
            self.assertTrue(all(not actor.GetMapper().GetScalarVisibility() for actor in surfaces))
            self.assertTrue(all(not bar.GetVisibility() for bar in plotter.scalar_bars.values()))
            surface.GetRepresentation().SetState(0)
            surface.InvokeEvent('StateChangedEvent')
            self.assertTrue(all(not actor.GetVisibility() for actor in surfaces))
            self.assertTrue(all(renderer.actors['overlay:Element edges:0'].GetVisibility()
                                for renderer in plotter.renderers))
            color.GetRepresentation().SetState(1)
            color.InvokeEvent('StateChangedEvent')
            self.assertTrue(all(not bar.GetVisibility() for bar in plotter.scalar_bars.values()))
            surface.GetRepresentation().SetState(1)
            surface.InvokeEvent('StateChangedEvent')
            self.assertTrue(all(actor.GetVisibility() for actor in surfaces))
            self.assertTrue(all(bar.GetVisibility() for bar in plotter.scalar_bars.values()))
        finally:
            plotter.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
