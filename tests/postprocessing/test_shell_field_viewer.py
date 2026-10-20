"""Independent section fields and interactive selection of shell results."""
import unittest
from types import SimpleNamespace
import numpy as np

from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.elements import RMshell6p, RMshell5p, RMshell5p_hier, KLshell3p
from IGAeasy.postprocessing.postprocessing import update_cps
from IGAeasy.postprocessing.shell_fields import ShellResultFields
from IGAeasy.postprocessing.visualization import build_surface_data, plot_solution


class ShellFieldViewerTests(unittest.TestCase):
    def space(self, formulation='RM5'):
        geometry = get_geometry.surface(2, 2, 1., 1., 1., 1., 1, 2, distortion=0., n_waves=0)
        space = HierarchicalSpace(*geometry.knot_vectors, 2, geometry.control_points)
        modules = {'RM6': RMshell6p, 'RM5': RMshell5p, 'RM5_hier': RMshell5p_hier, 'KL': KLshell3p}
        shell = KLshell3p.KLShell() if formulation == 'KL' else modules[formulation].RMShell()
        dof = shell.dof_per_cp
        assign_gp_and_properties(space, material_model=LinearElasticMaterial(100., .3, 1.),
                                 properties={'t': .2, 'rho': 1.}, full_quad=True)
        coefficients = np.zeros((len(space.basis), dof))
        for basis in space.basis:
            x, y, _ = basis.cp
            coefficients[basis.id, :3] = [.002*x+.0005*y, -.001*y+.0005*x, 0.]
        update_cps(space, coefficients.ravel(), dof_per_cp=dof, formulation=formulation)
        for cell in space.elements:
            if formulation == 'RM6':
                shell.element_routine(cell, np.polynomial.legendre.leggauss(3))
            else:
                shell.element_routine(cell)
        return space

    def test_all_formulations_recover_constant_plane_stress_and_tensor_strain(self):
        expected = {'stress_11': 100/(1-.3**2)*(.002-.3*.001),
                    'stress_22': 100/(1-.3**2)*(-.001+.3*.002),
                    'stress_12': 100/(2*1.3)*.001,
                    'strain_11': .002, 'strain_22': -.001, 'strain_12': .0005}
        for formulation in ('RM6', 'RM5', 'RM5_hier', 'KL'):
            space = self.space(formulation)
            fields = ShellResultFields(space)
            mesh, _ = build_surface_data(space, fields=('uz',))
            self.assertFalse(fields.ready)
            for representation in fields.representations:
                for field, value in expected.items():
                    with self.subTest(formulation=formulation, mode=representation, field=field):
                        np.testing.assert_allclose(fields.sample(mesh, field, 'Top', representation), value,
                                                   rtol=1e-8, atol=1e-12)

    def test_section_faces_have_opposite_bending_stress_and_strain(self):
        material = LinearElasticMaterial(100., 0., 1.)
        gp = SimpleNamespace(N=np.zeros(3), M=np.array([100*.2**3*.03/12, 0., 0.]),
                             Q=np.zeros(2), t=.2, material_model=material)
        for layer, stress in (('Bottom', -.3), ('Middle', 0.), ('Top', .3)):
            self.assertAlmostEqual(ShellResultFields.section_value(gp, 'stress_11', layer), stress)
            self.assertAlmostEqual(ShellResultFields.section_value(gp, 'strain_11', layer), stress/100)

    def test_linear_strain_is_projected_exactly_while_cell_means_keep_jumps(self):
        space = self.space()
        coefficients = np.zeros((len(space.basis), 5))
        for basis in space.basis:
            # Exact Bernstein/B-spline coefficient of u_x = .001*x**2.
            coefficients[basis.id, 0] = .001*basis.knots_u[1]*basis.knots_u[2]
        update_cps(space, coefficients.ravel(), dof_per_cp=5, formulation='RM5')
        fields = ShellResultFields(space)
        mesh, _ = build_surface_data(space, fields=('uz',))
        expected = .002*mesh.field_data['parametric_coordinates'][:, 0]
        projected = fields.sample(mesh, 'strain_11', representation='Global L2')
        np.testing.assert_allclose(projected, expected, atol=1e-12)
        means = fields.sample(mesh, 'strain_11', representation='Cell mean')
        self.assertGreater(np.max(np.abs(means-expected)), .0004)
        for index, cell in enumerate(space.elements):
            mask = mesh.field_data['element_indices'].ravel() == index
            np.testing.assert_allclose(means[mask], .001*(cell.u_min+cell.u_max), atol=1e-12)
        coordinates = [(gp.xi, gp.eta) for cell in space.elements for gp in cell.gausspoints]
        indices = [i for i, cell in enumerate(space.elements) for _ in cell.gausspoints]
        samples = SimpleNamespace(field_data={'parametric_coordinates': np.array(coordinates),
                                             'element_indices': np.array(indices)})
        np.testing.assert_allclose(fields.sample(samples, 'strain_11', representation='Element interpolation'),
                                   .002*np.array(coordinates)[:, 0], atol=1e-12)

    def test_incomplete_solution_is_rejected_before_results_are_prepared(self):
        space = self.space()
        with self.assertRaises(ValueError):
            update_cps(space, np.zeros(3*len(space.basis)), dof_per_cp=5, formulation='RM5')
        space.basis[0].dof_values = np.zeros(3)
        mesh, _ = build_surface_data(space, fields=('uz',))
        with self.assertRaisesRegex(ValueError, 'complete'):
            ShellResultFields(space).sample(mesh, 'stress_11')

    def test_surface_export(self):
        import tempfile
        from pathlib import Path
        import pyvista as pv
        from IGAeasy.postprocessing.result_export import export_shell_results
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'shell.vtm'
            export_shell_results(self.space(), filename, fields=('strain_11',))
            blocks = pv.read(filename)
            self.assertEqual(blocks.keys(), list(ShellResultFields.representations))
            for surface in blocks:
                np.testing.assert_allclose(surface['strain_11'], .002, atol=1e-12)
                np.testing.assert_allclose(surface.points[:, 2], 0., atol=1e-12)
                np.testing.assert_allclose(surface['displacement'][:, 0],
                    .002*surface.points[:, 0]+.0005*surface.points[:, 1], atol=1e-12)
                self.assertEqual(surface.field_data['section_layer'][0], 'Middle')

    def test_selectors_preserve_fields_when_deformation_or_color_is_disabled(self):
        space = self.space()
        plotter = plot_solution(space, deformation_scale=3., show=False, off_screen=True)
        try:
            self.assertEqual(len(plotter.slider_widgets), 0)
            surface = plotter.renderer.actors['result_surface']
            self.assertNotIn('stress_11', surface.mapper.dataset.point_data)
            controls = plotter.result_controls
            deformation = plotter.button_widgets[-1-sum(len(group) for group in controls.values())]
            def click(group, name, checked=True):
                button = controls[group][name]
                button.GetRepresentation().SetState(int(checked))
                button.InvokeEvent('StateChangedEvent')
            click('Field', 'stress_11')
            click('Section display', 'Global L2')
            # Returning to a field must reuse both projection and sampled values.
            from unittest.mock import patch
            with patch('IGAeasy.postprocessing.shell_fields.ShellResultFields.sample',
                       side_effect=AssertionError('Cached field was evaluated again')):
                click('Field', 'ux')
                click('Field', 'stress_11')
            click('Field', 'stress_11', False)
            self.assertEqual(controls['Field']['stress_11'].GetRepresentation().GetState(), 1)
            self.assertEqual(sum(w.GetRepresentation().GetState()
                                 for w in controls['Field'].values()), 1)
            expected = 100/(1-.3**2)*(.002-.3*.001)
            np.testing.assert_allclose(surface.mapper.dataset['stress_11'], expected, atol=1e-10)
            self.assertEqual(surface.mapper.GetArrayName(), 'stress_11')
            self.assertEqual(surface.mapper.dataset.active_scalars_name, 'stress_11')
            for widget in (plotter.button_widgets[0], deformation):
                widget.GetRepresentation().SetState(0)
                widget.InvokeEvent('StateChangedEvent')
            click('Field', 'strain_11')
            self.assertFalse(surface.mapper.GetScalarVisibility())
            np.testing.assert_allclose(surface.mapper.dataset['strain_11'], .002, atol=1e-12)
            self.assertEqual(surface.mapper.dataset.active_scalars_name, 'strain_11')
            np.testing.assert_allclose(surface.mapper.dataset.points[:, :2],
                                       surface.mapper.dataset.field_data['parametric_coordinates'], atol=1e-12)
            deformation.GetRepresentation().SetState(1)
            deformation.InvokeEvent('StateChangedEvent')
            np.testing.assert_allclose(surface.mapper.dataset['strain_11'], .002, atol=1e-12)
            self.assertEqual(surface.mapper.dataset.active_scalars_name, 'strain_11')
        finally:
            plotter.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
