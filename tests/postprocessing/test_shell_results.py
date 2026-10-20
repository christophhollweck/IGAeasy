"""Analytical membrane, bending and transverse shear resultants."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from shapely.geometry import Polygon, box
import numpy as np
from IGAeasy.trimming.Brep import BRep
from IGAeasy.elements import RMshell5p, RMshell5p_v2, RMshell5p_hier, KLshell3p, RMshell6p
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.constraints import helpers, dirichlet
from IGAeasy.loads import SurfaceLoad
from IGAeasy.solver import direct_solver
from IGAeasy.numerical_integration.costum_rule import gauss_lobatto
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.postprocessing import postprocessing
from IGAeasy.geometry import get_geometry
from tests.support import plate_rm5_reference as ex1_plate_RM5
from tests.support import plate_kl_reference as ex1_plate_KL
from IGAeasy.postprocessing.result_export import export_integration_point_results


class ShellResultTests(unittest.TestCase):
    def test_solved_curved_roof_results_for_all_formulations(self):
        geo = get_geometry.scordelis_lo_roof(1, 1, 2, full=True)
        for module, family in ((KLshell3p, 'KL'), (RMshell6p, 'RM6dof'),
                               (RMshell5p, 'RM5dof'), (RMshell5p_v2, 'RM5dof'),
                               (RMshell5p_hier, 'RM5dof')):
            with self.subTest(element=module.__name__):
                rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points.copy())
                shell = module.KLShell() if family == 'KL' else module.RMShell()
                dof = shell.dof_per_cp
                assign_gp_and_properties(rs, material_model=LinearElasticMaterial(4.32e8, 0., 1.),
                                         properties={'t': .25, 'E': 4.32e8, 'rho': 1.}, full_quad=True)
                load = SurfaceLoad(dof, [lambda u, v: 0., lambda u, v: 0., lambda u, v: -90.]
                                   +[lambda u, v: 0.]*(dof-3))
                fixed = set()
                for edge in ('ymin', 'ymax'):
                    ids = helpers.find_control_point_ids_on_edges(rs, **{edge: True})
                    fixed.update(dof*i+j for i in ids for j in (0, 2))
                corners = set(helpers.find_control_point_ids_on_edges(rs, xmin=True)) & set(
                    helpers.find_control_point_ids_on_edges(rs, ymin=True))
                fixed.add(dof*min(corners)+1)
                for cell in rs.elements:
                    if family == 'RM6dof':
                        shell.element_routine(cell, gauss_lobatto(3))
                    else:
                        shell.element_routine(cell)
                    cell.f_el += load.integrate_contribution(cell, dof)
                K = MatrixAssembler(dof_per_cp=dof).assemble(rs, 'K_el', use_sparse=True)
                f = VectorAssembler(dof_per_cp=dof).assemble(rs, 'f_el')
                reduced, rhs, free = dirichlet.block_partitioning_dbc(K=K, f=f, prescribed_DOFs=fixed)
                values = np.zeros(len(f))
                values[free] = direct_solver.solve(reduced, rhs)
                residual = np.linalg.norm(reduced@values[free]-rhs)/np.linalg.norm(rhs)
                self.assertLess(residual, 1e-8)
                postprocessing.update_cps(rs, values, dof_per_cp=dof, dimensions=3)
                update = getattr(postprocessing, 'update_stress_strain_'+family)
                interpolate = getattr(postprocessing, 'interpolate_stress_'+family)
                evaluate = getattr(postprocessing, 'eval_stress_'+family)
                update(rs)
                first = np.array([gp.sigma_vM for cell in rs.elements for gp in cell.gausspoints])
                interpolate(rs)
                update(rs)
                second = np.array([gp.sigma_vM for cell in rs.elements for gp in cell.gausspoints])
                np.testing.assert_array_equal(first, second)
                self.assertTrue(np.all(np.isfinite(second)))
                self.assertGreater(second.max(), 0.)
                interpolate(rs)
                for quantity in ('N', 'M', 'vM')+(() if family == 'KL' else ('Q',)):
                    self.assertTrue(np.all(np.isfinite(evaluate(rs, .37, .43, quantity))))

    def test_solved_plate_with_hole_postprocessing(self):
        samples = []
        for name, solver, options in (
                ('RM5', ex1_plate_RM5, {}),
                ('RM5_v2', ex1_plate_RM5, {'shell_module': RMshell5p_v2}),
                ('KL', ex1_plate_KL, {})):
            geo = get_geometry.surface(4, 4, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
            rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
            rs.BReps = [BRep(1, polygon=box(.35, .35, .65, .65), outerloop=False)]
            solver.solve(rs, use_sparse=True, compute_stress=True, verbose=False,
                         boundary='simply_supported', full_quad=True, **options)
            interpolate = (postprocessing.interpolate_stress_KL if name == 'KL'
                           else postprocessing.interpolate_stress_RM5dof)
            evaluate = (postprocessing.eval_stress_KL if name == 'KL'
                        else postprocessing.eval_stress_RM5dof)
            interpolate(rs)
            # This point belongs to a trimmed cell, but lies in the material.
            result = evaluate(rs, .3, .4, 'M')
            self.assertTrue(np.all(np.isfinite(result)))
            self.assertGreater(np.linalg.norm(result), 0.)
            with self.assertRaises(ValueError):
                evaluate(rs, .5, .5, 'M')
            if name != 'KL':
                samples.append(result)
        np.testing.assert_allclose(samples[0], samples[1], rtol=1e-10, atol=1e-12)

    def test_trimmed_interpolation_excludes_holes_and_accepts_boundaries(self):
        material = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)],
                           holes=[[(.3, .3), (.7, .3), (.7, .7), (.3, .7)]])
        gp = SimpleNamespace(xi=.1, eta=.1, N=np.array([1., 2., 3.]),
                             M=np.zeros(3), sigma_vM=6.)
        cell = SimpleNamespace(gausspoints=[gp], active=True, void=False,
                               trimmed=True, material_polygons=[material],
                               u_min=0., u_max=1., v_min=0., v_max=1.)
        space = SimpleNamespace(elements=[cell])
        postprocessing.interpolate_stress_KL(space)
        for x, y in ((.1, .5), (.3, .5), (0., .5)):
            np.testing.assert_allclose(postprocessing.eval_stress_KL(space, x, y, 'N'), [[1., 2., 3.]])
        for x, y in ((.5, .5), (1.1, .5)):
            with self.assertRaisesRegex(ValueError, 'outside the active material'):
                postprocessing.eval_stress_KL(space, x, y, 'N')
        cell.material_polygons = [box(0, 0, .2, 1), box(.8, 0, 1, 1)]
        with self.assertRaises(ValueError):
            postprocessing.eval_stress_KL(space, .5, .1)
        np.testing.assert_allclose(postprocessing.eval_stress_KL(space, .9, .5), [6.])

    def test_interpolation_handles_one_point_and_skips_inactive_cells(self):
        gp = SimpleNamespace(xi=.5, eta=.5, N=np.array([1., 2., 3.]),
                             M=np.zeros(3), Q=np.array([4., 5.]), sigma_vM=6.)
        cell = SimpleNamespace(gausspoints=[gp], active=True, void=False,
                               trimmed=False, u_min=0., u_max=1., v_min=0., v_max=1.)
        inactive = SimpleNamespace(active=False, void=False, N_interp='stale')
        space = SimpleNamespace(elements=[inactive, cell])
        postprocessing.interpolate_stress_RM5dof(space)
        self.assertFalse(hasattr(inactive, 'N_interp'))
        np.testing.assert_allclose(postprocessing.eval_stress_RM5dof(space, .2, .8, 'Q'), [[4., 5.]])
        postprocessing.interpolate_stress_RM5dof(space, 'N')
        np.testing.assert_allclose(postprocessing.eval_stress_RM5dof(space, .2, .8, 'N'), [[1., 2., 3.]])
        with self.assertRaises(ValueError):
            postprocessing.interpolate_stress_KL(space, 'Q')
        del gp.Q
        with self.assertRaisesRegex(ValueError, 'Missing Q'):
            postprocessing.interpolate_stress_RM5dof(space, 'Q')
        self.assertFalse(hasattr(cell, 'Q_interp'))

    def test_KL_membrane_and_bending(self):
        cps = [[u, v, 0.] for v in (0., .5, 1.) for u in (0., .5, 1.)]
        knots = [0., 0., 0., 1., 1., 1.]
        rs = HierarchicalSpace(knots, knots, 2, cps)
        assign_gp_and_properties(rs, material_model=LinearElasticMaterial(100., 0., 1.),
                                 properties={'t': .2, 'E': 100., 'rho': 1.}, full_quad=True)
        for b in rs.basis:
            i, j = int(round(2*b.cp[0])), int(round(2*b.cp[1]))
            b.dof_values = [.01*b.cp[0], .02*b.cp[1], -.015*i*(i-1)/2-.02*j*(j-1)/2]
        for cell in rs.elements:
            KLshell3p.KLShell().element_routine(cell)
        postprocessing.update_stress_strain_KL(rs)
        postprocessing.interpolate_stress_KL(rs)
        postprocessing.update_stress_strain_KL(rs)
        self.assertFalse(hasattr(rs.elements[0], 'N_interp'))
        for gp in rs.elements[0].gausspoints:
            np.testing.assert_allclose(gp.N, [.2, .4, 0.], atol=1e-13)
            np.testing.assert_allclose(gp.M, 100*.2**3/12*np.array([.03, .04, 0.]), atol=1e-13)
            self.assertIsNone(gp.Q)
        with tempfile.TemporaryDirectory() as directory:
            import pyvista as pv
            path = Path(directory)/'results.vtp'
            cloud = export_integration_point_results(rs, path)
            loaded = pv.read(path)
            self.assertEqual(loaded.n_points, sum(len(el.gausspoints) for el in rs.elements))
            np.testing.assert_allclose(loaded['N'], cloud['N'])
            np.testing.assert_allclose(loaded['M'], cloud['M'])
            self.assertIn('local_e1', loaded.point_data)
            self.assertIn('stress_von_mises', loaded.point_data)
            with self.assertRaisesRegex(ValueError, 'Missing Q'):
                export_integration_point_results(rs, path, ('Q',))

    def test_RM6_integrates_physical_thickness_and_components(self):
        t = .4
        zeta, weights = gauss_lobatto(3)
        C = LinearElasticMaterial(100., 0., 1.).get_C_voigt_cart_3D()
        constant = np.array([2., 3., 0., .7, .8, .9])
        slope = np.array([4., 5., 0., 1.2, 0., 0.])
        matrices = []
        for point in zeta:
            stress = constant + slope*point*t/2
            strain = np.zeros(6)
            strain[[0, 1, 3, 4, 5]] = stress[[0, 1, 3, 4, 5]]/np.diag(C)[[0, 1, 3, 4, 5]]
            matrices.append(strain[:, None])
        gp = SimpleNamespace(t=t, zeta=zeta, weight_outerplane=weights,
                             Bmatrices=matrices, rotationmatrices=[np.eye(6)]*3,
                             material_model=LinearElasticMaterial(100., 0., 1.))
        cell = SimpleNamespace(supported_b_splines=[SimpleNamespace(id=0, dof_values=[1.])],
                               gausspoints=[gp], active=True, void=False)
        rs = SimpleNamespace(elements=[cell])
        for _ in range(2):
            postprocessing.update_stress_strain_RM6dof(rs)
        np.testing.assert_allclose(gp.N, t*constant[[0, 1, 3]], atol=1e-14)
        np.testing.assert_allclose(gp.M, t**3/12*slope[[0, 1, 3]], atol=1e-14)
        np.testing.assert_allclose(gp.Q, t*constant[[5, 4]], atol=1e-14)
        self.assertEqual(len(gp.stress_local), 3)

    def test_physical_resultants_on_affine_skew_surface(self):
        E, t, kappa = 100., .2, 5/6
        for module in (RMshell5p, RMshell5p_v2, RMshell5p_hier):
            for skew in (0., .4):
                cov = np.array([[2., skew], [0., 3.]])
                director_basis = np.linalg.inv(cov).T if module is RMshell5p else cov
                cps = [[*(cov@np.array([u, v])), 0.]
                       for v in (0., .5, 1.) for u in (0., .5, 1.)]
                knots = [0., 0., 0., 1., 1., 1.]
                rs = HierarchicalSpace(knots, knots, 2, cps)
                assign_gp_and_properties(rs, material_model=LinearElasticMaterial(E, 0., 1., kappa),
                                         properties={'t': t, 'E': E, 'rho': 1.}, full_quad=True)
                for b in rs.basis:
                    x, y = b.cp[:2]
                    rotation = np.linalg.solve(director_basis, [.03*x, .04*y])
                    b.dof_values = np.r_[.01*x, .02*y, 0., rotation]
                for cell in rs.elements:
                    module.RMShell().element_routine(cell)
                    module.RMShell().element_routine(cell)
                    self.assertTrue(all(len(gp.Bmatrices) == 1 for gp in cell.gausspoints))
                postprocessing.update_stress_strain_RM5dof(rs)
                postprocessing.update_stress_strain_RM5dof(rs)
                for cell in rs.elements:
                    for gp in cell.gausspoints:
                        position = sum(b.cp*b(gp.xi, gp.eta) for b in cell.supported_b_splines)
                        np.testing.assert_allclose(gp.N, E*t*np.array([.01, .02, 0.]), atol=1e-13)
                        np.testing.assert_allclose(gp.M, E*t**3/12*np.array([.03, .04, 0.]), atol=1e-13)
                        expected_q = kappa*E/2*t*np.array([.03*position[0], .04*position[1]])
                        np.testing.assert_allclose(gp.Q, expected_q, atol=1e-13)
                        sigma = gp.N/t + 6*gp.M/t**2
                        vm = np.sqrt(sigma[0]**2+sigma[1]**2-sigma[0]*sigma[1]
                                     +3*(sigma[2]**2+np.dot(expected_q/t, expected_q/t)))
                        self.assertAlmostEqual(gp.sigma_vM, vm, places=12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
