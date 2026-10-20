"""Batched shell kernels reproduce scalar geometry, strain and load assembly."""
import unittest
from unittest.mock import patch
import numpy as np
from shapely.geometry import box, LineString
from shapely.affinity import rotate
from shapely.geometry.polygon import orient
from IGAeasy.trimming.Brep import BRep
from IGAeasy.constraints import shell_weak
from IGAeasy.constraints.line_helpers import get_integration_points
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.splines.batch_evaluation import evaluate_derivatives_batch
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.differential_geometry.derivatives import get_full_geometric_data, get_local_basis_and_variations
from IGAeasy.elements import KLshell3p, RMshell5p_hier, RMshell5p, RMshell5p_v2, RMshell6p
from IGAeasy.loads import SurfaceLoad


def space(family):
    ku, kv = [0.]*4+[.2, .7]+[1.]*4, [0.]*4+[.4, .9]+[1.]*4
    gu = [np.mean(ku[i+1:i+4]) for i in range(6)]
    gv = [np.mean(kv[i+1:i+4]) for i in range(6)]
    cps = [[u, v, .2*u*v] for v in gv for u in gu]
    rs = HierarchicalSpace(ku, kv, 3, cps, 2) if family == 'THB' else LRSpline(3, ku, kv, cps)
    if family == 'THB':
        rs.refine({0: [rs.basis[0].loc_id]})
    else:
        rs.refine_by_structured_mesh([rs.basis[0]])
        rs.get_connectivity()
    return rs


def scalar_batch(basis, points, cell, max_order=2):
    return {(a, total-a): np.array([[b(u, v, r1=a, r2=total-a, cell=cell) for b in basis] for u, v in points])
            for total in range(max_order+1) for a in range(total+1)}


class ShellBatchTests(unittest.TestCase):
    def test_rm_boundary_matrices_loads_and_stabilization_match_scalar_path(self):
        def scalar_full(cell, u, v, geometry_derivatives=None):
            return get_full_geometric_data(cell, u, v)
        def scalar_variations(cell, u, v, geometry_derivatives=None):
            return get_local_basis_and_variations(cell, u, v)
        rule = np.polynomial.legendre.leggauss(3)
        outer = box(.03, .02, .91, .87)
        hole = orient(rotate(box(.32, .33, .51, .55), 23, origin='centroid'), sign=-1.)
        for family in ('THB', 'LR'):
            rs = space(family)
            rs.BReps = [BRep(1, polygon=outer, outerloop=True),
                        BRep(2, polygon=hole, outerloop=False)]
            trim(rs)
            assign_gp_and_properties(rs, material_model=LinearElasticMaterial(10920., .3, 1.),
                                     properties={'t': .1, 'rho': 1.}, full_quad=True)
            for polygon in (outer, hole):
                line = LineString(polygon.exterior.coords)
                candidates = [cell for cell in rs.elements if get_integration_points(line, cell)]
                cell = max(candidates, key=lambda c: len(c.supported_b_splines))
                for formulation, module in (('RM6', RMshell6p), ('RM5', RMshell5p),
                                             ('RM5_v2', RMshell5p_v2)):
                    shell = module.RMShell()
                    if formulation == 'RM6':
                        shell.element_routine(cell, rule)
                    else:
                        shell.element_routine(cell)
                    for method in ('penalty', 'nitsche'):
                        with self.subTest(family=family, hole=polygon is hole,
                                          formulation=formulation, method=method):
                            # Translation data is attainable in both director conventions.
                            field = lambda x, y, z: np.array([x*x*.001, y*.002, z*.001, 0., 0., 0.])
                            boundary = (shell_weak.WeakRM6Boundary(line, field, rule, method=method)
                                        if formulation == 'RM6' else
                                        shell_weak.WeakRM5Boundary(line, field, formulation=formulation,
                                                                  method=method))
                            with patch.object(shell_weak, 'evaluate_derivatives_batch', scalar_batch), patch.object(
                                    shell_weak, 'get_full_geometric_data', scalar_full), patch.object(
                                    shell_weak, 'get_local_basis_and_variations', scalar_variations):
                                expected = boundary.integrate_contribution(cell)
                            actual = boundary.integrate_contribution(cell)
                            self.assertTrue(np.isfinite(actual[2]))
                            self.assertGreater(actual[2], 0.)
                            for result, reference in zip(actual, expected):
                                np.testing.assert_allclose(result, reference, rtol=2e-7, atol=1e-8)

    def test_rm6_stiffness_mass_and_thickness_strains_match_scalar_path(self):
        def scalar_geometry(cell, u, v, geometry_derivatives=None):
            return get_local_basis_and_variations(cell, u, v)
        thickness_rule = np.polynomial.legendre.leggauss(3)
        shell = RMshell6p.RMShell()
        for family in ('THB', 'LR'):
            with self.subTest(family=family):
                rs = space(family)
                assign_gp_and_properties(rs, material_model=LinearElasticMaterial(10920., .3, 1.),
                                         properties={'t': .1, 'rho': 1.}, full_quad=True)
                cell = max(rs.elements, key=lambda cell: len(cell.supported_b_splines))
                cell.gausspoints = cell.gausspoints[:3]
                with patch.object(RMshell6p, 'evaluate_derivatives_batch', scalar_batch), patch.object(
                        RMshell6p.derivatives, 'get_local_basis_and_variations', scalar_geometry):
                    shell.element_routine(cell, thickness_rule)
                reference_K, reference_M = cell.K_el.copy(), cell.M_el.copy()
                reference_B = [[B.copy() for B in gp.Bmatrices] for gp in cell.gausspoints]
                shell.element_routine(cell, thickness_rule)
                np.testing.assert_allclose(cell.K_el, reference_K, rtol=1e-11, atol=1e-9)
                np.testing.assert_allclose(cell.M_el, reference_M, rtol=1e-11, atol=1e-12)
                for gp, old_B in zip(cell.gausspoints, reference_B):
                    for actual, expected in zip(gp.Bmatrices, old_B):
                        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)
                load = SurfaceLoad(6, [lambda u, v, j=j: (j+1)*(u*u+v) for j in range(6)])
                cached = load.integrate_contribution(cell, 6)
                for gp in cell.gausspoints:
                    gp.N_evals = None
                np.testing.assert_allclose(cached, load.integrate_contribution(cell, 6),
                                           rtol=1e-12, atol=1e-12)

    def test_third_derivatives_nonuniform_cubic_and_cell_traces(self):
        for family in ('THB', 'LR'):
            for cell in space(family).elements:
                basis = list(cell.supported_b_splines)
                points = [(u, v) for u in np.linspace(cell.u_min, cell.u_max, 5)
                          for v in np.linspace(cell.v_min, cell.v_max, 5)]
                expected = scalar_batch(basis, points, cell, 3)
                actual = evaluate_derivatives_batch(basis, points, cell, 3)
                for order in actual:
                    np.testing.assert_allclose(actual[order], expected[order], rtol=1e-11, atol=1e-8)

    def test_bulk_matrices_and_surface_load_match_scalar_path(self):
        def scalar_geometry(cell, u, v, geometry_derivatives=None):
            return get_full_geometric_data(cell, u, v)
        for family in ('THB', 'LR'):
            rs = space(family)
            assign_gp_and_properties(rs, material_model=LinearElasticMaterial(10920., .3, 1.), properties={'t': .1}, full_quad=True)
            cell = max(rs.elements, key=lambda cell: len(cell.supported_b_splines))
            cell.gausspoints = cell.gausspoints[:3]
            for module, shell, dof in ((KLshell3p, KLshell3p.KLShell(), 3),
                                       (RMshell5p_hier, RMshell5p_hier.RMShell(), 5),
                                       (RMshell5p, RMshell5p.RMShell(), 5),
                                       (RMshell5p_v2, RMshell5p_v2.RMShell(), 5)):
                with self.subTest(family=family, formulation=module.__name__):
                    integration = RMshell5p if module is RMshell5p_v2 else module
                    with patch.object(integration, 'evaluate_derivatives_batch', scalar_batch), patch.object(
                            integration.derivatives, 'get_full_geometric_data', scalar_geometry):
                        shell.element_routine(cell)
                    reference_K = cell.K_el.copy()
                    reference_B = [[B.copy() for B in gp.Bmatrices[0]] for gp in cell.gausspoints]
                    shell.element_routine(cell)
                    np.testing.assert_allclose(cell.K_el, reference_K, rtol=1e-11, atol=1e-9)
                    for gp, old_B in zip(cell.gausspoints, reference_B):
                        for actual, expected in zip(gp.Bmatrices[0], old_B):
                            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)
                    load = SurfaceLoad(dof, [lambda u, v, j=j: (j+1)*(u*u+v) for j in range(dof)])
                    cached = load.integrate_contribution(cell, dof)
                    vector_load = SurfaceLoad(dof, lambda u, v: (np.arange(dof)+1)*(u*u+v))
                    np.testing.assert_allclose(vector_load.integrate_contribution(cell, dof),
                                               cached, rtol=1e-12, atol=1e-12)
                    values = [gp.N_evals for gp in cell.gausspoints]
                    for gp in cell.gausspoints:
                        gp.N_evals = None
                    uncached = load.integrate_contribution(cell, dof)
                    np.testing.assert_allclose(cached, uncached, rtol=1e-12, atol=1e-12)
                    for gp, value in zip(cell.gausspoints, values):
                        gp.N_evals = value
                    original_ids = cell.bfnums.copy()
                    cell.bfnums = cell.bfnums[::-1]
                    np.testing.assert_allclose(load.integrate_contribution(cell, dof), uncached, rtol=1e-12, atol=1e-12)
                    cell.bfnums = original_ids


if __name__ == '__main__':
    unittest.main(verbosity=2)
