"""General curve/trim penalties agree with the established director trace."""
import unittest
import numpy as np
from shapely.geometry import box, LineString, Point
from shapely.geometry.polygon import orient
from IGAeasy.trimming.Brep import BRep
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.constraints.shell_gradient_penalty import WeakGradientShellPenalty
from IGAeasy.constraints.shell_penalty import assemble_director_penalty
from IGAeasy.assembler import MatrixAssembler, VectorAssembler


class GradientPenaltyTests(unittest.TestCase):
    def test_loaded_plate_approaches_strong_clamping(self):
        from IGAeasy.elements import KLshell3p, RMshell5p_hier
        from IGAeasy.material.linear_elastic import LinearElasticMaterial
        from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
        from IGAeasy.constraints.boundary_selection import select_patch_boundary
        from IGAeasy.constraints.linear_constraints import LinearConstraints, solve_linear_constraints
        from IGAeasy.constraints.shell_constraints import constrain_affine_director_boundary
        from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature
        from IGAeasy.loads import SurfaceLoad
        from IGAeasy.solver.direct_solver import solve
        for formulation, module, dof in (('KL', KLshell3p, 3), ('RM5_hier', RMshell5p_hier, 5)):
            geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
            rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
            assign_gp_and_properties(rs, material_model=LinearElasticMaterial(10920., .3, 1.),
                                     properties={'t': .1, 'rho': 1.}, full_quad=True)
            shell = module.KLShell() if formulation == 'KL' else module.RMShell()
            load = SurfaceLoad(dof, [lambda x, y: 0., lambda x, y: 0., lambda x, y: -1.]+[lambda x, y: 0.]*(dof-3))
            for cell in rs.elements:
                shell.element_routine(cell)
                cell.f_el += load.integrate_contribution(cell, dof)
            bulk = MatrixAssembler(dof_per_cp=dof).assemble(rs, 'K_el', use_sparse=True)
            force = VectorAssembler(dof_per_cp=dof).assemble(rs, 'f_el')
            names = ('x', 'y', 'z') if dof == 3 else ('x', 'y', 'z', 'r_a1', 'r_a2')
            constraints = LinearConstraints.for_control_points(names, len(rs.basis))
            edges = ('umin', 'umax', 'vmin', 'vmax')
            ids = set().union(*(select_patch_boundary(rs, edge) for edge in edges))
            constraints.prescribe(ids, ('x', 'y', 'z'))
            for edge in edges:
                for direction in ([1., 0., 0.], [0., 1., 0.]):
                    constraints = constrain_affine_director_boundary(rs, constraints, edge, direction, formulation)
            reference, _ = solve_linear_constraints(bulk, force, constraints)
            boundary = WeakGradientShellPenalty(LineString(box(0., 0., 1., 1.).exterior.coords),
                lambda x, y, z: np.zeros(6), formulation, penalty_u=100., penalty_d=1.)
            for cell in rs.elements:
                cell.K_el, cell.f_el = boundary.integrate_contribution(cell)
            penalty = MatrixAssembler(dof_per_cp=dof).assemble(rs, 'K_el', use_sparse=True)
            norm = SurfaceL2Quadrature(rs)
            errors = []
            for factor in (10., 100., 1000.):
                solution = solve(bulk+factor*penalty, force, scaling='auto')
                errors.append(norm.error(solution.reshape(-1, dof)[:, :3], reference.reshape(-1, dof)[:, :3]))
            self.assertTrue(all(b < a for a, b in zip(errors[:-1], errors[1:])))
            self.assertLess(errors[-1], errors[0]/20)

    def test_curved_rigid_motion_refined_space_and_hole(self):
        angle = .6
        R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                      [-np.sin(angle), 0., np.cos(angle)]])
        omega = R@np.array([.003, -.002, .004])
        for formulation, dof in (('KL', 3), ('RM5_hier', 5)):
            for family in ('THB', 'LR'):
                geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
                rs = (HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points, 2) if family == 'THB'
                      else LRSpline(2, *geo.knot_vectors, geo.control_points))
                for b in rs.basis:
                    cp = np.asarray(b.cp).copy()
                    cp[2] = .2*cp[0]*cp[1]
                    b.cp = R@cp
                if family == 'THB':
                    rs.refine({0: [rs.basis[0].loc_id]})
                else:
                    rs.refine_by_structured_mesh([rs.basis[0]])
                    rs.get_connectivity()
                hole = orient(Point(.5, .5).buffer(.12, quad_segs=8), sign=-1.)
                rs.BReps = [BRep(2, polygon=hole, outerloop=False)]
                trim(rs)

                def field(x, y, z):
                    local = R.T@np.array([x, y, z])
                    n = np.array([-.2*local[1], -.2*local[0], 1.])
                    n = R@(n/np.linalg.norm(n))
                    return np.r_[np.cross(omega, [x, y, z]), np.cross(omega, n)]

                boundary = WeakGradientShellPenalty(LineString(hole.exterior.coords), field,
                    formulation, penalty_u=10., penalty_d=10.)
                exact = np.zeros(dof*len(rs.basis))
                for b in rs.basis:
                    exact[dof*b.id:dof*b.id+3] = np.cross(omega, b.cp)
                for cell in rs.elements:
                    cell.bfnums = np.array(sorted(b.id for b in cell.supported_b_splines), dtype=int)
                    cell.K_el, cell.f_el = boundary.integrate_contribution(cell)
                K = MatrixAssembler(dof_per_cp=dof).assemble(rs, 'K_el', use_sparse=True)
                f = VectorAssembler(dof_per_cp=dof).assemble(rs, 'f_el')
                np.testing.assert_allclose(K@exact, f, atol=1e-11)

    def test_patch_edge_matches_existing_director_penalty(self):
        for formulation, dof in (('KL', 3), ('RM5_hier', 5)):
            for trimmed in (False, True):
                geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
                rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
                for b in rs.basis:
                    b.cp = np.asarray(b.cp)*[2., 3., 1.]
                if trimmed:
                    rs.BReps = [BRep(1, polygon=box(-.1, -.1, .3, .5), outerloop=False)]
                trim(rs)
                boundary = WeakGradientShellPenalty(LineString([(0., 1.), (0., 0.)]),
                    lambda x, y, z: [0., 0., 0., .1, 0., 0.], formulation,
                    penalty_d=10., director_directions=[[1., 0., 0.]])
                for cell in rs.elements:
                    cell.bfnums = np.array(sorted(b.id for b in cell.supported_b_splines), dtype=int)
                    cell.K_el, cell.f_el = boundary.integrate_contribution(cell)
                actual_K = MatrixAssembler(dof_per_cp=dof).assemble(rs, 'K_el', use_sparse=True)
                actual_f = VectorAssembler(dof_per_cp=dof).assemble(rs, 'f_el')
                components = ('x', 'y', 'z') if dof == 3 else ('x', 'y', 'z', 'r_a1', 'r_a2')
                expected_K, expected_f = assemble_director_penalty(rs, components, 'umin',
                    formulation, 10., direction=[1., 0., 0.], value=.1)
                np.testing.assert_allclose(actual_K.toarray(), expected_K.toarray(), atol=1e-11)
                np.testing.assert_allclose(actual_f, expected_f, atol=1e-11)

    def test_constant_translation_on_inner_trim_boundary(self):
        geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
        rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
        hole = orient(box(.3, .3, .7, .7), sign=-1.)
        rs.BReps = [BRep(2, polygon=hole, outerloop=False)]
        trim(rs)
        field = np.tile([.01, -.02, .03], len(rs.basis))
        boundary = WeakGradientShellPenalty(LineString(hole.exterior.coords),
            lambda x, y, z: [.01, -.02, .03, 0., 0., 0.], 'KL', penalty_u=10., penalty_d=10.)
        for cell in rs.elements:
            cell.bfnums = np.array(sorted(b.id for b in cell.supported_b_splines), dtype=int)
            cell.K_el, cell.f_el = boundary.integrate_contribution(cell)
        K = MatrixAssembler(dof_per_cp=3).assemble(rs, 'K_el', use_sparse=True)
        f = VectorAssembler(dof_per_cp=3).assemble(rs, 'f_el')
        np.testing.assert_allclose(K@field, f, atol=1e-11)
        self.assertAlmostEqual(field@K@field, 10*hole.length*(.01**2+.02**2+.03**2), places=11)


if __name__ == '__main__':
    unittest.main(verbosity=2)
