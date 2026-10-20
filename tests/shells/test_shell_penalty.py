"""Physical boundary measure and material clipping for director penalties."""
import unittest
import numpy as np
from shapely.geometry import box
from IGAeasy.trimming.Brep import BRep
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.constraints.shell_penalty import assemble_director_penalty, assemble_KL_slope_nitsche
from IGAeasy.trimming.trim_elements import trim


class ShellPenaltyTests(unittest.TestCase):
    def test_KL_nitsche_quadratic_patch_with_nonzero_slope(self):
        from IGAeasy.geometry import get_geometry
        from IGAeasy.material.linear_elastic import LinearElasticMaterial
        from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
        from IGAeasy.elements import KLshell3p
        from IGAeasy.assembler import MatrixAssembler
        from IGAeasy.constraints.boundary_selection import select_patch_boundary
        from IGAeasy.constraints.dirichlet import block_partitioning_dbc
        from IGAeasy.solver.direct_solver import solve
        from scipy.interpolate import BSpline
        geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
        rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
        assign_gp_and_properties(rs, material_model=LinearElasticMaterial(10920., .3, 1.),
                                 properties={'t': .1}, full_quad=True)
        for cell in rs.elements:
            KLshell3p.KLShell().element_routine(cell)
        K = MatrixAssembler(dof_per_cp=3).assemble(rs, 'K_el', use_sparse=True)
        f = np.zeros(K.shape[0])
        # Independent polynomial representation of w=x^2 in the tensor spline basis.
        knots = np.asarray(geo.knot_vectors[0])
        count = len(knots)-3
        samples = np.linspace(0, 1, count)
        coefficients = np.linalg.solve(BSpline.design_matrix(samples, knots, 2).toarray(), samples**2)
        exact = np.zeros(len(f))
        for basis in rs.basis:
            i = next(i for i in range(count) if np.array_equal(basis.knots_u, knots[i:i+4]))
            exact[3*basis.id+2] = coefficients[i]
        ids = set()
        for edge in ('umin', 'umax', 'vmin', 'vmax'):
            contribution, rhs = assemble_KL_slope_nitsche(rs, edge, 100.,
                value=0. if edge.startswith('v') or edge == 'umin' else -2.)
            K += contribution
            f += rhs
            ids.update(select_patch_boundary(rs, edge))
        fixed = sorted(3*i+j for i in ids for j in range(3))
        reduced, rhs, free = block_partitioning_dbc(K=K, f=f, prescribed_DOFs=fixed, u_c=exact[fixed])
        np.testing.assert_allclose(reduced.toarray(), reduced.T.toarray(), atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(reduced.toarray()).min(), 0.)
        np.testing.assert_allclose(reduced@exact[free], rhs, atol=1e-11)
        np.testing.assert_allclose(solve(reduced, rhs), exact[free], atol=1e-11)

    def test_penalty_plate_approaches_strong_director_constraints(self):
        from IGAeasy.geometry import get_geometry
        from IGAeasy.elements import KLshell3p
        from IGAeasy.material.linear_elastic import LinearElasticMaterial
        from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
        from IGAeasy.assembler import MatrixAssembler, VectorAssembler
        from IGAeasy.loads import SurfaceLoad
        from IGAeasy.constraints.boundary_selection import select_patch_boundary
        from IGAeasy.constraints.linear_constraints import LinearConstraints, solve_linear_constraints
        from IGAeasy.constraints.shell_constraints import constrain_affine_director_boundary
        from IGAeasy.constraints.dirichlet import block_partitioning_dbc
        geo = get_geometry.surface(4, 4, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
        rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
        assign_gp_and_properties(rs, material_model=LinearElasticMaterial(10920., .3, 1.),
                                 properties={'t': .1, 'rho': 1.}, full_quad=True)
        load = SurfaceLoad(3, [lambda u, v: 0., lambda u, v: 0., lambda u, v: -1.])
        for cell in rs.elements:
            KLshell3p.KLShell().element_routine(cell)
            cell.f_el += load.integrate_contribution(cell, 3)
        K = MatrixAssembler(dof_per_cp=3).assemble(rs, 'K_el', use_sparse=True)
        f = VectorAssembler(dof_per_cp=3).assemble(rs, 'f_el')
        edges = ('umin', 'umax', 'vmin', 'vmax')
        ids = set().union(*(select_patch_boundary(rs, edge) for edge in edges))
        strong = LinearConstraints.for_control_points(('x', 'y', 'z'), len(rs.basis))
        strong.prescribe(ids, ('x', 'y', 'z'))
        fixed = sorted(3*i+j for i in ids for j in range(3))
        boundary_K = K*0
        for edge in edges:
            strong = constrain_affine_director_boundary(rs, strong, edge, [1., 0., 0.], 'KL')
            strong = constrain_affine_director_boundary(rs, strong, edge, [0., 1., 0.], 'KL')
            contribution, _ = assemble_director_penalty(rs, ('x', 'y', 'z'), edge, 'KL', 1.)
            boundary_K += contribution
        reference, _ = solve_linear_constraints(K, f, strong)
        differences, traces = [], []
        for penalty in (10., 100., 1000.):
            reduced, rhs, free = block_partitioning_dbc(K=K+penalty*boundary_K, f=f,
                                                       prescribed_DOFs=fixed)
            from IGAeasy.solver.direct_solver import solve
            values = np.zeros(len(f))
            values[free] = solve(reduced, rhs)
            differences.append(np.linalg.norm(values-reference))
            traces.append(np.sqrt(max(values@boundary_K@values, 0.)))
        self.assertTrue(all(b < a for a, b in zip(differences[:-1], differences[1:])))
        self.assertTrue(all(b < a for a, b in zip(traces[:-1], traces[1:])))
        self.assertLess(differences[-1], differences[0]/50)

    def space(self):
        knots = [0., 0., 0., 1., 1., 1.]
        cps = [[2*u, 3*v, 0.] for v in (0., .5, 1.) for u in (0., .5, 1.)]
        return HierarchicalSpace(knots, knots, 2, cps)

    def test_physical_edge_measure_and_nonzero_value(self):
        rs = self.space()
        values = np.concatenate([[0., 0., .1*b.cp[0]] for b in rs.basis])
        K, f = assemble_director_penalty(rs, ('x', 'y', 'z'), 'umin', 'KL', 10.,
                                         direction=[1., 0., 0.], value=-.1)
        self.assertAlmostEqual(values@K@values, 10*.1**2*3, places=12)
        np.testing.assert_allclose(K@values, f, atol=1e-12)
        np.testing.assert_allclose(K.toarray(), K.T.toarray(), atol=1e-13)

    def test_trimmed_edge_and_rotated_local_direction(self):
        for angle in (0., .7):
            rs = self.space()
            rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                                 [-np.sin(angle), 0., np.cos(angle)]])
            values = []
            for b in rs.basis:
                values.extend(rotation@np.array([0., 0., .1*b.cp[0]]))
                b.cp = rotation@b.cp
            rs.BReps = [BRep(1, polygon=box(-.1, -.1, .2, .5), outerloop=False)]
            trim(rs)
            K, _ = assemble_director_penalty(rs, ('x', 'y', 'z'), 'umin', 'KL', 10.)
            # Half of the length-3 edge remains in the material.
            self.assertAlmostEqual(np.array(values)@K@values, 10*.1**2*1.5, places=11)


if __name__ == '__main__':
    unittest.main(verbosity=2)
