"""Physical RM5/RM5_v2 traces, nonzero strain and bending, trimmed affine patches."""
import unittest
import numpy as np
from scipy.interpolate import BSpline
from shapely.geometry import box, LineString
from IGAeasy.trimming.Brep import BRep
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.elements import RMshell5p, RMshell5p_v2
from IGAeasy.constraints.shell_weak import WeakRM5Boundary
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.solver.direct_solver import solve


class WeakRM5Tests(unittest.TestCase):
    def patch(self, formulation, angle, trimmed, bending=False, discretization='THB', refined=False, nonuniform=False):
        geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
        knots = ([0., 0., 0., .3, 1., 1., 1.], [0., 0., 0., .65, 1., 1., 1.]) if nonuniform else geo.knot_vectors
        rs = (HierarchicalSpace(*knots, 2, geo.control_points, 2) if discretization == 'THB'
              else LRSpline(2, *knots, geo.control_points))
        R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                      [-np.sin(angle), 0., np.cos(angle)]])
        for b in rs.basis:
            if nonuniform:
                b.cp = np.array([np.mean(b.knots_u[1:3]), np.mean(b.knots_v[1:3]), 0.])
            b.cp = R@(np.asarray(b.cp)*[2., 3., 1.])
        if refined:
            if discretization == 'THB':
                rs.refine({0: [rs.basis[0].loc_id]})
            else:
                rs.refine_by_structured_mesh([rs.basis[0]])
                rs.get_connectivity()
        domain = box(.13, .07, .91, .87) if trimmed else box(0., 0., 1., 1.)
        if trimmed:
            rs.BReps = [BRep(1, polygon=domain, outerloop=True)]
            trim(rs)
        assign_gp_and_properties(rs, material_model=LinearElasticMaterial(100., .3, 1.),
                                 properties={'t': .1, 'rho': 1.}, full_quad=True)

        def prescribed(x, y, z):
            local = R.T@np.array([x, y, z])
            if bending:
                u, d = [0., 0., .01*local[0]**2], [-.02*local[0], 0., 0.]
            else:
                u, d = [.01*local[0]+.002*local[1], -.003*local[0]+.02*local[1], 0.], [0., 0., 0.]
            return np.r_[R@u, R@d]

        exact = np.zeros(5*len(rs.basis))
        knots = np.asarray(geo.knot_vectors[0])
        count = len(knots)-3
        samples = np.linspace(0, 1, count)
        polynomial = np.linalg.solve(BSpline.design_matrix(samples, knots, 2).toarray(), .04*samples**2)
        for b in rs.basis:
            field = prescribed(*b.cp)
            exact[5*b.id:5*b.id+3] = field[:3]
            if bending:
                index = next(i for i in range(count) if np.array_equal(b.knots_u, knots[i:i+4]))
                exact[5*b.id:5*b.id+3] = R@np.array([0., 0., polynomial[index]])
                physical_d = (R.T@field[3:])[0]
                exact[5*b.id+3] = (2. if formulation == 'RM5' else .5)*physical_d
        boundary = WeakRM5Boundary(LineString(domain.exterior.coords), prescribed, formulation=formulation)
        shell = (RMshell5p if formulation == 'RM5' else RMshell5p_v2).RMShell()
        for cell in rs.elements:
            if cell.void or not getattr(cell, 'active', True):
                continue
            shell.element_routine(cell)
            addition, force, _ = boundary.integrate_contribution(cell)
            cell.K_el += addition
            cell.f_el += force
        K = MatrixAssembler(dof_per_cp=5).assemble(rs, 'K_el', use_sparse=True)
        f = VectorAssembler(dof_per_cp=5).assemble(rs, 'f_el')
        self.assertLess(np.linalg.norm(K@exact-f)/np.linalg.norm(f), 1e-10)
        np.testing.assert_allclose(K.toarray(), K.T.toarray(), atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(K.toarray()).min(), 0.)
        np.testing.assert_allclose(solve(K, f, scaling='auto'), exact, atol=1e-9)

    def test_nonzero_strain(self):
        for formulation in ('RM5', 'RM5_v2'):
            for angle in (0., .6):
                for trimmed in (False, True):
                    with self.subTest(formulation=formulation, angle=angle, trimmed=trimmed):
                        self.patch(formulation, angle, trimmed)

    def test_nonzero_bending(self):
        for formulation in ('RM5', 'RM5_v2'):
            for angle in (0., .6):
                for trimmed in (False, True):
                    with self.subTest(formulation=formulation, angle=angle, trimmed=trimmed):
                        self.patch(formulation, angle, trimmed, bending=True)

    def test_refined_deformation_patch(self):
        for formulation in ('RM5', 'RM5_v2'):
            for discretization in ('THB', 'LR'):
                for nonuniform in (False, True):
                    with self.subTest(formulation=formulation, discretization=discretization, nonuniform=nonuniform):
                        self.patch(formulation, .6, True, discretization=discretization,
                                   refined=True, nonuniform=nonuniform)

    def test_curved_boundary_energy_is_objective(self):
        for formulation in ('RM5', 'RM5_v2'):
            energies, gammas, fixed_energies, bulk_energies = [], [], [], []
            coefficients = np.random.default_rng(43).normal(size=(9, 5))*.01
            for angle in (0., .6):
                geo = get_geometry.surface(1, 1, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
                rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
                R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                              [-np.sin(angle), 0., np.cos(angle)]])
                for b in rs.basis:
                    cp = np.asarray(b.cp).copy()
                    cp[2] = .2*cp[0]*cp[1]
                    b.cp = R@cp
                assign_gp_and_properties(rs, material_model=LinearElasticMaterial(100., .3, 1.),
                                         properties={'t': .1, 'rho': 1.}, full_quad=True)
                cell = rs.elements[0]
                (RMshell5p if formulation == 'RM5' else RMshell5p_v2).RMShell().element_routine(cell)
                boundary = WeakRM5Boundary(LineString(box(0, 0, 1, 1).exterior.coords),
                    lambda x, y, z: np.zeros(6), formulation=formulation)
                matrix, _, gamma = boundary.integrate_contribution(cell)
                q = np.column_stack((coefficients[:, :3]@R.T, coefficients[:, 3:])).ravel()
                energies.append(q@matrix@q)
                gammas.append(gamma)
                bulk_energies.append(q@cell.K_el@q)
                fixed = WeakRM5Boundary(LineString(box(0, 0, 1, 1).exterior.coords),
                    lambda x, y, z: np.zeros(6), formulation=formulation, penalty=100.)
                fixed_matrix, _, _ = fixed.integrate_contribution(cell)
                fixed_energies.append(q@fixed_matrix@q)
            np.testing.assert_allclose(bulk_energies[0], bulk_energies[1], rtol=1e-10)
            np.testing.assert_allclose(fixed_energies[0], fixed_energies[1], rtol=1e-10)
            np.testing.assert_allclose(energies[0], energies[1], rtol=1e-8)
            np.testing.assert_allclose(gammas[0], gammas[1], rtol=1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
