"""Nonzero affine RM6 patch tests for weak shell boundary conditions."""
import unittest
import numpy as np
from scipy.interpolate import BSpline
from shapely.geometry import box, LineString, Point
from shapely.geometry.polygon import orient
from IGAeasy.trimming.Brep import BRep
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.elements.RMshell6p import RMShell
from IGAeasy.constraints.shell_weak import WeakRM6Boundary
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.solver.direct_solver import solve


class WeakRM6Tests(unittest.TestCase):
    def test_selected_translation_does_not_constrain_other_translation(self):
        geo = get_geometry.surface(1, 1, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
        rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
        assign_gp_and_properties(rs, material_model=LinearElasticMaterial(100., .3, 1.),
                                 properties={'t': .1, 'rho': 1.}, full_quad=True)
        cell = rs.elements[0]
        rule = np.polynomial.legendre.leggauss(3)
        RMShell().element_routine(cell, rule)
        for method in ('penalty', 'nitsche'):
            boundary = WeakRM6Boundary(LineString(box(0, 0, 1, 1).exterior.coords),
                lambda x, y, z: [0., 2., 0., 0., 0., 0.], rule,
                method=method, components=('ux',))
            Kbc, fbc, _ = boundary.integrate_contribution(cell)
            free = np.tile([0., 1., 0., 0., 0., 0.], len(cell.supported_b_splines))
            np.testing.assert_allclose(Kbc@free, 0., atol=1e-11)
            np.testing.assert_allclose(fbc, 0., atol=1e-11)

    def test_curved_boundary_energy_is_invariant_under_spatial_rotation(self):
        energy, gammas = [], []
        q = np.random.default_rng(24).normal(size=(9, 6))*.01
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
            rule = np.polynomial.legendre.leggauss(3)
            RMShell().element_routine(cell, rule)
            boundary = WeakRM6Boundary(LineString(box(0, 0, 1, 1).exterior.coords),
                                      lambda x, y, z: np.zeros(6), rule)
            Kbc, _, gamma = boundary.integrate_contribution(cell)
            rotated = np.column_stack((q[:, :3]@R.T, q[:, 3:]@R.T)).ravel()
            energy.append(rotated@Kbc@rotated)
            gammas.append(gamma)
        np.testing.assert_allclose(energy[0], energy[1], rtol=1e-8)
        np.testing.assert_allclose(gammas[0], gammas[1], rtol=1e-8)

    def patch(self, angle=0., trimmed=False, bending=False, rigid=False, method='nitsche',
              curved=False, discretization='THB', refined=False, hole=False, nonuniform=False):
        geometry = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
        knots = ([0., 0., 0., .3, 1., 1., 1.], [0., 0., 0., .65, 1., 1., 1.]) if nonuniform else geometry.knot_vectors
        rs = (HierarchicalSpace(*knots, 2, geometry.control_points, 2)
              if discretization == 'THB' else
              LRSpline(2, *knots, geometry.control_points))
        R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                      [-np.sin(angle), 0., np.cos(angle)]])
        for b in rs.basis:
            if nonuniform:
                b.cp = np.array([np.mean(b.knots_u[1:3]), np.mean(b.knots_v[1:3]), 0.])
            if curved:
                b.cp = np.asarray(b.cp).copy()
                b.cp[2] = .2*b.cp[0]*b.cp[1]
            b.cp = R@b.cp
        if refined:
            if discretization == 'THB':
                rs.refine({0: [rs.basis[0].loc_id]})
            else:
                rs.refine_by_structured_mesh([rs.basis[0]])
                rs.get_connectivity()
        domain = box(.13, .07, .91, .87) if trimmed else box(0., 0., 1., 1.)
        if trimmed:
            rs.BReps = [BRep(1, polygon=domain, outerloop=True)]
            if hole:
                inner = orient(Point(.5, .5).buffer(.12, quad_segs=8), sign=-1.)
                rs.BReps.append(BRep(2, polygon=inner, outerloop=False))
            trim(rs)
        assign_gp_and_properties(rs, material_model=LinearElasticMaterial(100., .3, 1.),
                                 properties={'t': .1, 'rho': 1.}, full_quad=True)
        rule = np.polynomial.legendre.leggauss(3)
        omega = np.array([.003, -.002, .004])

        def field(x, y, z):
            local = R.T@np.array([x, y, z])
            if rigid:
                u = np.array([.01, -.02, .03])+np.cross(omega, local)
                rotation = omega
            elif bending:
                u = np.array([0., 0., .01*local[0]**2])
                rotation = np.array([0., -.02*local[0], 0.])
            else:
                u = np.array([.01*local[0]+.002*local[1], -.003*local[0]+.02*local[1], 0.])
                rotation = np.array([0., 0., -.0025])
            return np.concatenate((R@u, R@rotation))

        exact = np.concatenate([field(*b.cp) for b in sorted(rs.basis, key=lambda b: b.id)])
        if bending:
            knots = np.asarray(geometry.knot_vectors[0])
            count = len(knots)-3
            samples = np.linspace(0., 1., count)
            coefficients = np.linalg.solve(BSpline.design_matrix(samples, knots, 2).toarray(), samples**2)
            for b in rs.basis:
                index = next(i for i in range(count) if np.array_equal(b.knots_u, knots[i:i+4]))
                exact[6*b.id:6*b.id+3] = R@np.array([0., 0., .01*coefficients[index]])
        lines = [LineString(domain.exterior.coords)]
        if hole:
            lines.append(LineString(inner.exterior.coords))
        boundaries = [WeakRM6Boundary(line, field, rule, method=method, safety_factor=4.) for line in lines]
        for cell in rs.elements:
            if cell.void or not getattr(cell, 'active', True):
                continue
            RMShell().element_routine(cell, rule)
            contributions = [boundary.integrate_contribution(cell) for boundary in boundaries]
            for addition, force, gamma in contributions:
                self.assertGreaterEqual(gamma, 0.)
                cell.K_el += addition
                cell.f_el += force
        K = MatrixAssembler(dof_per_cp=6).assemble(rs, 'K_el', use_sparse=True)
        f = VectorAssembler(dof_per_cp=6).assemble(rs, 'f_el')
        np.testing.assert_allclose(K.toarray(), K.T.toarray(), atol=1e-10)
        residual = np.linalg.norm(K@exact-f)/np.linalg.norm(f)
        self.assertLess(residual, 1e-10)
        computed = solve(K, f, scaling='auto')
        np.testing.assert_allclose(computed, exact, atol=1e-9)
        active = np.flatnonzero(np.asarray(abs(K).sum(axis=1)).ravel() > 0)
        self.assertGreater(np.linalg.eigvalsh(K[active, :][:, active].toarray()).min(), 0.)

    def test_curved_rigid_motion_trimmed_and_locally_refined(self):
        for discretization in ('THB', 'LR'):
            for refined in (False, True):
                for method in ('penalty', 'nitsche'):
                    with self.subTest(discretization=discretization, refined=refined, method=method):
                        self.patch(angle=.6, trimmed=True, rigid=True, curved=True,
                                   discretization=discretization, refined=refined, method=method)

    def test_affine_strain_locally_refined(self):
        for discretization in ('THB', 'LR'):
            for nonuniform in (False, True):
                with self.subTest(discretization=discretization, nonuniform=nonuniform):
                    self.patch(angle=.6, trimmed=True, discretization=discretization,
                               refined=True, nonuniform=nonuniform)

    def test_curved_nonuniform_refined_patch_with_hole(self):
        for discretization in ('THB', 'LR'):
            with self.subTest(discretization=discretization):
                self.patch(angle=.6, trimmed=True, rigid=True, curved=True,
                           discretization=discretization, refined=True, hole=True, nonuniform=True)

    def test_nonzero_membrane_patch_rotated_and_trimmed(self):
        for angle in (0., .6):
            for trimmed in (False, True):
                with self.subTest(angle=angle, trimmed=trimmed):
                    self.patch(angle=angle, trimmed=trimmed)

    def test_constant_bending_patch_rotated_and_trimmed(self):
        for angle in (0., .6):
            for trimmed in (False, True):
                with self.subTest(angle=angle, trimmed=trimmed):
                    self.patch(angle=angle, trimmed=trimmed, bending=True)

    def test_rigid_motion_both_methods(self):
        for method in ('penalty', 'nitsche'):
            self.patch(angle=.6, trimmed=True, rigid=True, method=method)


if __name__ == '__main__':
    unittest.main(verbosity=2)
