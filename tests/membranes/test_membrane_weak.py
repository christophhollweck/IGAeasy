"""Consistency and automatic stabilization of weak membrane boundaries."""
import unittest
import numpy as np
from shapely.geometry import box, LineString
from IGAeasy.trimming.Brep import BRep
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.elements.membrane2p import Membrane
from IGAeasy.constraints.membrane_weak import WeakMembraneBoundary
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.splines.LRSplines.lr_spline import LRSpline


class WeakMembraneTests(unittest.TestCase):
    def test_penalty_affine_boundary_error_decreases_with_parameter(self):
        geo = get_geometry.surface(1, 1, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
        rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
        assign_gp_and_properties(rs, material_model=LinearElasticMaterial(100., .3, 1.),
                                 properties={'t': .2, 'rho': 1.}, full_quad=True)
        cell = rs.elements[0]
        Membrane().element_routine(cell)
        basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
        field = lambda x, y: [.01*x, .02*y]
        exact = np.concatenate([field(*b.cp[:2]) for b in basis])
        errors = []
        for gamma in (1e3, 1e5):
            boundary = WeakMembraneBoundary(LineString(box(0, 0, 1, 1).exterior.coords),
                                            field, method='penalty', penalty=gamma)
            addition, force, _ = boundary.integrate_contribution(cell)
            computed = np.linalg.solve(cell.K_el+addition, force)
            errors.append(np.linalg.norm(computed-exact))
        # A finite pure penalty leaves a consistency error even for an affine field.
        self.assertGreater(errors[0], 1e-6)
        self.assertLess(errors[1], errors[0]/50)

    def test_circle_with_rotated_hole_affine_consistency(self):
        from shapely.geometry import Point
        from shapely.affinity import rotate
        from shapely.geometry.polygon import orient
        outer = orient(Point(.5, .5).buffer(.45, quad_segs=64), sign=1.)
        hole = orient(rotate(box(.35, .4, .65, .6), 30, origin=(.5, .5)), sign=-1.)
        geo = get_geometry.surface(4, 4, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
        rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
        rs.BReps = [BRep(1, polygon=outer, outerloop=True),
                    BRep(2, polygon=hole, outerloop=False)]
        trim(rs)
        assign_gp_and_properties(rs, material_model=LinearElasticMaterial(100., .3, 1.),
                                 properties={'t': .2, 'rho': 1.}, full_quad=True)
        field = lambda x, y: [.01*x+.002*y, -.003*x+.02*y]
        boundaries = [WeakMembraneBoundary(LineString(polygon.exterior.coords), field,
                                           safety_factor=4.) for polygon in (outer, hole)]
        for cell in rs.elements:
            if cell.void:
                continue
            Membrane().element_routine(cell)
            contributions = [boundary.integrate_contribution(cell) for boundary in boundaries]
            for addition, force, gamma in contributions:
                cell.K_el += addition
                cell.f_el += force
        K = MatrixAssembler(dof_per_cp=2).assemble(rs, 'K_el')
        f = VectorAssembler(dof_per_cp=2).assemble(rs, 'f_el')
        exact = np.concatenate([field(*b.cp[:2]) for b in sorted(rs.basis, key=lambda b:b.id)])
        self.assertLess(np.linalg.norm(K@exact-f)/np.linalg.norm(f), 1e-11)
        np.testing.assert_allclose(np.linalg.solve(K, f), exact, atol=1e-10)

    def test_single_direction_keeps_transverse_rigid_motion_free(self):
        geo = get_geometry.surface(1, 1, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
        rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
        assign_gp_and_properties(rs, material_model=LinearElasticMaterial(100., .3, 1.),
                                 properties={'t': .2, 'rho': 1.}, full_quad=True)
        cell = rs.elements[0]
        Membrane().element_routine(cell)
        for method in ('penalty', 'nitsche'):
            boundary = WeakMembraneBoundary(LineString(box(0, 0, 1, 1).exterior.coords),
                                             lambda x, y: [0., 4.], method=method, direction=[2., 0.])
            addition, force, gamma = boundary.integrate_contribution(cell)
            shared = boundary.integrate_both(cell)[method]
            np.testing.assert_allclose(shared[0], addition, atol=1e-12)
            np.testing.assert_allclose(shared[1], force, atol=1e-12)
            self.assertEqual(shared[2], gamma)
            free_translation = np.tile([0., 1.], len(cell.supported_b_splines))
            np.testing.assert_allclose(addition@free_translation, 0., atol=1e-12)
            np.testing.assert_allclose(force, 0., atol=1e-12)
            self.assertGreater(gamma, 0.)

    def test_refined_THB_and_LR_patch(self):
        geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
        for kind in ('THB', 'LR'):
            rs = (HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points, 2) if kind == 'THB'
                  else LRSpline(2, *geo.knot_vectors, geo.control_points))
            if kind == 'THB':
                rs.refine({0: [rs.basis[0].loc_id]})
            else:
                rs.refine_by_structured_mesh([rs.basis[0]])
                rs.get_connectivity()
            assign_gp_and_properties(rs, material_model=LinearElasticMaterial(100., .3, 1.),
                                     properties={'t': .2, 'rho': 1.}, full_quad=True)
            boundary = WeakMembraneBoundary(LineString(box(0, 0, 1, 1).exterior.coords),
                                            lambda x, y: [.01*x, .02*y])
            for cell in rs.elements:
                if not getattr(cell, 'active', True):
                    continue
                Membrane().element_routine(cell)
                addition, force, _ = boundary.integrate_contribution(cell)
                cell.K_el += addition
                cell.f_el += force
            K = MatrixAssembler(dof_per_cp=2).assemble(rs, 'K_el')
            f = VectorAssembler(dof_per_cp=2).assemble(rs, 'f_el')
            exact = np.concatenate([[.01*b.cp[0], .02*b.cp[1]]
                                    for b in sorted(rs.basis, key=lambda b: b.id)])
            np.testing.assert_allclose(K@exact, f, atol=1e-11)
            active = np.flatnonzero(np.any(K != 0., axis=1))
            self.assertGreater(np.linalg.eigvalsh(K[np.ix_(active, active)]).min(), 0.)
            np.testing.assert_allclose(np.linalg.solve(K[np.ix_(active, active)], f[active]),
                                       exact[active], atol=1e-11)

    def test_trimmed_patch_automatic_nitsche(self):
        self.check_trimmed_patch(.2)
        self.check_trimmed_patch(.05)

    def check_trimmed_patch(self, material_fraction):
        geo = get_geometry.surface(1, 1, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
        rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
        domain = box(1-material_fraction, 0., 1., 1.)
        rs.BReps = [BRep(1, polygon=domain, outerloop=True)]
        trim(rs)
        assign_gp_and_properties(rs, material_model=LinearElasticMaterial(100., .3, 1.),
                                 properties={'t': .2, 'rho': 1.}, full_quad=True)
        cell = rs.elements[0]
        Membrane().element_routine(cell)
        boundary = WeakMembraneBoundary(LineString(domain.exterior.coords), lambda x, y: [.01*x, .02*y])
        addition, force, gamma = boundary.integrate_contribution(cell)
        K = cell.K_el+addition
        self.assertGreater(gamma, 0.)
        self.assertGreater(np.linalg.eigvalsh(K).min(), 0.)
        exact = np.concatenate([[.01*b.cp[0], .02*b.cp[1]]
                                for b in sorted(cell.supported_b_splines, key=lambda b: b.id)])
        np.testing.assert_allclose(K@exact, force, atol=1e-10)
        np.testing.assert_allclose(np.linalg.solve(K, force), exact, atol=1e-9)

    def test_affine_patch_with_automatic_nitsche(self):
        geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
        rs = HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points)
        assign_gp_and_properties(rs, material_model=LinearElasticMaterial(100., .3, 1.),
                                 properties={'t': .2, 'rho': 1.}, full_quad=True)
        boundary = WeakMembraneBoundary(LineString(box(0, 0, 1, 1).exterior.coords),
                                        lambda x, y: [.01*x+.002*y, -.003*x+.02*y])
        for cell in rs.elements:
            Membrane().element_routine(cell)
            addition, force, gamma = boundary.integrate_contribution(cell)
            self.assertGreater(gamma, 0.)
            cell.K_el += addition
            cell.f_el += force
        K = MatrixAssembler(dof_per_cp=2).assemble(rs, 'K_el')
        f = VectorAssembler(dof_per_cp=2).assemble(rs, 'f_el')
        self.assertGreater(np.linalg.eigvalsh(K).min(), 0.)
        exact = np.concatenate([[.01*b.cp[0]+.002*b.cp[1], -.003*b.cp[0]+.02*b.cp[1]]
                                for b in sorted(rs.basis, key=lambda b: b.id)])
        np.testing.assert_allclose(K@exact, f, atol=1e-12)
        np.testing.assert_allclose(np.linalg.solve(K, f), exact, atol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
