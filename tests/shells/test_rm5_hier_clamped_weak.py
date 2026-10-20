"""Full hierarchical RM5 Nitsche against a shear-consistent cubic solution."""
import unittest
import numpy as np
from shapely.geometry import box, LineString
from shapely.affinity import rotate
from shapely.geometry.polygon import orient
from IGAeasy.trimming.Brep import BRep
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.elements.RMshell5p_hier import RMShell
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.constraints.kl_clamped_weak import WeakRM5HierAffineClampedBoundary
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.solver.direct_solver import solve
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.manufactured_solutions.shell_verification import solve_gradient_patch as solve_level
from IGAeasy.manufactured_solutions.kl_parabolic_reference import ParabolicRM5HierReference
from IGAeasy.error_norms.surface_norms import SurfaceDirectorL2Quadrature


class RM5HierClampedTests(unittest.TestCase):
    def test_curved_manufactured_field_with_hole_and_rotation(self):
        domain = box(.07, .09, .93, .91).difference(rotate(box(.37, .39, .63, .61), 27., origin=(.5, .5)))
        axis = np.array([1., 2., 3.])/np.sqrt(14.)
        cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
        R = np.eye(3)+np.sin(.6)*cross+(1-np.cos(.6))*cross@cross
        for family in ('THB', 'LR'):
            with self.subTest(family=family):
                original_space, original, row0 = solve_level(4, family=family, power=3,
                    local_refinement=True, domain=domain, formulation='RM5_hier')
                _, rotated, row = solve_level(4, family=family, power=3, angle=.6,
                    local_refinement=True, domain=domain, formulation='RM5_hier')
                self.assertLess(row['l2'], 1e-8)
                self.assertLess(row['patch_residual'], 1e-9)
                a, b = original.reshape(-1, 5), rotated.reshape(-1, 5)
                np.testing.assert_allclose(b[:, :3], a[:, :3]@R.T, atol=1e-9, rtol=1e-7)
                # Weakly supported cut coefficients amplify roundoff. Compare
                # the actual physical director field, including displacement gradients.
                back_rotated = b.copy()
                back_rotated[:, :3] = b[:, :3]@R
                director_norm = SurfaceDirectorL2Quadrature(original_space, 'RM5_hier', order=6)
                self.assertLess(director_norm.error(back_rotated, a), 1e-10)
                self.assertLess(row['director_l2'], 1e-8)
                self.assertAlmostEqual(row['gamma_max']/row0['gamma_max'], 1., places=7)

    def test_curved_quartic_refinement_and_director_rate(self):
        rows = [solve_level(n, trimmed=True, angle=.6, formulation='RM5_hier')[2] for n in (2, 4, 8)]
        rate = np.log2(rows[-2]['l2']/rows[-1]['l2'])
        self.assertGreater(rate, 3.)
        self.assertLess(rate, 4.5)
        director_rate = np.log2(rows[-2]['director_l2']/rows[-1]['director_l2'])
        self.assertGreater(director_rate, 2.7)
        self.assertLess(director_rate, 3.3)
        for row in rows:
            self.assertLess(row['residual'], 1e-11)

    def test_curved_reference_flat_limit(self):
        E, nu, t, W = 10920., .3, .1, .01
        D, G = E*t**3/(12*(1-nu**2)), (5/6)*E*t/(2*(1+nu))
        reference = ParabolicRM5HierReference(power=3, curvature=0., eta=-6*D*W/G)
        for u in (.1, .6, .9):
            np.testing.assert_allclose(reference.body(u, .4), 0., atol=1e-14)

    def test_shear_consistent_cubic_on_scaled_rotated_trimmed_patch(self):
        E, nu, t, W = 10920., .3, .1, .01
        D, G = E*t**3/(12*(1-nu**2)), (5/6)*E*t/(2*(1+nu))
        extra_director = -6*D*W/G
        angle = .6
        R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
        for family in ('THB', 'LR'):
            for with_hole in (False, True):
                with self.subTest(family=family, with_hole=with_hole):
                    geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 2, 3, distortion=0, n_waves=0)
                    rs = (HierarchicalSpace(*geo.knot_vectors, 3, geo.control_points, 2) if family == 'THB'
                          else LRSpline(3, *geo.knot_vectors, geo.control_points))
                    for b in rs.basis:
                        b.cp = R@(np.asarray(b.cp)*[2., 3., 1.])
                        b.scalar_value = 8*W*np.prod(b.knots_u[1:-1])
                    if family == 'THB':
                        rs.refine({0: [rs.basis[0].loc_id]})
                    else:
                        rs.refine_by_structured_mesh([rs.basis[0]])
                        rs.get_connectivity()
                    domain = box(0., 0., 1., 1.)
                    if with_hole:
                        domain = domain.difference(rotate(box(.37, .39, .63, .61), 27., origin=(.5, .5)))
                    domain = orient(domain, sign=1.)
                    rs.BReps = [BRep(1, polygon=domain, outerloop=True)]
                    trim(rs)
                    assign_gp_and_properties(rs, material_model=LinearElasticMaterial(E, nu, 1.),
                        properties={'t': t}, full_quad=True)
                    def field(x, y, z):
                        local = R.T@np.array([x, y, z])
                        return np.r_[R@[0., 0., W*local[0]**3], R@[extra_director-3*W*local[0]**2, 0., 0.]]
                    boundaries = [WeakRM5HierAffineClampedBoundary(LineString(ring.coords), field)
                        for ring in (domain.exterior, *domain.interiors)]
                    exact = np.zeros(5*len(rs.basis))
                    for b in rs.basis:
                        if getattr(b, 'active', True):
                            exact[5*b.id:5*b.id+3] = R@[0., 0., b.scalar_value]
                            exact[5*b.id+3] = extra_director/2
                    for cell in rs.elements:
                        if cell.void:
                            continue
                        RMShell().element_routine(cell)
                        contributions = [boundary.integrate_contribution(cell) for boundary in boundaries]
                        for kb, fb, gamma in contributions:
                            cell.K_el += kb
                            cell.f_el += fb
                    K = MatrixAssembler(dof_per_cp=5).assemble(rs, 'K_el', use_sparse=True)
                    f = VectorAssembler(dof_per_cp=5).assemble(rs, 'f_el')
                    np.testing.assert_allclose(K@exact, f, atol=1e-9, rtol=1e-10)
                    solution = solve(K, f, scaling='auto')
                    active = np.flatnonzero(K.diagonal() > 0.)
                    np.testing.assert_allclose(solution[active], exact[active], atol=1e-8, rtol=1e-7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
