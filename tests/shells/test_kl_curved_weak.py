"""Curved KL manufactured solutions, trimming, local refinement and rotation."""
import unittest
import numpy as np
from shapely.geometry import box
from shapely.affinity import rotate
from IGAeasy.trimming.Brep import BRep
from IGAeasy.manufactured_solutions.shell_verification import solve_gradient_patch as solve_level
from IGAeasy.manufactured_solutions.kl_parabolic_reference import ParabolicKLReference
from shapely.geometry import LineString
from shapely.geometry.polygon import orient
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.elements.KLshell3p import KLShell
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.constraints.kl_clamped_weak import WeakKLClampedBoundary
from IGAeasy.constraints.kl_mixed_weak import WeakKLMixedBoundary
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.trimming.trim_elements import trim


class KLCurvedTests(unittest.TestCase):
    def test_degree_four_exact_quartic_and_degree_two_approximation(self):
        _, _, exact = solve_level(2, degree=4)
        self.assertLess(exact['l2'], 1e-8)
        self.assertLess(exact['patch_residual'], 1e-8)
        _, _, approximate = solve_level(2, degree=2, power=3)
        self.assertNotIn('patch_residual', approximate)
        self.assertGreater(approximate['l2'], 0.)

    def test_cubic_l2_rate_with_and_without_trimming(self):
        # Cubic bases approximate the quartic solution: displacement L2 ~ h**4.
        for trimmed in (False, True):
            with self.subTest(trimmed=trimmed):
                rows = [solve_level(n, trimmed=trimmed, angle=.6)[2] for n in (2, 4, 8)]
                rates = np.log2(np.array([r['l2'] for r in rows[:-1]])/np.array([r['l2'] for r in rows[1:]]))
                self.assertGreater(rates[-1], 3.6)
                self.assertLess(rates[-1], 4.5)
                for row in rows:
                    self.assertLess(row['residual'], 1e-11)
                    self.assertTrue(np.isfinite(row['gamma_max']) and row['gamma_max'] > 0.)

    def test_locally_refined_thb_lr_and_rotation(self):
        # Cubic reference lies in both spaces; remaining error is quadrature.
        for family in ('THB', 'LR'):
            with self.subTest(family=family):
                reference_space, reference, original = solve_level(4, family=family, trimmed=True,
                    power=3, local_refinement=True)
                rotated_space, rotated, row = solve_level(4, family=family, trimmed=True,
                    angle=.6, power=3, local_refinement=True)
                axis = np.array([1., 2., 3.])/np.sqrt(14.)
                cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
                R = np.eye(3)+np.sin(.6)*cross+(1-np.cos(.6))*cross@cross
                np.testing.assert_allclose(rotated.reshape(-1, 3), reference.reshape(-1, 3)@R.T, atol=1e-9, rtol=1e-7)
                self.assertLess(row['l2'], 1e-7)
                self.assertAlmostEqual(row['gamma_max']/original['gamma_max'], 1., places=7)

    def test_reference_flat_limit(self):
        quartic = ParabolicKLReference(curvature=0.)
        D = quartic.E*quartic.thickness**3/(12*(1-quartic.nu**2))
        for u in (.1, .6, .9):
            np.testing.assert_allclose(quartic.body(u, .4), [0., 0., 24*D*quartic.amplitude])
            np.testing.assert_allclose(ParabolicKLReference(power=3, curvature=0.).body(u, .4), 0.)

    def test_curved_patch_with_rotated_inner_trim(self):
        domain = box(.07, .09, .93, .91).difference(rotate(box(.37, .39, .63, .61), 27., origin=(.5, .5)))
        for family in ('THB', 'LR'):
            with self.subTest(family=family):
                _, _, row = solve_level(4, family=family, angle=.6, power=3,
                    local_refinement=True, domain=domain)
                self.assertLess(row['l2'], 1e-8)
                self.assertLess(row['residual'], 1e-11)

    def test_rigid_motion_on_doubly_curved_refined_patch(self):
        omega = np.array([.003, -.002, .004])
        translation = np.array([.001, .002, -.001])
        domain = orient(box(0., 0., 1., 1.).difference(rotate(box(.37, .39, .63, .61), 27., origin=(.5, .5))), sign=1.)
        cases = [(family, boundary) for family in ('THB', 'LR')
                 for boundary in (WeakKLClampedBoundary, WeakKLMixedBoundary)]
        for family, boundary_class in cases:
            with self.subTest(family=family, boundary=boundary_class.__name__):
                geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 2, 3, distortion=0, n_waves=0)
                rs = (HierarchicalSpace(*geo.knot_vectors, 3, geo.control_points, 2) if family == 'THB'
                      else LRSpline(3, *geo.knot_vectors, geo.control_points))
                for b in rs.basis:
                    ku, kv = b.knots_u[1:-1], b.knots_v[1:-1]
                    squares = [sum(k[i]*k[j] for i in range(3) for j in range(i+1, 3))/3 for k in (ku, kv)]
                    b.cp = np.array([b.cp[0], b.cp[1], .1*squares[0]+.2*np.mean(ku)*np.mean(kv)+.15*squares[1]])
                if family == 'THB':
                    rs.refine({0: [rs.basis[0].loc_id]})
                else:
                    rs.refine_by_structured_mesh([rs.basis[0]])
                    rs.get_connectivity()
                rs.BReps = [BRep(1, polygon=domain, outerloop=True)]
                trim(rs)
                assign_gp_and_properties(rs, material_model=LinearElasticMaterial(10920., .3, 1.), properties={'t': .1}, full_quad=True)
                def field(x, y, z):
                    normal = np.array([-.2*x-.2*y, -.2*x-.3*y, 1.])
                    normal /= np.linalg.norm(normal)
                    return np.r_[translation+np.cross(omega, [x, y, z]), np.cross(omega, normal)]
                boundaries = [boundary_class(LineString(ring.coords), field)
                              for ring in (domain.exterior, *domain.interiors)]
                exact = np.zeros(3*len(rs.basis))
                for b in rs.basis:
                    if getattr(b, 'active', True):
                        exact[3*b.id:3*b.id+3] = translation+np.cross(omega, b.cp)
                for cell in rs.elements:
                    if cell.void:
                        continue
                    KLShell().element_routine(cell)
                    contributions = [boundary.integrate_contribution(cell) for boundary in boundaries]
                    for kb, fb, _ in contributions:
                        cell.K_el += kb
                        cell.f_el += fb
                K = MatrixAssembler(dof_per_cp=3).assemble(rs, 'K_el', use_sparse=True)
                f = VectorAssembler(dof_per_cp=3).assemble(rs, 'f_el')
                np.testing.assert_allclose(K@exact, f, atol=2e-9, rtol=1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
