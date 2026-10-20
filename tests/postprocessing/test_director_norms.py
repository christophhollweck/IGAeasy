"""Physical director norms against exact polynomial integrals."""
import unittest
import numpy as np
from shapely.geometry import box
from IGAeasy.trimming.Brep import BRep
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.error_norms.surface_norms import SurfaceDirectorL2Quadrature


class DirectorNormTests(unittest.TestCase):
    def test_standard_rm_directors_on_rotated_curved_refined_trimmed_surfaces(self):
        angle = .6
        R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                      [-np.sin(angle), 0., np.cos(angle)]])
        for family in ('THB', 'LR'):
            geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
            rs = (HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points, 2)
                  if family == 'THB' else LRSpline(2, *geo.knot_vectors, geo.control_points))
            for b in rs.basis:
                u, v = b.cp[:2]
                b.cp = R@np.array([2*u, 3*v, .2*u*v])
            if family == 'THB':
                rs.refine({0: [rs.basis[0].loc_id]})
            else:
                rs.refine_by_structured_mesh([rs.basis[0]])
                rs.get_connectivity()
            rs.BReps = [BRep(1, polygon=box(.1, .2, .9, .8), outerloop=True)]
            trim(rs)
            for formulation, dof in (('RM6', 6), ('RM5', 5), ('RM5_v2', 5)):
                with self.subTest(family=family, formulation=formulation):
                    coefficients = np.zeros((len(rs.basis), dof))
                    components = R@np.array([.01, .02, .03]) if dof == 6 else np.array([.01, .02])
                    for b in rs.basis:
                        coefficients[b.id, 3:] = components
                    def reference(x, y, z):
                        local = R.T@np.array([x, y, z])
                        u, v = local[0]/2, local[1]/3
                        tangents = np.array([[2., 0., .2*v], [0., 3., .2*u]])
                        if formulation == 'RM6':
                            normal = np.cross(*tangents)
                            normal /= np.linalg.norm(normal)
                            return R@np.cross([.01, .02, .03], normal)
                        vectors = (np.linalg.solve(tangents@tangents.T, tangents)
                                   if formulation == 'RM5' else tangents)
                        return R@(components@vectors)
                    norm = SurfaceDirectorL2Quadrature(rs, formulation, order=5)
                    self.assertGreater(norm.error(coefficients), 0.)
                    self.assertLess(norm.error(coefficients, reference), 1e-12)
                    self.assertLess(norm.error(coefficients, reference, relative=True), 1e-11)

    def test_rotated_trimmed_polynomial_after_local_refinement(self):
        angle = .6
        R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
        for family in ('THB', 'LR'):
            for formulation, dof in (('KL', 3), ('RM5_hier', 5)):
                with self.subTest(family=family, formulation=formulation):
                    geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
                    rs = (HierarchicalSpace(*geo.knot_vectors, 2, geo.control_points, 2) if family == 'THB'
                          else LRSpline(2, *geo.knot_vectors, geo.control_points))
                    for b in rs.basis:
                        b.cp = R@(np.asarray(b.cp)*[2., 3., 1.])
                        b.scalar_value = .04*np.prod(b.knots_u[1:-1])
                    if family == 'THB':
                        rs.refine({0: [rs.basis[0].loc_id]})
                    else:
                        rs.refine_by_structured_mesh([rs.basis[0]])
                        rs.get_connectivity()
                    rs.BReps = [BRep(1, polygon=box(.1, .2, .9, .8), outerloop=True)]
                    trim(rs)
                    coefficients = np.zeros((len(rs.basis), dof))
                    for b in rs.basis:
                        if getattr(b, 'active', True):
                            coefficients[b.id, :3] = R@[0., 0., b.scalar_value]
                            if dof == 5:
                                coefficients[b.id, 3] = .03
                    a, slope = (.06 if dof == 5 else 0.), -.02
                    def reference(x, y, z):
                        local_x = (R.T@np.array([x, y, z]))[0]
                        return R@[a+slope*local_x, 0., 0.]
                    antiderivative = lambda x: a*a*x+a*slope*x*x+slope*slope*x**3/3
                    expected = np.sqrt(1.8*(antiderivative(1.8)-antiderivative(.2)))
                    norm = SurfaceDirectorL2Quadrature(rs, formulation)
                    self.assertAlmostEqual(norm.error(coefficients), expected, places=12)
                    self.assertLess(norm.error(coefficients, reference), 1e-12)
                    self.assertLess(norm.error(coefficients, reference, relative=True), 1e-11)
                    self.assertEqual(norm.error(coefficients, coefficients), 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
