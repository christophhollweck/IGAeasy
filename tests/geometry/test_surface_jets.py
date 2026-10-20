"""Geometry derivatives against an analytic graph, also after local refinement."""
import unittest
import numpy as np
from IGAeasy.differential_geometry.surface_jets import surface_jet
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline


class SurfaceJetTests(unittest.TestCase):
    def test_graph_and_local_refinement(self):
        knots = [0.]*4+[.3, .7]+[1.]*4
        greville = [np.mean(knots[i+1:i+4]) for i in range(6)]
        cp = [[u, v, 0.] for v in greville for u in greville]
        angle = .7
        R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
        for family in ('THB', 'LR'):
            rs = HierarchicalSpace(knots, knots, 3, cp, 2) if family == 'THB' else LRSpline(3, knots, knots, cp)
            for basis in rs.basis:
                ku, kv = basis.knots_u[1:-1], basis.knots_v[1:-1]
                u2 = sum(ku[i]*ku[j] for i in range(3) for j in range(i+1, 3))/3
                z = .2*u2+.3*np.mean(ku)*np.mean(kv)+.1*np.prod(kv)
                basis.cp = R@np.array([basis.cp[0], basis.cp[1], z])
            if family == 'THB':
                rs.refine({0: [rs.basis[0].loc_id]})
            else:
                rs.refine_by_structured_mesh([rs.basis[0]])
                rs.get_connectivity()
            for cell in rs.elements:
                u, v = (cell.u_min+cell.u_max)/2, (cell.v_min+cell.v_max)/2
                jet = surface_jet(cell, u, v)
                tangents = np.array([[1., 0., .4*u+.3*v], [0., 1., .3*u+.3*v*v]])@R.T
                np.testing.assert_allclose(jet.tangents, tangents, atol=2e-12)
                np.testing.assert_allclose(jet.third[1, 1, 1], R@[0., 0., .6], atol=2e-10)
                np.testing.assert_allclose(jet.third[0], 0., atol=2e-10)
                def normal(x, y):
                    vector = np.array([-.4*x-.3*y, -.3*x-.3*y*y, 1.])
                    return R@(vector/np.linalg.norm(vector))
                h = 1e-4
                for a in range(2):
                    step = np.eye(2)[a]*h
                    plus, minus = normal(u+step[0], v+step[1]), normal(u-step[0], v-step[1])
                    np.testing.assert_allclose(jet.normal_gradient[a], (plus-minus)/(2*h), atol=2e-9)
                    np.testing.assert_allclose(jet.normal_hessian[a, a], (plus-2*normal(u, v)+minus)/h**2, atol=6e-8)
                np.testing.assert_allclose(jet.contravariant@jet.tangents.T, np.eye(2), atol=1e-12)
                mixed = (normal(u+h, v+h)-normal(u+h, v-h)-normal(u-h, v+h)+normal(u-h, v-h))/(4*h*h)
                np.testing.assert_allclose(jet.normal_hessian[0, 1], mixed, atol=3e-8)
                for a in range(2):
                    delta = np.eye(2)[a]*h
                    left, right = surface_jet(cell, u-delta[0], v-delta[1]), surface_jet(cell, u+delta[0], v+delta[1])
                    np.testing.assert_allclose(jet.gamma_gradient[a], (right.gamma-left.gamma)/(2*h), atol=3e-9)
                    np.testing.assert_allclose(jet.inverse_metric_gradient[a], (right.inverse_metric-left.inverse_metric)/(2*h), atol=3e-9)

    def test_material_directional_derivative(self):
        material = LinearElasticMaterial(10920., .3, 1.)
        metric = np.array([[1.3, .2], [.2, .8]])
        direction = np.array([[.4, -.3], [-.3, .2]])
        h = 1e-5
        derivative = material.compute_C_voigt_curv_2D_directional(metric, direction, .1)
        left = material.compute_C_voigt_curv_2D(metric-h*direction, .1)
        right = material.compute_C_voigt_curv_2D(metric+h*direction, .1)
        for actual, a, b in zip(derivative, left, right):
            np.testing.assert_allclose(actual, (b-a)/(2*h), rtol=1e-9, atol=1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
