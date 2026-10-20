"""Exact mixed-curvature geometry and independent nonzero-director RM5 loads."""
import unittest
import numpy as np
from IGAeasy.geometry import get_geometry
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.differential_geometry.derivatives import get_full_geometric_data
from IGAeasy.manufactured_solutions.rm_paraboloid_reference import ParaboloidRMReference
from IGAeasy.examples.shells.verification.manufactured.paraboloid_convergence import solve_level


class PolynomialShellTests(unittest.TestCase):
    def test_symbolic_graph_metric_matches_direct_matrix_inverse(self):
        import sympy as sp
        from IGAeasy.manufactured_solutions.shell_symbolics import paraboloid_geometry
        coordinates, _, metric, inverse, J, normal = paraboloid_geometry((.4, -.2), True, (1.2, -.8))
        evaluate = sp.lambdify(coordinates, (metric, inverse, J, normal), 'numpy', cse=True)
        for u, v in ((.1, .2), (.5, .5), (.8, .9)):
            G, Ginv, area, n = evaluate(u, v)
            np.testing.assert_allclose(Ginv, np.linalg.inv(G), atol=1e-14)
            self.assertAlmostEqual(area, np.sqrt(np.linalg.det(G)), places=13)
            self.assertAlmostEqual(np.linalg.norm(n), 1., places=13)

    def test_exact_mixed_geometry_and_its_fundamental_forms(self):
        a, b, c, d = .4, -.2, 1.2, -.8
        R = rotation_matrix(.6)
        for degree in (2, 3):
            geometry = get_geometry.polynomial_shell(3, 2, degree, (a, b), (c, d))
            space = HierarchicalSpace(*geometry.knot_vectors, degree, geometry.control_points@R.T)
            for cell in space.elements:
                u, v = (cell.u_min+cell.u_max)/2, (cell.v_min+cell.v_max)/2
                x, y = u-.5, v-.5
                expected = R@[x, y, -a*x*x-b*y*y+c*x*x*y+d*x*y*y]
                basis = list(cell.supported_b_splines)
                actual = sum((bf(u, v, cell=cell)*bf.cp for bf in basis), np.zeros(3))
                np.testing.assert_allclose(actual, expected, atol=1e-14)
                slope = np.array([-2*a*x+2*c*x*y+d*y*y, -2*b*y+c*x*x+2*d*x*y])
                hessian = np.array([[-2*a+2*c*y, 2*c*x+2*d*y],
                                    [2*c*x+2*d*y, -2*b+2*d*x]])
                _, _, _, metric, second, _ = get_full_geometric_data(cell, u, v)
                np.testing.assert_allclose(metric, np.eye(2)+np.outer(slope, slope), atol=3e-14)
                np.testing.assert_allclose(second, hessian/np.sqrt(1+slope@slope), atol=1e-13)
                self.assertGreaterEqual(np.linalg.det(metric), 1.-1e-13)

    def test_nonzero_director_flat_load_matches_continuum_equilibrium(self):
        E, nu, t, A, Q = 100., .3, .1, .001, .0003
        reference = ParaboloidRMReference((0., 0.), centered=True,
                                         director_amplitude=Q, enriched=True)
        shear = (5/6)*E*t/(2*(1+nu))
        for u, v in ((.23, .41), (.61, .72)):
            w = A*np.sin(np.pi*u)*np.sin(np.pi*v)
            gradient = A*np.pi*np.array([np.cos(np.pi*u)*np.sin(np.pi*v),
                                        np.sin(np.pi*u)*np.cos(np.pi*v)])
            q = Q*np.array([1+u, .5+v])
            fy = -E*t*A/(4*(1+nu))-nu*E*t*A/(10*(1-nu**2))
            expected = np.r_[0., fy, shear*(2*np.pi**2*w-2*Q), shear*(gradient+q)]
            np.testing.assert_allclose(reference.body(u, v), expected, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(reference.director(u-.5, v-.5, 0.), np.r_[q, 0.], atol=1e-15)

    def test_enriched_rm5_saddle_convergence_and_rotation(self):
        rows = [solve_level(n, director_amplitude=.0003, enriched=True,
                            surface_curvature=(1., -1.))[1] for n in (4, 8)]
        original = solve_level(4, rotation_angle=0., director_amplitude=.0003,
                               enriched=True, surface_curvature=(1., -1.))[1]
        self.assertGreater(np.log2(rows[0]['l2']/rows[1]['l2']), 3.5)
        self.assertLess(rows[1]['director_l2'], 4e-8)
        for name in ('l2', 'director_l2'):
            np.testing.assert_allclose(original[name], rows[0][name], rtol=2e-7, atol=1e-13)


if __name__ == '__main__':
    unittest.main(verbosity=2)
