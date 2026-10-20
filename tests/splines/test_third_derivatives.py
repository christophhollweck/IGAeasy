"""Third derivatives against global SciPy splines and exact polynomial transfer."""
import unittest
import numpy as np
from scipy.interpolate import BSpline
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline


class ThirdDerivativeTests(unittest.TestCase):
    def space(self, family, refined=False):
        ku = [0.]*4+[.2, .7]+[1.]*4
        kv = [0.]*4+[.4, .9]+[1.]*4
        gu = [np.mean(ku[i+1:i+4]) for i in range(6)]
        gv = [np.mean(kv[i+1:i+4]) for i in range(6)]
        cps = [[u, v, 0.] for v in gv for u in gu]
        rs = (HierarchicalSpace(ku, kv, 3, cps, 2) if family == 'THB' else LRSpline(3, ku, kv, cps))
        for b in rs.basis:
            b.scalar_value = np.prod(b.knots_u[1:-1])*np.prod(b.knots_v[1:-1])
        if refined:
            if family == 'THB':
                rs.refine({0: [rs.basis[0].loc_id]})
            else:
                rs.refine_by_structured_mesh([rs.basis[0]])
                rs.get_connectivity()
        return rs, ku, kv

    def test_nonuniform_tensor_basis_against_scipy(self):
        for family in ('THB', 'LR'):
            rs, ku, kv = self.space(family)
            for b in rs.basis:
                i = next(i for i in range(6) if np.array_equal(b.knots_u, ku[i:i+5]))
                j = next(j for j in range(6) if np.array_equal(b.knots_v, kv[j:j+5]))
                u_basis, v_basis = BSpline(ku, np.eye(6)[i], 3), BSpline(kv, np.eye(6)[j], 3)
                for u, v in ((0., 0.), (.13, .37), (.45, .73), (1., 1.)):
                    cell = next(cell for cell in rs.elements if cell.u_min <= u <= cell.u_max
                                and cell.v_min <= v <= cell.v_max)
                    expected = [u_basis(u, nu=a)*v_basis(v, nu=3-a) for a in (3, 2, 1, 0)]
                    np.testing.assert_allclose(b.grad3(u, v, cell=cell), expected, rtol=1e-12, atol=1e-10)

    def test_polynomial_reproduction_after_refinement_and_cell_sidedness(self):
        for family in ('THB', 'LR'):
            for refined in (False, True):
                rs, _, _ = self.space(family, refined)
                for cell in rs.elements:
                    if not getattr(cell, 'active', True):
                        continue
                    for u, v in ((cell.u_min, cell.v_min), (cell.u_max, cell.v_max),
                                  ((cell.u_min+cell.u_max)/2, (cell.v_min+cell.v_max)/2)):
                        basis = list(cell.supported_b_splines)
                        values = np.array([b.grad3(u, v, cell=cell) for b in basis])
                        np.testing.assert_allclose(values.sum(axis=0), 0., atol=2e-9)
                        computed = np.array([b.scalar_value for b in basis])@values
                        expected = [6*v**3, 18*u*v**2, 18*u**2*v, 6*u**3]
                        np.testing.assert_allclose(computed, expected, rtol=1e-9, atol=2e-9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
