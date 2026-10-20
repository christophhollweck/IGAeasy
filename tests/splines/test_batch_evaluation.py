import unittest
import numpy as np
from IGAeasy.splines.batch_evaluation import evaluate_basis_batch, evaluate_derivatives_batch
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline


class BatchTests(unittest.TestCase):
    def test_refined_basis_values_derivatives_and_boundaries(self):
        knots = [0., 0., 0., .3, .7, 1., 1., 1.]
        cps = [[u, v, 0.] for v in (0., .15, .5, .85, 1.) for u in (0., .15, .5, .85, 1.)]
        for kind in ('THB', 'LR'):
            space = (HierarchicalSpace(knots, knots, 2, cps, 2) if kind == 'THB'
                     else LRSpline(2, knots, knots, cps))
            if kind == 'THB':
                space.refine({0: [space.basis[0].loc_id]})
            else:
                space.refine_by_structured_mesh([space.basis[0]])
                space.get_connectivity()
            for cell in space.elements:
                if not getattr(cell, 'active', True):
                    continue
                points = np.array([[u, v] for u in np.linspace(cell.u_min, cell.u_max, 5)
                                   for v in np.linspace(cell.v_min, cell.v_max, 5)])
                basis = list(cell.supported_b_splines)
                values, gradients = evaluate_basis_batch(basis, points, cell)
                expected = np.array([[b(u, v, cell=cell) for b in basis] for u, v in points])
                derivative = np.array([[b.grad(u, v, cell=cell) for b in basis] for u, v in points])
                np.testing.assert_allclose(values, expected, atol=2e-13, rtol=1e-12)
                np.testing.assert_allclose(gradients, derivative, atol=2e-12, rtol=1e-12)
                all_derivatives = evaluate_derivatives_batch(basis, points, cell, max_order=3)
                for (a, b), data in all_derivatives.items():
                    scalar = np.array([[f(u, v, r1=a, r2=b, cell=cell) for f in basis] for u, v in points])
                    np.testing.assert_allclose(data, scalar, atol=2e-9, rtol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
