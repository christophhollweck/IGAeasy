"""Constraint collection, conflict detection and nonzero elimination."""
import unittest
import numpy as np
from IGAeasy.constraints.prescribed_dofs import PrescribedDofs
from IGAeasy.constraints.dirichlet import block_partitioning_dbc


class PrescribedDofTests(unittest.TestCase):
    def test_component_order_and_overlapping_constraints(self):
        conditions = PrescribedDofs(('x', 'y', 'z', 'r_a1', 'r_a2'), 4)
        conditions.add({3, 1}, ('z', 'r_a2')).add([1], 'z')
        indices, values = conditions.arrays()
        np.testing.assert_array_equal(indices, [7, 9, 17, 19])
        np.testing.assert_array_equal(values, np.zeros(4))
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            conditions.add([0, 1], 'z', 2.)
        np.testing.assert_array_equal(conditions.arrays()[0], indices)

    def test_nonzero_constraints_with_existing_solver(self):
        conditions = PrescribedDofs(('temperature',), 2).add([0], 'temperature', 3.)
        indices, values = conditions.arrays()
        K, f, free = block_partitioning_dbc(K=np.array([[2., -1.], [-1., 2.]]),
                                           f=np.zeros(2), prescribed_DOFs=indices, u_c=values)
        np.testing.assert_array_equal(free, [1])
        np.testing.assert_allclose(np.linalg.solve(K, f), [1.5])

    def test_invalid_ids_components_and_values(self):
        conditions = PrescribedDofs(('x', 'y'), 2)
        for cp in (-1, 2, True, .5):
            with self.assertRaises(ValueError):
                conditions.add([cp], 'x')
        with self.assertRaises(ValueError):
            conditions.add([0], 'rz')
        with self.assertRaises(ValueError):
            conditions.add([0], 'x', np.nan)


if __name__ == '__main__':
    unittest.main(verbosity=2)
