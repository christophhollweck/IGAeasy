"""Regression tests for global matrix and vector assembly."""
import unittest
from types import SimpleNamespace
import numpy as np
from IGAeasy.assembler import Assembler, MatrixAssembler, VectorAssembler


class AssemblyTests(unittest.TestCase):
    def test_selection_empty_mesh_and_vector_shape(self):
        empty = SimpleNamespace(basis=[None]*2, elements=[])
        result = Assembler(2).assemble(empty, use_sparse=True, assemble_K=False, assemble_M=True, assemble_f=False)
        self.assertEqual(set(result), {'M'})
        self.assertEqual(result['M'].shape, (4, 4))
        self.assertEqual(result['M'].nnz, 0)
        bad = SimpleNamespace(basis=[None], elements=[SimpleNamespace(bfnums=[0], f_el=[1, 2])])
        with self.assertRaises(ValueError):
            VectorAssembler(1).assemble(bad, 'f_el')
        for count in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                Assembler(count)

    def test_local_order_and_shared_dofs(self):
        cells = [SimpleNamespace(bfnums=np.array([2, 0]), K_el=np.arange(16).reshape(4, 4),
                                 M_el=np.eye(4), f_el=np.array([1, 2, 3, 4])),
                 SimpleNamespace(bfnums=np.array([0, 1]), K_el=np.ones((4, 4)),
                                 M_el=2*np.eye(4), f_el=np.ones(4))]
        rs = SimpleNamespace(basis=[None]*3, elements=cells)
        K, M, f = np.zeros((6, 6)), np.zeros((6, 6)), np.zeros(6)
        for cell, dofs in zip(cells, ([4, 5, 0, 1], [0, 1, 2, 3])):
            K[np.ix_(dofs, dofs)] += cell.K_el
            M[np.ix_(dofs, dofs)] += cell.M_el
            f[dofs] += cell.f_el
        for sparse in (False, True):
            result = Assembler(2).assemble(rs, use_sparse=sparse, assemble_M=True)
            matrix = MatrixAssembler(2).assemble(rs, 'K_el', use_sparse=sparse)
            np.testing.assert_array_equal(result['K'].toarray() if sparse else result['K'], K)
            np.testing.assert_array_equal(result['M'].toarray() if sparse else result['M'], M)
            np.testing.assert_array_equal(matrix.toarray() if sparse else matrix, K)
            np.testing.assert_array_equal(result['f'], f)
        np.testing.assert_array_equal(VectorAssembler(2).assemble(rs, 'f_el'), f)

    def test_void_and_inactive_elements_need_no_local_data(self):
        rs = SimpleNamespace(basis=[None], elements=[SimpleNamespace(void=True), SimpleNamespace(active=False)])
        np.testing.assert_array_equal(MatrixAssembler(1).assemble(rs, 'K_el'), [[0]])
        np.testing.assert_array_equal(VectorAssembler(1).assemble(rs, 'f_el'), [0])
        result = Assembler(1).assemble(rs, assemble_M=True)
        np.testing.assert_array_equal(result['M'], [[0]])

    def test_invalid_indices_and_shapes_are_rejected(self):
        for ids, K in (([1], np.eye(1)), ([-1], np.eye(1)), ([.5], np.eye(1)),
                       ([0, 0], np.eye(2)), ([0], np.eye(2))):
            rs = SimpleNamespace(basis=[None], elements=[SimpleNamespace(bfnums=ids, K_el=K)])
            with self.assertRaises(ValueError):
                MatrixAssembler(1).assemble(rs, 'K_el')


if __name__ == '__main__':
    unittest.main(verbosity=2)
