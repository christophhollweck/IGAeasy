"""Rotated support, nonzero ties and multiplier equilibrium."""
import unittest
import numpy as np
from IGAeasy.constraints.linear_constraints import LinearConstraints, solve_linear_constraints


class LinearConstraintTests(unittest.TestCase):
    def test_rotated_membrane_patch_with_directional_boundary_values(self):
        from IGAeasy.geometry import get_geometry
        from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
        from IGAeasy.material.linear_elastic import LinearElasticMaterial
        from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
        from IGAeasy.elements.membrane2p import Membrane
        from IGAeasy.assembler import MatrixAssembler
        from IGAeasy.constraints.boundary_selection import select_patch_boundary
        energies = []
        for angle in (0., .7):
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, 2, distortion=0, n_waves=0)
            reference = geo.control_points.copy()
            cps = reference.copy()
            cps[:, :2] = cps[:, :2]@rotation.T
            rs = HierarchicalSpace(*geo.knot_vectors, 2, cps)
            assign_gp_and_properties(rs, material_model=LinearElasticMaterial(100., .3, 1.),
                                     properties={'t': .2, 'rho': 1.}, full_quad=True)
            for cell in rs.elements:
                Membrane().element_routine(cell)
            K = MatrixAssembler(dof_per_cp=2).assemble(rs, 'K_el', use_sparse=True)
            conditions = LinearConstraints.for_control_points(('x', 'y'), len(rs.basis))
            boundary = set().union(*(select_patch_boundary(rs, edge)
                                    for edge in ('umin', 'umax', 'vmin', 'vmax')))
            expected = np.zeros(2*len(rs.basis))
            for b in rs.basis:
                original = rotation.T@b.cp[:2]
                local_value = np.array([.01*original[0], -.003*original[1]])
                expected[2*b.id:2*b.id+2] = rotation@local_value
                if b.id in boundary:
                    for index in range(2):
                        conditions.direction([b.id], rotation[:, index], local_value[index],
                                             components=('x', 'y'))
            u, multipliers = solve_linear_constraints(K, np.zeros(K.shape[0]), conditions)
            np.testing.assert_allclose(u, expected, rtol=1e-10, atol=1e-12)
            C, d = conditions.matrices()
            np.testing.assert_allclose(C@u, d, atol=1e-12)
            np.testing.assert_allclose(K@u+C.T@multipliers, 0., atol=1e-12)
            energies.append(u@K@u)
        self.assertAlmostEqual(energies[0], energies[1], places=12)

    def test_named_components_direction_and_tie(self):
        conditions = LinearConstraints.for_control_points(('x', 'y'), 2)
        conditions.prescribe([0], ('x', 'y'))
        conditions.tie(0, 1, 'x', offset=2.)
        conditions.direction([1], [.6, .8], value=2., components=('x', 'y'))
        u, _ = solve_linear_constraints(np.eye(4), np.zeros(4), conditions)
        np.testing.assert_allclose(u, [0., 0., 2., 1.])
        before = conditions.matrices()[0].shape
        with self.assertRaises(ValueError):
            conditions.prescribe([0, 2], 'x')
        self.assertEqual(conditions.matrices()[0].shape, before)
        with self.assertRaises(ValueError):
            conditions.direction([1], [0., 0.], components=('x', 'y'))

    def test_rotated_support_and_reaction(self):
        normal = np.array([.6, .8])
        conditions = LinearConstraints(2).add(dict(enumerate(normal)))
        force = np.array([1., 2.])
        u, multipliers = solve_linear_constraints(np.eye(2), force, conditions)
        np.testing.assert_allclose(u, force-normal*(normal@force), atol=1e-14)
        C, d = conditions.matrices()
        np.testing.assert_allclose(C@u, d, atol=1e-14)
        np.testing.assert_allclose(u+C.T@multipliers, force)

    def test_nonzero_tie_and_prescribed_value(self):
        conditions = LinearConstraints(2).add({0: 1.}, 3.).add({1: 1., 0: -1.}, 2.)
        u, _ = solve_linear_constraints(np.eye(2), np.zeros(2), conditions)
        np.testing.assert_allclose(u, [3., 5.])

    def test_redundant_equations_and_invalid_input(self):
        conditions = LinearConstraints(2).add({0: 1.}).add({0: 1.})
        with self.assertRaisesRegex(ValueError, 'Singular'):
            solve_linear_constraints(np.eye(2), np.zeros(2), conditions)
        with self.assertRaises(ValueError):
            conditions.add({2: 1.})
        with self.assertRaises(ValueError):
            conditions.add({0: 0.})


if __name__ == '__main__':
    unittest.main(verbosity=2)
