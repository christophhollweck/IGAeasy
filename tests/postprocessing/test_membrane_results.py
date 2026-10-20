"""Membrane patch tests in physical and rotated local coordinates."""
import unittest
import numpy as np
from IGAeasy.elements.membrane2p import Membrane
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.postprocessing import postprocessing


class MembraneResultTests(unittest.TestCase):
    def test_affine_patch_stress_energy_and_repeatability(self):
        strain = np.array([[.01, .007], [.007, -.004]])
        material = LinearElasticMaterial(210., .3, 1.)
        t = .2
        C = material.get_C_voigt_cart_2D()
        engineering = np.array([strain[0, 0], strain[1, 1], 2*strain[0, 1]])
        cart_stress = C@engineering
        stress_tensor = np.array([[cart_stress[0], cart_stress[2]],
                                  [cart_stress[2], cart_stress[1]]])
        knots = [0., 0., 0., 1., 1., 1.]
        for kind in ('THB', 'LR'):
            for angle in (0., .6):
                rot = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
                mapping = rot@np.array([[2., .4], [0., 3.]])
                cps = [[*(mapping@np.array([u, v])), 0.]
                       for v in (0., .5, 1.) for u in (0., .5, 1.)]
                rs = (HierarchicalSpace(knots, knots, 2, cps) if kind == 'THB'
                      else LRSpline(2, knots, knots, cps))
                assign_gp_and_properties(rs, material_model=material,
                                         properties={'t': t, 'rho': 1.}, full_quad=True)
                for b in rs.basis:
                    b.dof_values = strain@b.cp[:2]
                for cell in rs.elements:
                    Membrane().element_routine(cell)
                    Membrane().element_routine(cell)
                    values = np.concatenate([b.dof_values for b in sorted(cell.supported_b_splines, key=lambda b: b.id)])
                    expected_energy_twice = t*np.linalg.det(mapping)*(engineering@C@engineering)
                    self.assertAlmostEqual(values@cell.K_el@values, expected_energy_twice, places=12)
                for _ in range(2):
                    postprocessing.update_stress_strain_membrane2D(rs)
                    postprocessing.interpolate_stress_membrane2D(rs)
                for gp in rs.elements[0].gausspoints:
                    e1, e2, _ = gp.local_basis[0]
                    frame = np.array([e1[:2], e2[:2]])
                    local = frame@stress_tensor@frame.T
                    np.testing.assert_allclose(gp.N, t*np.array([local[0, 0], local[1, 1], local[0, 1]]), atol=1e-13)
                    self.assertEqual(len(gp.stress_local), 1)
                    self.assertAlmostEqual(gp.sigma_vM, np.sqrt(cart_stress[0]**2+cart_stress[1]**2
                                           -cart_stress[0]*cart_stress[1]+3*cart_stress[2]**2), places=12)
                np.testing.assert_allclose(postprocessing.eval_stress_membrane2D(rs, .4, .6, 'N'),
                                           [rs.elements[0].gausspoints[0].N], atol=1e-13)


if __name__ == '__main__':
    unittest.main(verbosity=2)
