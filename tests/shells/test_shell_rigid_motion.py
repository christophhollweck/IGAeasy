"""Infinitesimal rigid motions: exact discrete fields and pointwise RM5 jets."""
import unittest
import numpy as np
from IGAeasy.elements import KLshell3p
from IGAeasy.elements import RMshell6p, RMshell5p, RMshell5p_v2, RMshell5p_hier
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.numerical_integration.costum_rule import gauss_lobatto
from tests.shells.test_rm5_kinematics import geometry
from IGAeasy.postprocessing import postprocessing
from scipy.spatial.transform import Rotation


class RigidMotionTests(unittest.TestCase):
    def test_resultants_and_stress_invariants_under_spatial_rotation(self):
        rotation = Rotation.from_rotvec([.4, -.7, .2]).as_matrix()
        for module in (KLshell3p, RMshell6p,
                       RMshell5p, RMshell5p_v2, RMshell5p_hier):
            snapshots = []
            for transform in (np.eye(3), rotation):
                rs = self.space()
                shell = module.KLShell() if hasattr(module, 'KLShell') else module.RMShell()
                for b in rs.basis:
                    x, y, z = b.cp
                    displacement = np.array([.01*x+.003*y, -.004*x+.02*y, .007*x-.002*z])
                    extra = np.array([.002*x, -.003*y, .001*z]) if module is RMshell6p else np.array([.002*x, -.003*y])
                    if shell.dof_per_cp == 3:
                        extra = np.zeros(0)
                    elif module is RMshell6p:
                        extra = transform@extra
                    b.dof_values = np.r_[transform@displacement, extra]
                    b.cp = transform@b.cp
                cell = rs.elements[0]
                if module is RMshell6p:
                    shell.element_routine(cell, gauss_lobatto(3))
                    update = postprocessing.update_stress_strain_RM6dof
                elif shell.dof_per_cp == 3:
                    shell.element_routine(cell)
                    update = postprocessing.update_stress_strain_KL
                else:
                    shell.element_routine(cell)
                    update = postprocessing.update_stress_strain_RM5dof
                update(rs)
                snapshots.append(np.array([np.r_[gp.N, gp.M,
                    [] if gp.Q is None else gp.Q, gp.sigma_vM] for gp in cell.gausspoints]))
            with self.subTest(element=module.__name__):
                np.testing.assert_allclose(snapshots[0], snapshots[1], rtol=1e-10, atol=1e-12)

    def space(self):
        knots = [0., 0., 0., 1., 1., 1.]
        cps = [[u+.3*v, 1.2*v, .2*(i*(i-1)/2)+.15*u*v+.3*(j*(j-1)/2)]
               for j, v in enumerate((0., .5, 1.))
               for i, u in enumerate((0., .5, 1.))]
        rs = HierarchicalSpace(knots, knots, 2, cps)
        assign_gp_and_properties(rs, material_model=LinearElasticMaterial(100., .3, 1.),
                                 properties={'t': .1, 'E': 100., 'rho': 1.}, full_quad=True)
        return rs

    def test_translation_for_every_element_and_rotation_when_representable(self):
        modules = (KLshell3p, RMshell6p,
                   RMshell5p, RMshell5p_v2, RMshell5p_hier)
        for module in modules:
            rs = self.space()
            shell = module.KLShell() if hasattr(module, 'KLShell') else module.RMShell()
            cell = rs.elements[0]
            if module is RMshell6p:
                shell.element_routine(cell, gauss_lobatto(3))
            else:
                shell.element_routine(cell)
            basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
            translation = np.tile(np.r_[[.2, -.4, .7], np.zeros(shell.dof_per_cp-3)], len(basis))
            self.assertLess(abs(translation@cell.K_el@translation), 1e-10)
            if module in (RMshell5p, RMshell5p_v2):
                continue  # Curved director coefficients need not lie in the spline space.
            for axis in np.eye(3):
                rotation = np.concatenate([np.r_[np.cross(axis, b.cp),
                     axis if module is RMshell6p else np.zeros(shell.dof_per_cp-3)] for b in basis])
                for gp in cell.gausspoints:
                    matrices = gp.Bmatrices if module is RMshell6p else gp.Bmatrices[0]
                    for matrix in matrices:
                        np.testing.assert_allclose(matrix@rotation, 0., atol=2e-12)
                if module is not RMshell6p:
                    self.assertLess(abs(rotation@cell.K_el@rotation), 1e-10)

    def test_RM6_stabilization_has_zero_rigid_rotation_energy(self):
        rs = self.space()
        cell = rs.elements[0]
        RMshell6p.RMShell().element_routine(cell, gauss_lobatto(3))
        basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
        for axis in np.eye(3):
            values = np.concatenate([np.r_[np.cross(axis, b.cp), axis] for b in basis])
            self.assertLess(abs(values@cell.K_el@values), 1e-10)

    def test_RM6_total_energy_is_invariant_under_spatial_rotation(self):
        energies = []
        for transform in (np.eye(3), Rotation.from_rotvec([.4, -.7, .2]).as_matrix()):
            rs = self.space()
            vectors = []
            for b in rs.basis:
                x, y, z = b.cp
                vectors.append(np.r_[transform@np.array([.01*x, .02*y, -.03*z]),
                                     transform@np.array([.002*x, -.003*y, .004*z])])
                b.cp = transform@b.cp
            cell = rs.elements[0]
            RMshell6p.RMShell().element_routine(cell, gauss_lobatto(3))
            values = np.concatenate(vectors)
            energies.append(values@cell.K_el@values)
        np.testing.assert_allclose(energies[0], energies[1], rtol=1e-11, atol=1e-13)

    def test_exact_curved_RM5_rigid_rotation_director_fields(self):
        u, v = .37, .61
        data = geometry(u, v)
        for module in (RMshell5p, RMshell5p_v2):
            for axis in np.eye(3):
                def coefficients(x, y):
                    d = geometry(x, y)
                    director = np.cross(axis, d[2])
                    covariant = np.array([d[0]@director, d[1]@director])
                    if module is RMshell5p:
                        return covariant
                    metric = np.array([[d[0]@d[0], d[0]@d[1]], [d[1]@d[0], d[1]@d[1]]])
                    return np.linalg.solve(metric, covariant)
                h = 1e-5
                c = coefficients(u, v)
                dc1 = (coefficients(u+h, v)-coefficients(u-h, v))/(2*h)
                dc2 = (coefficients(u, v+h)-coefficients(u, v-h))/(2*h)
                translation_part = module.compute_B_matrix(0., 1., 0., *data)
                second_part = module.compute_B_matrix(0., 0., 1., *data)
                strains = [B[:, :3]@np.cross(axis, data[0])+D[:, :3]@np.cross(axis, data[1])
                           for B, D in zip(translation_part, second_part)]
                for component in range(2):
                    matrices = module.compute_B_matrix(c[component], dc1[component], dc2[component], *data)
                    for strain, matrix in zip(strains, matrices):
                        strain += matrix[:, 3+component]
                for strain in strains:
                    np.testing.assert_allclose(strain, 0., atol=2e-9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
