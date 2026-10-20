"""KL tangential moment work versus effective edge forces and corner forces."""
import unittest
import numpy as np
from numpy.polynomial.legendre import leggauss
from shapely.geometry import LineString, box

from IGAeasy.geometry.get_geometry import polynomial_shell
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.elements.KLshell3p import KLShell
from IGAeasy.constraints.kl_clamped_weak import WeakKLClampedBoundary
from IGAeasy.constraints.kl_mixed_weak import WeakKLMixedBoundary
from IGAeasy.constraints.kl_plate_weak import WeakKLPlateBoundary
from IGAeasy.splines.batch_evaluation import evaluate_derivatives_batch
from IGAeasy.trimming.Brep import BRep
from IGAeasy.trimming.trim_elements import trim


class KLCornerWorkTests(unittest.TestCase):
    amplitude = .01

    def make_patch(self, family='THB', curved=False, trimmed=False, angle=.6):
        self.curvature = (.2, .3) if curved else (0., 0.)
        self.rotation = rotation_matrix(angle, (1., 2., 3.))
        geometry = polynomial_shell(3, 3, curvature=self.curvature, centered=False)
        cps = geometry.control_points@self.rotation.T
        if family == 'THB':
            space = HierarchicalSpace(*geometry.knot_vectors, 3, cps)
        else:
            space = LRSpline(3, *geometry.knot_vectors, cps)
        domain = box(.1, .1, .9, .9) if trimmed else box(0., 0., 1., 1.)
        if trimmed:
            space.BReps = [BRep(1, polygon=domain, outerloop=True)]
            trim(space)
        self.material = LinearElasticMaterial(10920., .3, 1.)
        self.thickness = .1
        assign_gp_and_properties(space, self.material, {'t': self.thickness}, full_quad=True)
        for cell in space.elements:
            if not cell.void:
                KLShell().element_routine(cell)
        coefficients = np.zeros((len(space.basis), 3))
        for basis in space.basis:
            u, v = np.mean(basis.knots_u[1:-1]), np.mean(basis.knots_v[1:-1])
            coefficients[basis.id] = self.rotation[:, 2]*self.amplitude*u*v
        coordinates = list(domain.exterior.coords)
        lines = [LineString([a, b]) for a, b in zip(coordinates[:-1], coordinates[1:])]
        return space, coefficients, lines

    def exact_data(self, u, v, tangent):
        """Independent twist field, moment and uncondensed tangential traction."""
        a, b = self.curvature
        tangents = np.array([[1., 0., 2*a*u], [0., 1., 2*b*v]])
        normal = np.cross(*tangents)
        jacobian = np.linalg.norm(normal)
        normal /= jacobian
        inverse_metric = np.linalg.inv(tangents@tangents.T)
        contra = inverse_metric@tangents
        du, dv = self.amplitude*v, self.amplitude*u
        director = -(du*contra[0]+dv*contra[1])*normal[2]
        # Minus the variation of the second fundamental form, engineering shear.
        curvature = -np.array([2*a*director[2], 2*b*director[2],
                               2*self.amplitude*normal[2]])
        Dm, Db, _ = self.material.compute_C_voigt_curv_2D(inverse_metric, self.thickness)
        moment = Db@curvature
        length = np.linalg.norm(tangent@tangents)
        tau = tangent@tangents/length
        nu = np.array([tangent[1], -tangent[0]])
        edge_moment = jacobian/length*tangents.T@np.array([
            nu[0]*moment[0]+nu[1]*moment[2], nu[0]*moment[2]+nu[1]*moment[1]])
        h = (tau@edge_moment)*normal
        membrane_strain = np.array([tangents[0, 2]*du, tangents[1, 2]*dv,
                                   tangents[0, 2]*dv+tangents[1, 2]*du])
        stress = Dm@membrane_strain
        membrane_force = jacobian/length*tangents.T@np.array([
            nu[0]*stress[0]+nu[1]*stress[2], nu[0]*stress[2]+nu[1]*stress[1]])
        cross_gradient = np.array([[-2*a, 0., 0.], [0., -2*b, 0.]])
        normal_gradient = (cross_gradient-(cross_gradient@normal)[:, None]*normal)/jacobian
        bending_force = jacobian/length*np.array([
            nu[0]*moment[0]+nu[1]*moment[2], nu[0]*moment[2]+nu[1]*moment[1]])@normal_gradient
        raw_tangential_force = membrane_force+bending_force
        displacement = np.array([0., 0., self.amplitude*u*v])
        return (self.rotation@h, self.rotation@displacement, self.rotation@director,
                self.rotation@raw_tangential_force)

    def test_condensed_edge_and_corner_work_matches_uncondensed_work(self):
        for family in ('THB', 'LR'):
            for curved, trimmed in ((False, False), (True, False), (True, True)):
                with self.subTest(family=family, curved=curved, trimmed=trimmed):
                    space, coefficients, lines = self.make_patch(family, curved, trimmed)
                    uncondensed = np.zeros(coefficients.size)
                    effective_edge = np.zeros_like(uncondensed)
                    edge_without_normal_change = np.zeros_like(uncondensed)
                    corners = np.zeros_like(uncondensed)
                    artificial_endpoints = {}
                    for line in lines:
                        tangent = np.diff(np.asarray(line.coords), axis=0)[0]
                        tangent /= np.linalg.norm(tangent)
                        boundary = WeakKLClampedBoundary(line, lambda *x: np.zeros(6))
                        for cell in space.elements:
                            if cell.void:
                                continue
                            samples, _, frames, _ = boundary._boundary_data(cell)
                            if not samples:
                                continue
                            ids = np.array(sorted(b.id for b in cell.supported_b_splines))
                            dofs = (3*ids[:, None]+np.arange(3)).ravel()
                            local = coefficients[ids].ravel()
                            for (measure, trace, flux, _), (position, tau, mu, _) in zip(samples, frames):
                                uncondensed[dofs] += measure*(tau@trace[3:])*(tau@flux[3:]@local)
                                u, v, _ = self.rotation.T@position
                                expected_force = self.exact_data(u, v, tangent)[3]
                                normal = np.cross(mu, tau)
                                force = flux[:3]@local
                                np.testing.assert_allclose(force-normal*(normal@force), expected_force,
                                                           rtol=2e-10, atol=2e-10)

                            # Condense -integral h.dot(v_s) independently on each segment.
                            segment = line.intersection(box(cell.u_min, cell.v_min, cell.u_max, cell.v_max))
                            endpoints = np.asarray(segment.coords)
                            start, end = endpoints[0], endpoints[-1]
                            if (end-start)@tangent < 0:
                                start, end = end, start
                            length = np.linalg.norm(end-start)
                            basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
                            nodes, weights = leggauss(12)
                            for s, weight in zip((nodes+1)/2, weights*length/2):
                                point = start+s*(end-start)
                                # Differentiate h wrt parameter arclength: ds_phys cancels.
                                step = 1e-5*length
                                h_plus = self.exact_data(*(point+step*tangent), tangent)[0]
                                h_minus = self.exact_data(*(point-step*tangent), tangent)[0]
                                derivative = (h_plus-h_minus)/(2*step)
                                N = evaluate_derivatives_batch(basis, [point], cell, max_order=0)[0, 0][0]
                                effective_edge[dofs] += weight*(N[:, None]*derivative).ravel()
                                a, b = self.curvature
                                normal = np.array([-2*a*point[0], -2*b*point[1], 1.])
                                normal = self.rotation@normal/np.linalg.norm(normal)
                                scalar_derivative = normal*(normal@derivative)
                                edge_without_normal_change[dofs] += weight*(N[:, None]*scalar_derivative).ravel()
                            for point, sign in ((start, 1.), (end, -1.)):
                                force = sign*self.exact_data(*point, tangent)[0]
                                N = evaluate_derivatives_batch(basis, [point], cell, max_order=0)[0, 0][0]
                                corners[dofs] += (N[:, None]*force).ravel()
                                key = tuple(np.round(point, 12))
                                artificial_endpoints[key] = artificial_endpoints.get(key, np.zeros(3))+force
                    scale = np.linalg.norm(uncondensed)
                    self.assertGreater(scale, 1e-5)
                    np.testing.assert_allclose(uncondensed, effective_edge+corners, rtol=2e-7, atol=2e-10)
                    self.assertGreater(np.linalg.norm(corners), 1e-3)
                    physical_corners = {tuple(np.round(p, 12)) for line in lines for p in line.coords}
                    for point, force in artificial_endpoints.items():
                        if point not in physical_corners:
                            np.testing.assert_allclose(force, 0., atol=1e-12)
                    if curved:
                        self.assertGreater(np.linalg.norm(effective_edge), 1e-5)
                        # Keeping only d_s(M_nt) misses M_nt*d_s(n) on a shell.
                        self.assertGreater(np.linalg.norm(effective_edge-edge_without_normal_change), 1e-4)
                    else:
                        # Constant twist: no line force, only four alternating corner forces.
                        np.testing.assert_allclose(effective_edge, 0., atol=1e-12)
                        D = self.material.E*self.thickness**3/(12*(1-self.material.nu**2))
                        expected = 2*D*(1-self.material.nu)*self.amplitude
                        for force in artificial_endpoints.values():
                            if np.linalg.norm(force) > 1e-12:
                                self.assertAlmostEqual(np.linalg.norm(force), expected, places=12)

    def test_nonzero_twist_data_needs_no_extra_corner_load_on_essential_edges(self):
        for boundary_type in (WeakKLClampedBoundary, WeakKLMixedBoundary, WeakKLPlateBoundary):
            with self.subTest(boundary=boundary_type.__name__):
                space, coefficients, lines = self.make_patch(trimmed=True)

                def prescribed(x, y, z):
                    u, v, _ = self.rotation.T@np.array([x, y, z])
                    _, displacement, director, _ = self.exact_data(u, v, np.array([1., 0.]))
                    return np.r_[displacement, director]

                residual = np.zeros(coefficients.size)
                for cell in space.elements:
                    if cell.void:
                        continue
                    ids = np.array(sorted(b.id for b in cell.supported_b_splines))
                    dofs = (3*ids[:, None]+np.arange(3)).ravel()
                    local = coefficients[ids].ravel()
                    residual[dofs] += cell.K_el@local
                    for line in lines:
                        matrix, force, _ = boundary_type(line, prescribed).integrate_contribution(cell)
                        residual[dofs] += matrix@local-force
                self.assertLess(np.linalg.norm(residual), 1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
