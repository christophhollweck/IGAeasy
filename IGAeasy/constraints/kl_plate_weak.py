"""Weak deflection and normal-slope constraints for affine KL plates.

Tangential moment work is retained as a tangential derivative of the
deflection trace. This is equivalent to effective shear plus corner forces,
including contributions at element intersections along a trimmed boundary.
It is not a component mask of the full-director clamped formulation.
"""
import numpy as np
from IGAeasy.constraints.weak_boundary import WeakBoundary
from IGAeasy.constraints.kl_clamped_weak import check_kl_continuity
from IGAeasy.constraints.kl_mixed_weak import lift_boundary_work
from IGAeasy.constraints.line_helpers import get_boundary_quadrature
from IGAeasy.splines.batch_evaluation import evaluate_derivatives_batch


class WeakKLPlateBoundary(WeakBoundary):
    """Constrain w=n dot u and/or theta_n=-mu dot grad(w) on a material line.

    prescribed(x,y,z) returns six physical (u,delta_director) components.
    For nonzero w, its tangential derivative must match the supplied director.
    A free normal rotation has its natural moment condition, zero by default.
    In-plane displacements are not constrained. Geometry must be affine and
    the bulk basis C1. Lines follow the usual material-left orientation.
    normal_moment(x,y,z,conormal) optionally applies the natural bending moment
    when the normal rotation is free, with the bulk moment/director sign.
    """
    def __init__(self, line, prescribed, displacement=True, normal_rotation=False,
                 method='nitsche', penalty=None, safety_factor=4., normal_moment=None):
        if method not in ('penalty', 'nitsche'):
            raise ValueError('KL plate boundary supports penalty or symmetric nitsche only')
        if not displacement and not normal_rotation:
            raise ValueError('Select deflection, normal rotation, or both')
        self.displacement, self.normal_rotation = displacement, normal_rotation
        if normal_rotation and normal_moment is not None:
            raise ValueError('Choose prescribed normal rotation or applied normal moment')
        self.normal_moment = normal_moment
        super().__init__(line, prescribed, method=method,
                         penalty=penalty, safety_factor=safety_factor)

    def integrate_contribution(self, cell):
        basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
        size = 3*len(basis)
        points = get_boundary_quadrature(self.line, cell, quadrature_order=2*cell.p+1)
        if not points:
            return np.zeros((size, size)), np.zeros(size), 0.
        check_kl_continuity(cell)
        gp = cell.gausspoints[0]
        cps = np.asarray([b.cp for b in basis])
        geometry = evaluate_derivatives_batch(
            basis, [(g.xi, g.eta) for g in cell.gausspoints], cell, max_order=2)
        tangent_samples = np.stack((geometry[1, 0]@cps, geometry[0, 1]@cps), axis=1)
        second_samples = np.stack((geometry[2, 0]@cps, geometry[0, 2]@cps, geometry[1, 1]@cps), axis=1)
        tangents = tangent_samples[0]
        if (not np.allclose(tangent_samples, tangents, rtol=1e-11, atol=1e-12)
                or np.linalg.norm(second_samples, axis=2).max() > 1e-11*np.linalg.norm(tangents)):
            raise ValueError('WeakKLPlateBoundary requires affine geometry')
        inverse_metric = np.linalg.inv(tangents@tangents.T)
        contra = inverse_metric@tangents
        normal = np.cross(*tangents)
        normal /= np.linalg.norm(normal)
        t = gp.t
        if not np.isfinite(t) or t <= 0:
            raise ValueError('Plate thickness must be positive')
        for point in cell.gausspoints:
            if point.t != t or point.material_model is not gp.material_model:
                raise ValueError('Plate boundary requires constant cell material and thickness')
        # tensor4_to_voigt2D matches the constitutive path in KLShell.
        _, Cb, _ = gp.material_model.compute_C_tensors_curv_2D(inverse_metric, t)
        Db = gp.material_model.tensor4_to_voigt2D(Cb)
        values = evaluate_derivatives_batch(
            basis, [(point.x, point.y) for point, _, _ in points], cell, max_order=3)
        scalar_to_vector = np.kron(np.eye(len(basis)), normal[None, :])
        trace_rows, prescribed_rows = [], []
        consistency = np.zeros((size, size))
        prescribed_work = np.zeros(size)
        natural_load = np.zeros(size)
        jacobian = np.linalg.norm(np.cross(*tangents))
        for q, (point, weight, tangent_param) in enumerate(points):
            direction = tangent_param@tangents
            length = np.linalg.norm(direction)
            tangent = direction/length
            conormal = np.cross(tangent, normal)
            nu = np.array([tangent_param[1], -tangent_param[0]])
            gradient = np.column_stack((values[1, 0][q], values[0, 1][q]))@contra
            bending = -np.vstack((values[2, 0][q], values[0, 2][q], 2*values[1, 1][q]))
            moment = Db@bending
            moment_u = -Db@np.vstack((values[3, 0][q], values[1, 2][q], 2*values[2, 1][q]))
            moment_v = -Db@np.vstack((values[2, 1][q], values[0, 3][q], 2*values[1, 2][q]))
            force = jacobian/length*(nu[0]*(moment_u[0]+moment_v[2])
                                     +nu[1]*(moment_u[2]+moment_v[1]))
            director_moment = jacobian/length*tangents.T@np.vstack((
                nu[0]*moment[0]+nu[1]*moment[2], nu[0]*moment[2]+nu[1]*moment[1]))
            moment_t = tangent@director_moment
            moment_n = conormal@director_moment
            theta_n = -gradient@conormal
            value = np.asarray(self.prescribed(*(values[0, 0][q]@cps)), dtype=float)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError('prescribed must return six finite physical components')
            traces, data = [], []
            if self.displacement:
                N = values[0, 0][q]
                theta_t = -gradient@tangent
                # Both sides act on scalar deflection coefficients here.
                work = np.outer(N, force)+np.outer(theta_t, moment_t)
                consistency += weight*length*scalar_to_vector.T@work@scalar_to_vector
                prescribed_work -= weight*length*scalar_to_vector.T@(
                    force*(normal@value[:3])+moment_t*(tangent@value[3:]))
                traces.append(N)
                data.append(normal@value[:3])
            if self.normal_rotation:
                consistency += weight*length*scalar_to_vector.T@np.outer(theta_n, moment_n)@scalar_to_vector
                prescribed_work -= weight*length*scalar_to_vector.T@moment_n*(conormal@value[3:])
                traces.append(t*theta_n)
                data.append(t*(conormal@value[3:]))
            if self.normal_moment is not None:
                moment_value = self.normal_moment(*(values[0, 0][q]@cps), conormal)
                if np.ndim(moment_value) != 0 or not np.isfinite(moment_value):
                    raise ValueError('Applied normal moment must return one finite scalar')
                natural_load += weight*length*scalar_to_vector.T@theta_n*moment_value
            root_weight = np.sqrt(weight*length)
            trace_rows.append(root_weight*np.asarray(traces)@scalar_to_vector)
            prescribed_rows.extend(root_weight*np.asarray(data))
        trace = np.vstack(trace_rows)
        prescribed = np.asarray(prescribed_rows)
        # Lift the complete boundary work, including tangential/corner work,
        # into the discrete deflection/slope trace space.
        flux = lift_boundary_work(trace, consistency, cell.K_el)
        factors = []
        for point in cell.gausspoints:
            measure = point.weight_inplane*np.linalg.norm(np.cross(*point.surface_tangents))
            for material, strain in zip(point.D, point.Bmatrices[0]):
                factors.append(np.sqrt(measure)*np.linalg.cholesky(material).T@strain)
        Kbc, fbc, gamma = self._assemble(
            cell, [(1., trace, flux, prescribed)], size, energy_factor=np.vstack(factors))
        if self.method == 'nitsche':
            fbc += flux.T@prescribed+prescribed_work
        return Kbc, fbc+natural_load, gamma
