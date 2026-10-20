"""Penalty and symmetric/penalty-free Nitsche conditions for RM shell elements."""
import numpy as np
from IGAeasy.constraints.weak_boundary import WeakBoundary, coefficient_trace, surface_energy_factor
from IGAeasy.constraints.line_helpers import get_boundary_quadrature
from IGAeasy.differential_geometry.derivatives import get_local_basis_and_variations
from IGAeasy.differential_geometry.derivatives import get_full_geometric_data
from IGAeasy.elements.RMshell6p import RMShell
from IGAeasy.splines.batch_evaluation import evaluate_derivatives_batch


def trace_projector(components, names, directions=None):
    """Select orthogonal physical directions on a six-component scaled trace."""
    if (not components or len(set(components)) != len(components)
            or any(name not in names for name in components)):
        raise ValueError('components must be distinct physical component names')
    if directions is None:
        return np.eye(6)[[names.index(name) for name in components]]
    if tuple(components) != names:
        raise ValueError('Use either components or directions')
    rows = np.asarray(directions, dtype=float)
    if (rows.ndim != 2 or rows.shape[1] != 6 or not 1 <= rows.shape[0] <= 6
            or not np.isfinite(rows).all() or np.any(np.linalg.norm(rows, axis=1) == 0)):
        raise ValueError('directions must be a finite nonzero row matrix with six columns')
    rows = rows/np.linalg.norm(rows, axis=1)[:, None]
    if not np.allclose(rows@rows.T, np.eye(len(rows)), rtol=0., atol=1e-12):
        raise ValueError('Boundary directions must be mutually orthogonal')
    return rows


def select_director_data(projector, value, normal):
    """Check a tangential completion of selected data; free values stay free."""
    point_map = np.vstack((projector, np.r_[np.zeros(3), normal]))
    prescribed = projector@value
    data = np.r_[prescribed, 0.]
    attainable = point_map@np.linalg.lstsq(point_map, data, rcond=None)[0]
    if not np.allclose(attainable, data, rtol=1e-10, atol=1e-12):
        raise ValueError('Selected director data are not attainable in the tangent plane')
    return prescribed


class WeakRM6Boundary(WeakBoundary):
    """Prescribe Cartesian translations and rotations on an oriented polyline.

    prescribed(x,y,z) returns (ux,uy,uz,rx,ry,rz). Material is to the left
    of the parametric line (outer CCW, holes CW). Geometry may be curved;
    thickness and material must be constant per cell. Use the original bulk
    K_el for every boundary bound, before adding any boundary contributions.
    The thickness rule must match the rule used in the bulk RM6 element.
    Rotations are scaled by thickness in the trace norm to match displacement
    units. The artificial drilling flux is included in the consistency terms.
    components selects Cartesian displacement/rotation traces by name.
    directions optionally supplies orthogonal rows acting on (u,t*omega).
    method='penalty_free' selects experimental nonsymmetric Nitsche with
    zero penalty and opposite adjoint sign; it does not use the energy bound.
    Its stability must be checked for the chosen mesh/support configuration.
    """
    def __init__(self, line, prescribed, gauss_t, method='nitsche',
                 penalty=None, safety_factor=4.,
                 components=('ux', 'uy', 'uz', 'rx', 'ry', 'rz'), directions=None):
        names = ('ux', 'uy', 'uz', 'rx', 'ry', 'rz')
        self.projector = trace_projector(components, names, directions)
        super().__init__(line, prescribed, method, penalty, safety_factor)
        nodes, weights = (np.asarray(part, dtype=float) for part in gauss_t)
        if (nodes.ndim != 1 or weights.shape != nodes.shape or not len(nodes)
                or not np.isfinite(nodes).all() or not np.isfinite(weights).all()
                or np.any(abs(nodes) > 1) or np.any(weights <= 0)):
            raise ValueError('Invalid thickness quadrature')
        self.gauss_t = nodes, weights

    def integrate_contribution(self, cell):
        basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
        size = 6*len(basis)
        points = get_boundary_quadrature(self.line, cell)
        if not points:
            return np.zeros((size, size)), np.zeros(size), 0.
        gp = cell.gausspoints[0]
        t = gp.t
        if not np.isfinite(t) or t <= 0:
            raise ValueError('Shell thickness must be positive')
        if any(g.t != t or g.material_model is not gp.material_model for g in cell.gausspoints):
            raise ValueError('Weak RM6 boundaries require constant cell material and thickness')
        trace_scale = np.array([1., 1., 1., t, t, t])
        cps = np.array([b.cp for b in basis])
        samples = []
        values = evaluate_derivatives_batch(
            basis, [(point.x, point.y) for point, _, _ in points], cell, max_order=2)
        for q, (point, weight, tangent_param) in enumerate(points):
            data = {order: value[q] for order, value in values.items()}
            flux, line_jacobian = self._point_flux(cell, point, tangent_param, data, cps)
            N = data[0, 0]
            trace = coefficient_trace(N, 6)
            # Scale the trace and its conjugate flux inversely to preserve work.
            trace = trace_scale[:, None]*trace
            flux = flux/trace_scale[:, None]
            prescribed = np.asarray(self.prescribed(*(N@cps)), dtype=float)
            if prescribed.shape != (6,) or not np.isfinite(prescribed).all():
                raise ValueError('Prescribed RM6 field must return six finite components')
            prescribed = trace_scale*prescribed
            trace = self.projector@trace
            flux = self.projector@flux
            prescribed = self.projector@prescribed
            samples.append((weight*line_jacobian, trace, flux, prescribed))
        energy_factor = self._bulk_energy_factor(cell) if self.penalty is None else None
        return self._assemble(cell, samples, size, energy_factor=energy_factor)

    def _point_flux(self, cell, point, tangent_param, data, cps):
        """Physical force/rotation flux, integrated through the thickness."""
        gp = cell.gausspoints[0]
        t = gp.t
        size = 6*len(cps)
        shell = RMShell()
        outward_param = np.array([tangent_param[1], -tangent_param[0]])
        a1, a2, normal, _, _, _, _, dn_du, dn_dv = get_local_basis_and_variations(
            cell, point.x, point.y, geometry_derivatives=(
                (data[1, 0] @ cps, data[0, 1] @ cps),
                (data[2, 0] @ cps, data[0, 2] @ cps, data[1, 1] @ cps)))
        geometry = np.column_stack((a1, a2))
        metric = geometry.T@geometry
        tangent = geometry@tangent_param
        line_jacobian = np.linalg.norm(tangent)
        N = data[0, 0]
        derivatives = np.vstack((data[1, 0], data[0, 1]))
        mapped = geometry@np.linalg.solve(metric, np.column_stack((derivatives, outward_param)))
        gradient, drill_gradient = mapped[:, :-1], mapped[:, -1]
        flux = np.zeros((6, size))
        drill = shell.compute_B_drill_matrix(len(cps), 6, gradient[0], gradient[1],
                                             N, dNdz=gradient[2], normal=normal)
        drill_derivative = shell.compute_B_drill_matrix(1, 6,
            np.array([drill_gradient[0]]), np.array([drill_gradient[1]]), np.zeros(1),
            dNdz=np.array([drill_gradient[2]]), normal=normal)
        material = gp.material_model.get_C_voigt_cart_3D()
        for zeta, wt in zip(*self.gauss_t):
            J, _, detJ, Jinv = shell.compute_jacobian(derivatives, *cps.T,
                dn_du, dn_dv, normal, t, zeta)
            if detJ <= 0:
                raise ValueError('Shell thickness mapping must have a positive Jacobian')
            thickness_data = shell.compute_basis_functions_and_normals(derivatives, Jinv,
                dn_du, dn_dv, len(cps))
            normal_derivatives = np.array(thickness_data[3:]).reshape(3, 3).T.ravel()
            B = shell.compute_B_matrix(len(cps), 6, *thickness_data[:3], N,
                *normal_derivatives, *Jinv[:, 2], *normal, zeta, t)
            local = shell.compute_local_basis(*J[0], normal)
            rotation = shell.get_rotationsmatrix(*local)
            C = rotation@material@rotation.T
            # Derivative of B with respect to the outward parametric gradient
            # of each of the six unknown fields. N=0 removes zero-order terms.
            derivative_gradient = Jinv[:, :2]@outward_param
            derivative_B = shell.compute_B_matrix(1, 6,
                *(np.array([value]) for value in derivative_gradient), np.zeros(1),
                *([0.]*9), *Jinv[:, 2], *normal, zeta, t)
            flux += wt*detJ/line_jacobian*(derivative_B.T@C@B
                + .05*gp.material_model.E*drill_derivative.T@drill)
        return flux, line_jacobian

    def _bulk_energy_factor(self, cell):
        """Reconstruct the quadrature strain factor including drilling energy."""
        factors = []
        for gp in cell.gausspoints:
            if (not hasattr(gp, 'volume_weights') or not hasattr(gp, 'Bdrill')
                    or not np.array_equal(gp.zeta, self.gauss_t[0])
                    or not np.array_equal(gp.weight_outerplane, self.gauss_t[1])):
                raise ValueError('Recompute RM6 bulk data with the boundary thickness rule')
            C = gp.material_model.get_C_voigt_cart_3D()
            eigenvalues, eigenvectors = np.linalg.eigh((C+C.T)/2)
            tolerance = len(C)*np.finfo(float).eps*max(abs(eigenvalues).max(), 1e-300)
            if eigenvalues.min() < -tolerance:
                raise ValueError('RM6 material energy must be positive-semidefinite')
            positive = eigenvalues > tolerance
            root = np.sqrt(eigenvalues[positive])[:, None]*eigenvectors[:, positive].T
            for weight, rotation, strain in zip(gp.volume_weights, gp.rotationmatrices, gp.Bmatrices):
                if not np.isfinite(weight) or weight <= 0:
                    raise ValueError('RM6 bulk integration requires positive volume weights')
                factors.append(np.sqrt(weight)*root@rotation.T@strain)
                factors.append(np.sqrt(weight*.05*gp.material_model.E)*gp.Bdrill)
        return np.vstack(factors)


class WeakRM5Boundary(WeakBoundary):
    """Weak physical displacement/director conditions for RM5 and RM5_v2.

    prescribed(x,y,z) returns (ux,uy,uz,dx,dy,dz), where d is the infinitesimal
    tangential director change, not a rotation vector. components uses those
    six names. directions acts on the scaled trace (u,t*d). Geometry may be
    curved. RM5_hier is deliberately excluded: its trace contains gradients.
    natural(x,y,z,conormal) optionally supplies physical force and director
    moment on the complementary trace, as six components per unit edge length.
    directions may also be callable(x,y,z), with a constant number of rows.
    """
    def __init__(self, line, prescribed, formulation='RM5', method='nitsche',
                 penalty=None, safety_factor=4.,
                 components=('ux', 'uy', 'uz', 'dx', 'dy', 'dz'), directions=None, natural=None):
        if formulation not in ('RM5', 'RM5_v2'):
            raise ValueError('formulation must be RM5 or RM5_v2')
        names = ('ux', 'uy', 'uz', 'dx', 'dy', 'dz')
        if any(name not in names for name in components):
            raise ValueError('Unknown physical displacement/director component')
        super().__init__(line, prescribed, method, penalty, safety_factor)
        self.directions = directions
        self.components = tuple(components)
        self.names = names
        if callable(directions):
            if self.components != names:
                raise ValueError('Use either components or directions')
        else:
            self.projector = trace_projector(components, names, directions)
        self.formulation = formulation
        self.natural = natural

    def integrate_contribution(self, cell):
        basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
        size = 5*len(basis)
        points = get_boundary_quadrature(self.line, cell)
        if not points:
            return np.zeros((size, size)), np.zeros(size), 0.
        gp = cell.gausspoints[0]
        t = gp.t
        if not np.isfinite(t) or t <= 0:
            raise ValueError('Shell thickness must be positive')
        if any(g.t != t or g.material_model is not gp.material_model for g in cell.gausspoints):
            raise ValueError('Weak RM5 boundaries require constant cell material and thickness')
        cps = np.array([b.cp for b in basis])
        samples = []
        natural_load = np.zeros(size)
        values = evaluate_derivatives_batch(
            basis, [(point.x, point.y) for point, _, _ in points], cell, max_order=2)
        for q, (point, weight, tangent_param) in enumerate(points):
            data = {order: value[q] for order, value in values.items()}
            trace, flux, line_jacobian, normal = self._point_operators(cell, point, tangent_param, data, cps)
            N = data[0, 0]
            projector = (trace_projector(self.components, self.names, self.directions(*(N@cps)))
                         if callable(self.directions) else self.projector)
            if samples and len(projector) != samples[0][1].shape[0]:
                raise ValueError('The number of selected directions must remain constant along the boundary')
            if self.natural is not None:
                tangent = (tangent_param[0]*data[1, 0]+tangent_param[1]*data[0, 1])@cps/line_jacobian
                natural = np.asarray(self.natural(*(N@cps), np.cross(tangent, normal)), dtype=float)
                if natural.shape != (6,) or not np.isfinite(natural).all():
                    raise ValueError('natural must return six finite physical force/moment values')
                natural = natural/np.array([1., 1., 1., t, t, t])
                natural_load += weight*line_jacobian*trace.T@(natural-projector.T@(projector@natural))
            trace = projector@trace
            flux = projector@flux
            value = np.asarray(self.prescribed(*(N@cps)), dtype=float)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError('Prescribed RM5 field must return six finite physical components')
            scaled_value = value*np.array([1., 1., 1., t, t, t])
            prescribed = select_director_data(projector, scaled_value, normal)
            measure = weight*line_jacobian
            samples.append((measure, trace, flux, prescribed))
        energy_factor = surface_energy_factor(cell) if self.penalty is None else None
        matrix, rhs, gamma = self._assemble(cell, samples, size, energy_factor=energy_factor)
        return matrix, rhs+natural_load, gamma

    def _point_operators(self, cell, point, tangent_param, data, cps):
        """Physical displacement/director trace and its work-conjugate flux."""
        from IGAeasy.elements import RMshell5p, RMshell5p_v2
        module = RMshell5p if self.formulation == 'RM5' else RMshell5p_v2
        gp = cell.gausspoints[0]
        t = gp.t
        outward = np.array([tangent_param[1], -tangent_param[0]])
        cov, contra, variations, metric, _, _ = get_full_geometric_data(
            cell, point.x, point.y, geometry_derivatives=(
                (data[1, 0] @ cps, data[0, 1] @ cps),
                (data[2, 0] @ cps, data[0, 2] @ cps, data[1, 1] @ cps)))
        a1, a2, normal = cov[1], cov[2], cov[3]
        line_jacobian = np.linalg.norm(tangent_param[0]*a1+tangent_param[1]*a2)
        area_jacobian = np.linalg.norm(np.cross(a1, a2))
        arguments = (a1, a2, normal, contra[1], contra[2], variations[1][1],
            variations[2][2], variations[1][2], variations[2][1], variations[3][1], variations[3][2])
        inverse_metric = np.linalg.inv(metric)
        if self.formulation == 'RM5':
            Cm, Cb, Ds = gp.material_model.compute_C_tensors_curv_2D(inverse_metric, t)
            D = gp.material_model.tensor4_to_voigt2D(Cm), gp.material_model.tensor4_to_voigt2D(Cb), Ds
        else:
            D = gp.material_model.compute_C_voigt_curv_2D(inverse_metric, t)
        N = data[0, 0]
        gradients = np.column_stack((data[1, 0], data[0, 1]))
        blocks = [module.compute_B_matrix(value, *gradient, *arguments)
                  for value, gradient in zip(N, gradients)]
        B = [np.hstack([block[i] for block in blocks]) for i in range(3)]
        derivative_B = module.compute_B_matrix(0., *outward, *arguments)
        flux_coordinates = area_jacobian/line_jacobian*sum(
            derivative.T@material@strain for derivative, material, strain in zip(derivative_B, D, B))
        vectors = contra if self.formulation == 'RM5' else cov
        physical_map = np.zeros((6, 5))
        physical_map[:3, :3] = np.eye(3)
        physical_map[3:, 3:] = t*np.column_stack((vectors[1], vectors[2]))
        # Dual map preserves virtual work despite the redundant 3D director trace.
        flux = physical_map@np.linalg.solve(physical_map.T@physical_map, flux_coordinates)
        trace = physical_map@coefficient_trace(N, 5)
        return trace, flux, line_jacobian, normal
