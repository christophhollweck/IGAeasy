"""Full-gradient symmetric Nitsche for completely clamped KL boundaries."""
import numpy as np
from IGAeasy.constraints.weak_boundary import WeakBoundary, surface_energy_factor
from IGAeasy.constraints.line_helpers import get_boundary_quadrature
from IGAeasy.differential_geometry.derivatives import get_full_geometric_data
from IGAeasy.differential_geometry.surface_jets import surface_jet
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.splines.batch_evaluation import evaluate_derivatives_batch


def check_kl_continuity(cell):
    """Require C1 bulk continuity, including interior lines away from the boundary."""
    if cell.p < 2:
        raise ValueError('KL requires at least quadratic C1 bases')
    mesh = getattr(cell, '_mesh_elements', [cell])
    ends = ((min(c.u_min for c in mesh), max(c.u_max for c in mesh)),
            (min(c.v_min for c in mesh), max(c.v_max for c in mesh)))
    basis = {id(b): b for c in mesh if not getattr(c, 'void', False)
             for b in c.supported_b_splines}
    for b in basis.values():
        for knots, boundary in zip((b.knots_u, b.knots_v), ends):
            values, counts = np.unique(knots, return_counts=True)
            if any(boundary[0] < value < boundary[1] and count > b.p-1
                   for value, count in zip(values, counts)):
                raise ValueError('KL bulk discretization must be C1 across interior knot lines')


class WeakKLClampedBoundary(WeakBoundary):
    """Prescribe complete physical displacement and director change.

    C1 polynomial geometry/bases and constant cell isotropic material/thickness
    are supported. Callback returns (ux,uy,uz,dx,dy,dz); director data must be
    compatible with the prescribed displacement trace. All six physical traces
    are used (the director has only two tangential components). No component
    selection or general mixed KL boundary conditions are provided.

    The uncondensed second-order Green identity retains the full gradient trace:
    tangential moment work is not integrated along the edge. Thus this clamped
    variant has no separately condensed twisting-moment corner force. It does
    not replace the corner terms required for general displacement/slope data.

    stabilization='split' bounds translation and scaled director flux separately.
    Defaults are safety_factor=4 for scalar, 8 for split. With m separately
    integrated boundary objects per cell, require a factor greater than m for
    scalar or 2*m for split. The returned gamma is the largest trace penalty;
    last_trace_penalties stores the six actual weights.
    """
    require_affine = False
    dof_per_cp = 3
    boundary_order = None
    require_tangential_data = True

    def __init__(self, line, prescribed, method='nitsche', penalty=None, safety_factor=None, stabilization='scalar'):
        if method not in ('penalty', 'nitsche'):
            raise ValueError('KL/gradient shell boundary supports penalty or symmetric nitsche only')
        if stabilization not in ('scalar', 'split'):
            raise ValueError('stabilization must be scalar or split')
        if safety_factor is None:
            safety_factor = 8. if stabilization == 'split' else 4.
        if stabilization == 'split' and penalty is None and safety_factor <= 2:
            raise ValueError('Split stabilization safety factor must exceed two')
        self.stabilization = stabilization
        super().__init__(line, prescribed, method=method,
                         penalty=penalty, safety_factor=safety_factor)

    def integrate_contribution(self, cell):
        samples, energy, frames, size = self._boundary_data(cell)
        if not samples:
            return np.zeros((size, size)), np.zeros(size), 0.
        groups = (slice(0, 3), slice(3, 6)) if self.stabilization == 'split' else None
        return self._assemble(cell, samples, size, energy_factor=energy, trace_groups=groups)

    def _boundary_data(self, cell):
        """Full physical traces and conjugate fluxes, before choosing a support."""
        basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
        size = self.dof_per_cp*len(basis)
        order = None if self.boundary_order is None else self.boundary_order(cell.p)
        points = get_boundary_quadrature(self.line, cell, quadrature_order=order)
        if not points:
            return [], None, [], size
        check_kl_continuity(cell)
        gp = cell.gausspoints[0]
        if not isinstance(gp.material_model, LinearElasticMaterial):
            raise ValueError('KL boundary currently supports LinearElasticMaterial only')
        if self.require_affine:
            cov, _, _, _, _, _ = get_full_geometric_data(cell, gp.xi, gp.eta)
            a1, a2 = cov[1], cov[2]
        for g in cell.gausspoints:
            if self.require_affine:
                data, _, variation, _, _, _ = get_full_geometric_data(cell, g.xi, g.eta)
                if (not np.allclose([data[1], data[2]], [a1, a2], rtol=1e-11, atol=1e-12)
                    or any(np.linalg.norm(variation[i][j]) > 1e-11*max(np.linalg.norm(a1), np.linalg.norm(a2))
                           for i in (1, 2) for j in (1, 2))):
                    raise ValueError('WeakKLAffineClampedBoundary requires affine geometry')
            if g.t != gp.t or g.material_model is not gp.material_model:
                raise ValueError('KL boundary requires constant cell material and thickness')
        t = gp.t
        if not np.isfinite(t) or t <= 0:
            raise ValueError('Shell thickness must be positive')
        cps = np.asarray([b.cp for b in basis])
        evaluations = evaluate_derivatives_batch(basis,
            [(point.x, point.y) for point, _, _ in points], cell, max_order=3)
        samples, frames = [], []
        for q, (point, weight, tangent) in enumerate(points):
            data = {order: values[q] for order, values in evaluations.items()}
            jet = surface_jet(cell, point.x, point.y, basis_evaluations=(basis, data))
            trace, flux, length = self._point_operators(jet, tangent, data, gp)
            N = data[0, 0]
            value = np.asarray(self.prescribed(*(N@cps)), dtype=float)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError('Prescribed KL field must return six finite physical components')
            if self.require_tangential_data and abs(jet.normal@value[3:]) > 1e-12+1e-10*np.linalg.norm(value[3:]):
                raise ValueError('Prescribed director change must be tangential')
            value = value*np.array([1., 1., 1., t, t, t])
            measure = weight*length
            samples.append((measure, trace, flux, value))
            physical_tangent = tangent@jet.tangents/length
            normal_s = tangent@jet.normal_gradient/length
            frames.append((jet.position, physical_tangent, np.cross(physical_tangent, jet.normal), normal_s))
        return samples, surface_energy_factor(cell), frames, size

    def _kinematic_operators(self, jet, data, t):
        """Trace, membrane/bending strains and parametric bending derivatives."""
        size = self.dof_per_cp*len(data[0, 0])
        a1, a2 = jet.tangents
        normal = jet.normal
        N = data[0, 0]
        gradient = np.column_stack((data[1, 0], data[0, 1]))
        second = np.column_stack((data[2, 0], data[0, 2], data[1, 1]))
        third = np.column_stack((data[3, 0], data[2, 1], data[1, 2], data[0, 3]))
        membrane, bending, bending_u, bending_v = (np.zeros((3, size)) for _ in range(4))
        trace = np.zeros((6, size))
        for i in range(len(N)):
            start = self.dof_per_cp*i
            block = slice(start, start+3)
            du, dv = gradient[i]
            membrane[:, block] = np.vstack((du*a1, dv*a2, dv*a1+du*a2))
            hessian = np.array([[second[i, 0], second[i, 2]], [second[i, 2], second[i, 1]]])
            covariant = hessian-np.einsum('abc,c->ab', jet.gamma, gradient[i])
            strain = np.array([covariant[0, 0], covariant[1, 1], 2*covariant[0, 1]])
            bending[:, block] = -np.outer(strain, normal)
            for direction, operator in enumerate((bending_u, bending_v)):
                derivative_hessian = np.array([[third[i, direction], third[i, direction+1]],
                    [third[i, direction+1], third[i, direction+2]]])
                covariant_derivative = (derivative_hessian
                    - np.einsum('abc,c->ab', jet.gamma_gradient[direction], gradient[i])
                    - np.einsum('abc,c->ab', jet.gamma, hessian[:, direction]))
                derivative_strain = np.array([covariant_derivative[0, 0],
                    covariant_derivative[1, 1], 2*covariant_derivative[0, 1]])
                operator[:, block] = (-np.outer(derivative_strain, normal)
                    - np.outer(strain, jet.normal_gradient[direction]))
            trace[:3, block] = N[i]*np.eye(3)
            trace[3:, block] = -t*np.outer(gradient[i]@jet.contravariant, normal)
            if self.dof_per_cp == 5:
                # Hierarchical director delta_d = delta_d_KL + a_gamma*eta^gamma.
                trace[3:, start+3:start+5] = t*N[i]*jet.tangents.T
                for gamma in range(2):
                    column = start+3+gamma
                    for row, pairs in enumerate((((0, 0),), ((1, 1),), ((0, 1), (1, 0)))):
                        for a, b in pairs:
                            metric = jet.tangents[gamma]@jet.tangents[b]
                            variation = jet.second[gamma, a]@jet.tangents[b]
                            bending[row, column] += gradient[i, a]*metric+N[i]*variation
                            for direction, operator in enumerate((bending_u, bending_v)):
                                metric_derivative = (jet.second[gamma, direction]@jet.tangents[b]
                                    + jet.tangents[gamma]@jet.second[b, direction])
                                variation_derivative = (jet.third[gamma, a, direction]@jet.tangents[b]
                                    + jet.second[gamma, a]@jet.second[b, direction])
                                operator[row, column] += (hessian[a, direction]*metric
                                    + gradient[i, a]*metric_derivative+gradient[i, direction]*variation
                                    + N[i]*variation_derivative)
        return trace, membrane, bending, (bending_u, bending_v)

    def _point_operators(self, jet, tangent, data, properties):
        """Physical force/director flux, including curved-surface terms."""
        t = properties.t
        size = self.dof_per_cp*len(data[0, 0])
        nu = np.array([tangent[1], -tangent[0]])
        a1, a2 = jet.tangents
        normal = jet.normal
        length = np.linalg.norm(tangent@jet.tangents)
        Dm, Db, _ = properties.material_model.compute_C_voigt_curv_2D(jet.inverse_metric, t)
        Db_gradient = [properties.material_model.compute_C_voigt_curv_2D_directional(
            jet.inverse_metric, direction, t)[1] for direction in jet.inverse_metric_gradient]
        derivative_membrane = (np.vstack((a1, np.zeros(3), a2)),
                               np.vstack((np.zeros(3), a2, a1)))
        trace, membrane, bending, (bending_u, bending_v) = self._kinematic_operators(jet, data, t)
        moment = Db@bending
        moment_u = Db@bending_u+Db_gradient[0]@bending
        moment_v = Db@bending_v+Db_gradient[1]@bending
        membrane_flux = sum(n*derivative.T@Dm@membrane for n, derivative in zip(nu, derivative_membrane))
        moment_divergence = nu[0]*(moment_u[0]+moment_v[2])+nu[1]*(moment_u[2]+moment_v[1])
        curvature_force = np.zeros((3, size))
        moment_tensor = np.array([[moment[0], moment[2]], [moment[2], moment[1]]])
        for direction in range(2):
            connection = (jet.gamma[0, 0, direction]*moment[0]
                + jet.gamma[1, 1, direction]*moment[1]+2*jet.gamma[0, 1, direction]*moment[2])
            curvature_force += nu[direction]*np.outer(normal, connection)
            for derivative in range(2):
                curvature_force += nu[direction]*np.outer(jet.normal_gradient[derivative]
                    + normal*jet.jacobian_gradient[derivative]/jet.jacobian,
                    moment_tensor[direction, derivative])
        force = jet.jacobian/length*(membrane_flux+np.outer(normal, moment_divergence)+curvature_force)
        director_flux = jet.jacobian/length*(np.outer(a1, nu[0]*moment[0]+nu[1]*moment[2])
                                     + np.outer(a2, nu[0]*moment[2]+nu[1]*moment[1]))/t
        flux = np.vstack((force, director_flux))
        return trace, flux, length


class WeakKLAffineClampedBoundary(WeakKLClampedBoundary):
    """Compatibility entry point that rejects non-affine geometry."""
    require_affine = True


class WeakRM5HierClampedBoundary(WeakKLClampedBoundary):
    """Full physical displacement/director conditions for RM5_hier.

    The two additional DOFs are hierarchical director coefficients, not physical
    rotations. Prescribed data are six Cartesian (u, delta_director) values.
    Shear energy enters stabilization; shear has no derivative boundary term
    in this hierarchy because its strain depends only on the additional DOFs.
    """
    dof_per_cp = 5


class WeakRM5HierAffineClampedBoundary(WeakRM5HierClampedBoundary):
    """Restricted entry point that rejects non-affine geometry."""
    require_affine = True
