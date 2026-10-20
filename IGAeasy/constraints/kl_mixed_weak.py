"""Weak displacement and normal-director conditions on polynomial KL shells."""
import numpy as np
from IGAeasy.constraints.kl_clamped_weak import WeakKLClampedBoundary
from IGAeasy.constraints.shell_gradient_penalty import _directions


def lift_boundary_work(trace, consistency, bulk, energy_factor=None, work_factors=None):
    """Represent complete boundary work in its sampled essential trace space.

    work_factors=(L,R) supplies C=L.T@R without forming its cross product
    before the energy inverse. This retains small modal work on tiny cuts.
    """
    diagonal = np.diag(bulk)
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise ValueError('Boundary lifting requires positive finite bulk diagonals')
    scale = np.sqrt(diagonal)
    U, singular, Vh = np.linalg.svd(trace/scale, full_matrices=False)
    positive = singular > max(trace.shape)*np.finfo(float).eps*singular.max()
    work = consistency/np.outer(scale, scale)
    if energy_factor is not None:
        normalized = energy_factor/scale
        _, roots, energy_vectors = np.linalg.svd(
            normalized, full_matrices=normalized.shape[0] < len(scale))
        roots = np.pad(roots, (0, len(scale)-len(roots)))
        active = roots > max(normalized.shape)*np.finfo(float).eps*roots.max()
        energy_vectors = energy_vectors[active]
        controlled = (work@energy_vectors.T)@energy_vectors
        if np.linalg.norm(work-controlled) > 1e-8*max(np.linalg.norm(work), 1e-300):
            raise ValueError('Boundary work acts on an uncontrolled bulk mode')
        # Enforce the exact zero work of rigid modes before the trace inverse
        # amplifies roundoff on short or nearly dependent boundary segments.
        work = controlled
    projected = Vh[positive].T@(Vh[positive]@work)
    if np.linalg.norm(work-projected) > 1e-8*max(np.linalg.norm(work), 1e-300):
        raise ValueError('Essential trace does not control tangential moment work')
    if work_factors is None:
        lifted = (U[:, positive]/singular[positive])@(Vh[positive]@work)
    else:
        if energy_factor is None:
            raise ValueError('Factored boundary work requires an energy factor')
        left, right = work_factors
        # Apply the energy inverse before forming the cross product. Forming
        # L.T@R first loses small modal work which both inverses amplify.
        inverse_root = energy_vectors.T/roots[active]
        modal_work = (left/scale).T@((right/scale)@inverse_root)
        modal_flux = (U[:, positive]/singular[positive])@(Vh[positive]@modal_work)
        lifted = modal_flux@(roots[active, None]*energy_vectors)
    if energy_factor is not None:
        lifted = (lifted@energy_vectors.T)@energy_vectors
    if np.linalg.norm((trace/scale).T@lifted-work) > 1e-8*max(np.linalg.norm(work), 1e-300):
        raise ValueError('Boundary lifting did not preserve the controlled work')
    return lifted*scale


class WeakKLMixedBoundary(WeakKLClampedBoundary):
    """Prescribe displacement directions, optionally also normal director change.

    With normal_rotation=False the normal director is free; normal_moment
    (x,y,z,conormal) supplies its conjugate physical moment, zero by default.
    prescribed returns compatible physical (u,delta_director) data. Its
    tangential director component is required even when normal rotation is free.
    The complete tangential moment work is retained, including corner effects.
    Constant displacement_directions may select a subspace containing the
    surface normal along the boundary (e.g. a cylindrical diaphragm).
    Other component masks are deliberately rejected. traction(x,y,z,conormal)
    optionally supplies physical force on the complementary directions.
    displacement_directions='normal' prescribes only n.u. Its curved trace uses
    d_t-t*n_s.u=-t*ds(n.u); traction then includes n_s times the tangential moment.
    Supplying the raw uncondensed force alone is not the matching natural datum.
    Geometry, isotropic material and C1 restrictions match the clamped operator.
    """
    boundary_order = staticmethod(lambda p: 2*p+1)

    def __init__(self, line, prescribed, normal_rotation=False, normal_moment=None,
                 method='nitsche', penalty=None, safety_factor=4., displacement_directions=None,
                 traction=None):
        if normal_rotation and normal_moment is not None:
            raise ValueError('Choose prescribed normal rotation or applied normal moment')
        self.normal_rotation = normal_rotation
        self.normal_moment = normal_moment
        self.traction = traction
        self.normal_only = isinstance(displacement_directions, str) and displacement_directions == 'normal'
        self.displacement_projector = None if self.normal_only else _directions(displacement_directions)
        if not self.normal_only and not len(self.displacement_projector):
            raise ValueError('Select at least one displacement direction')
        super().__init__(line, prescribed, method=method, penalty=penalty,
                         safety_factor=safety_factor)

    def integrate_contribution(self, cell):
        samples, energy, frames, size = self._boundary_data(cell)
        if not samples:
            return np.zeros((size, size)), np.zeros(size), 0.
        rows, data, work_left, work_right = [], [], [], []
        consistency = np.zeros((size, size))
        prescribed_work = np.zeros(size)
        natural_load = np.zeros(size)
        t = cell.gausspoints[0].t
        for (measure, trace, flux, value), (position, tangent, conormal, normal_s) in zip(samples, frames):
            normal = np.cross(conormal, tangent)
            P = normal[None, :] if self.normal_only else self.displacement_projector
            if np.linalg.norm(normal-P.T@(P@normal)) > 1e-10:
                raise ValueError('KL displacement directions must contain the surface normal along the boundary')
            theta_t, moment_t = tangent@trace[3:], tangent@flux[3:]
            theta_n, moment_n = conormal@trace[3:], conormal@flux[3:]
            force = flux[:3]
            given_theta_t = tangent@value[3:]
            if self.normal_only:
                # d_t - t*n_s.u = -t*ds(n.u): use the derivative of the essential trace.
                theta_t = theta_t-t*(normal_s@trace[:3])
                given_theta_t -= t*(normal_s@value[:3])
                force = force+t*np.outer(normal_s, moment_t)
            work = (P@trace[:3]).T@(P@force)+np.outer(theta_t, moment_t)
            left = np.vstack((P@trace[:3], theta_t))
            right = np.vstack((P@force, moment_t))
            prescribed_work -= measure*((P@force).T@(P@value[:3])+moment_t*given_theta_t)
            selected, given = P@trace[:3], P@value[:3]
            if self.normal_rotation:
                work += np.outer(theta_n, moment_n)
                prescribed_work -= measure*moment_n*(conormal@value[3:])
                selected = np.vstack((selected, theta_n))
                given = np.r_[given, conormal@value[3:]]
                left = np.vstack((left, theta_n))
                right = np.vstack((right, moment_n))
            if self.normal_moment is not None:
                moment = self.normal_moment(*position, conormal)
                if np.ndim(moment) != 0 or not np.isfinite(moment):
                    raise ValueError('Applied normal moment must return one finite scalar')
                natural_load += measure*theta_n*moment/t
            if self.traction is not None:
                force = np.asarray(self.traction(*position, conormal), dtype=float)
                if force.shape != (3,) or not np.isfinite(force).all():
                    raise ValueError('traction must return three finite physical components')
                natural_load += measure*trace[:3].T@(force-P.T@(P@force))
            consistency += measure*work
            rows.append(np.sqrt(measure)*selected)
            data.extend(np.sqrt(measure)*given)
            work_left.append(np.sqrt(measure)*left)
            work_right.append(np.sqrt(measure)*right)
        trace, value = np.vstack(rows), np.asarray(data)
        flux = lift_boundary_work(trace, consistency, cell.K_el, energy_factor=energy,
                                  work_factors=(np.vstack(work_left), np.vstack(work_right)))
        Kbc, fbc, gamma = self._assemble(
            cell, [(1., trace, flux, value)], size, energy_factor=energy)
        if self.method == 'nitsche':
            fbc += flux.T@value+prescribed_work
        return Kbc, fbc+natural_load, gamma
