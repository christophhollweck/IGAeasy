"""Common numerical assembly for displacement and director boundary traces."""
import numpy as np
from scipy.linalg import eigh


def coefficient_trace(values, dof_per_cp):
    """Interpolate coefficient components in control-point-major order."""
    return np.kron(np.asarray(values)[None, :], np.eye(dof_per_cp))


def surface_energy_factor(cell):
    """Factor the membrane/bending/shear energy of a surface shell element."""
    factors = []
    for gp in cell.gausspoints:
        measure = gp.weight_inplane*np.linalg.norm(np.cross(*gp.surface_tangents))
        for material, strain in zip(gp.D, gp.Bmatrices[0]):
            factors.append(np.sqrt(measure)*np.linalg.cholesky(material).T@strain)
    return np.vstack(factors)


class WeakBoundary:
    def __init__(self, line, prescribed, method='nitsche', penalty=None, safety_factor=4.):
        if method not in ('penalty', 'nitsche', 'penalty_free'):
            raise ValueError('method must be penalty, nitsche or penalty_free')
        if method == 'penalty_free' and penalty is not None:
            raise ValueError('penalty_free does not accept a penalty coefficient')
        if line.geom_type not in ('LineString', 'LinearRing') or line.is_empty or line.length == 0:
            raise ValueError('A nonempty oriented polyline is required')
        if not np.isfinite(safety_factor) or safety_factor <= 1:
            raise ValueError('safety_factor must exceed one')
        if penalty is not None and (not np.isfinite(penalty) or penalty <= 0):
            raise ValueError('penalty must be positive and finite')
        self.line, self.prescribed, self.method = line, prescribed, method
        self.penalty, self.safety_factor = penalty, safety_factor

    def _assemble(self, cell, samples, size, energy_factor=None, trace_groups=None):
        if self.method == 'penalty_free':
            gamma = 0.
        elif self.penalty is None:
            K = (cell.K_el+cell.K_el.T)/2
            diagonal = np.diag(K)
            if np.any(diagonal <= 0):
                raise ValueError('Automatic stabilization requires positive coefficient energies')
            energy_scale = np.sqrt(diagonal)
            if energy_factor is None:
                values, vectors = eigh(K/np.outer(energy_scale, energy_scale))
                threshold = 1e-12*max(abs(values).max(), 1e-300)
                if values.min() < -threshold:
                    raise ValueError('Automatic stabilization requires positive-semidefinite bulk stiffness')
                positive = values > threshold
                roots = np.sqrt(np.maximum(values, 0.))
            else:
                # Factorization before squaring retains small cut-cell energy modes.
                if not np.allclose(energy_factor.T@energy_factor, K, rtol=1e-10,
                                   atol=1e-12*max(abs(K).max(), 1e-300)):
                    raise ValueError('Energy factor must match the original bulk stiffness')
                normalized = energy_factor/energy_scale
                _, roots, Vh = np.linalg.svd(normalized, full_matrices=normalized.shape[0] < size)
                roots = np.pad(roots, (0, size-len(roots)))
                vectors = Vh.T
                rank_tolerance = max(normalized.shape)*np.finfo(float).eps*max(roots.max(), 1e-300)
                positive = roots > rank_tolerance
            null = vectors[:, ~positive]
            boundary_factor = np.vstack([np.sqrt(measure)*flux for measure, _, flux, _ in samples])
            normalized_flux = boundary_factor/energy_scale
            # Checking the flux itself avoids hiding small uncontrolled components.
            if np.linalg.norm(normalized_flux@null) > 1e-8*max(np.linalg.norm(normalized_flux), 1e-300):
                raise ValueError('Uncontrolled shell traction modes; check quadrature or cut stabilization')
            if not positive.any():
                raise ValueError('No positive bulk energy modes')
            inverse_root = (vectors[:, positive]/roots[positive])/energy_scale[:, None]
            if trace_groups is None:
                bound = np.linalg.svd(boundary_factor@inverse_root, compute_uv=False)[0]**2
                gamma = self.safety_factor*bound
            else:
                if self.safety_factor <= len(trace_groups):
                    raise ValueError('Split stabilization safety factor must exceed the number of trace groups')
                gamma = np.zeros(samples[0][1].shape[0])
                for group in trace_groups:
                    factor = np.vstack([np.sqrt(measure)*flux[group] for measure, _, flux, _ in samples])
                    bound = np.linalg.svd(factor@inverse_root, compute_uv=False)[0]**2
                    gamma[group] = self.safety_factor*bound
        else:
            gamma = self.penalty
        self.last_trace_penalties = np.broadcast_to(gamma, (samples[0][1].shape[0],)).copy()
        Kbc, fbc = np.zeros((size, size)), np.zeros(size)
        for measure, N, T, value in samples:
            stabilized_trace = self.last_trace_penalties[:, None]*N
            Kbc += measure*N.T@stabilized_trace
            fbc += measure*stabilized_trace.T@value
            if self.method == 'nitsche':
                Kbc -= measure*(N.T@T+T.T@N)
                fbc -= measure*T.T@value
            elif self.method == 'penalty_free':
                Kbc += measure*(-N.T@T+T.T@N)
                fbc += measure*T.T@value
        return Kbc, fbc, float(np.max(gamma))
