"""Penalty and symmetric Nitsche displacement conditions for linear XY membranes."""
import numpy as np
from scipy.linalg import eigh
from IGAeasy.constraints.line_helpers import get_integration_points


class WeakMembraneBoundary:
    """Prescribe both Cartesian displacements on an oriented parametric polyline.

    Material must lie to the left of the line: outer rings counterclockwise,
    hole rings clockwise. The outward physical normal follows this orientation.
    prescribed(x,y) returns two displacement components in physical coordinates.
    direction selects one normalized global displacement direction; None enforces
    both components. The prescribed vector is projected onto that direction.
    Automatic gamma uses a discrete local traction/energy bound; it requires
    adequately integrated, positive-semidefinite element stiffness.
    """
    def __init__(self, line, prescribed, method='nitsche', penalty=None, safety_factor=2.,
                 direction=None):
        if method not in ('penalty', 'nitsche'):
            raise ValueError('method must be penalty or nitsche')
        if not np.isfinite(safety_factor) or safety_factor <= 1:
            raise ValueError('safety_factor must exceed one')
        if penalty is not None and (not np.isfinite(penalty) or penalty <= 0):
            raise ValueError('penalty must be positive and finite')
        if line.geom_type not in ('LineString', 'LinearRing') or line.is_empty or line.length == 0:
            raise ValueError('A nonempty oriented polyline is required')
        self.line, self.prescribed = line, prescribed
        self.method, self.penalty, self.safety_factor = method, penalty, safety_factor
        if direction is None:
            self.projector = np.eye(2)
        else:
            vector = np.asarray(direction, dtype=float)
            if vector.shape != (2,) or not np.isfinite(vector).all() or not np.any(vector):
                raise ValueError('direction must be a finite nonzero Cartesian two-vector')
            vector = vector/np.linalg.norm(vector)
            self.projector = np.outer(vector, vector)

    def integrate_contribution(self, cell):
        return self._integrate(cell, both=False)

    def integrate_both(self, cell):
        """Compute penalty and Nitsche terms using one quadrature and energy bound."""
        return self._integrate(cell, both=True)

    def _integrate(self, cell, both):
        basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
        size = 2*len(basis)
        samples = []
        H = np.zeros((size, size))
        coords = np.asarray(self.line.coords)
        cps = np.array([b.cp for b in basis])
        lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        cumulative = np.cumsum(lengths)
        for s, weight in get_integration_points(self.line, cell):
            index = min(np.searchsorted(cumulative, s), len(lengths)-1)
            tangent_param = (coords[index+1]-coords[index])/lengths[index]
            point = self.line.interpolate(s)
            values = np.array([b(point.x, point.y, cell=cell) for b in basis])
            derivatives = np.array([b.grad(point.x, point.y, cell=cell) for b in basis]).T
            if not np.allclose(cps[:, 2], cps[0, 2], atol=1e-14, rtol=0):
                raise ValueError('WeakMembraneBoundary requires a planar XY membrane')
            J = derivatives@cps[:, :2]
            physical_tangent = J.T@tangent_param
            measure = weight*np.linalg.norm(physical_tangent)
            normal = np.linalg.solve(J, [tangent_param[1], -tangent_param[0]])
            normal /= np.linalg.norm(normal)
            grad = np.linalg.solve(J, derivatives)
            B = np.zeros((3, size))
            B[0, ::2], B[1, 1::2] = grad[0], grad[1]
            B[2, ::2], B[2, 1::2] = grad[1], grad[0]
            N = np.zeros((2, size))
            N[0, ::2], N[1, 1::2] = values, values
            gp = cell.gausspoints[0]
            traction = gp.t*np.array([[normal[0], 0., normal[1]],
                                      [0., normal[1], normal[0]]])@gp.material_model.get_C_voigt_cart_2D()@B
            value = np.asarray(self.prescribed(*(values@cps[:, :2])), dtype=float)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError('Prescribed field must return two finite components')
            N = self.projector@N
            traction = self.projector@traction
            value = self.projector@value
            H += measure*traction.T@traction
            samples.append((measure, N, traction, value))
        if not samples:
            empty = (np.zeros((size, size)), np.zeros(size), 0.)
            return {'penalty': empty, 'nitsche': empty} if both else empty
        if self.penalty is None:
            K = (cell.K_el+cell.K_el.T)/2
            # Cut cells can give vastly different coefficient energy scales.
            # Equilibrate both matrices before deciding which modes are null.
            diagonal = np.diag(K)
            if np.any(diagonal <= 0):
                raise ValueError('Automatic stabilization requires positive coefficient energies')
            scale = np.sqrt(diagonal)
            scaled_K = K/np.outer(scale, scale)
            scaled_H = H/np.outer(scale, scale)
            eigenvalues, vectors = eigh(scaled_K)
            threshold = 1e-12*max(abs(eigenvalues).max(), 1e-300)
            if eigenvalues.min() < -threshold:
                raise ValueError('Automatic stabilization requires positive-semidefinite element stiffness')
            positive = eigenvalues > threshold
            null = vectors[:, ~positive]
            if np.linalg.norm(scaled_H@null) > 1e-8*max(np.linalg.norm(scaled_H), 1e-300):
                raise ValueError('Uncontrolled traction modes: increase quadrature or stabilize the cut element')
            if not positive.any():
                raise ValueError('No positive element energy modes')
            inverse_root = (vectors[:, positive]/np.sqrt(eigenvalues[positive]))/scale[:, None]
            bound = np.linalg.eigvalsh(inverse_root.T@H@inverse_root).max()
            gamma = self.safety_factor*bound
        else:
            gamma = self.penalty
        Kbc, fbc = np.zeros((size, size)), np.zeros(size)
        Kcons, fcons = np.zeros((size, size)), np.zeros(size)
        for measure, N, T, value in samples:
            Kbc += measure*gamma*N.T@N
            fbc += measure*gamma*N.T@value
            if both or self.method == 'nitsche':
                Kcons -= measure*(N.T@T+T.T@N)
                fcons -= measure*T.T@value
        if both:
            return {'penalty': (Kbc, fbc, gamma),
                    'nitsche': (Kbc+Kcons, fbc+fcons, gamma)}
        return Kbc+Kcons, fbc+fcons, gamma
