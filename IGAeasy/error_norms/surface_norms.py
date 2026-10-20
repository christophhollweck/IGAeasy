"""Independent physical L2 quadrature for scalar/vector fields on spline surfaces."""
import copy
import numpy as np
from IGAeasy.splines.batch_evaluation import evaluate_basis_batch
from IGAeasy.numerical_integration.map_gp_to_element import get_untrimmed_element_gp, get_trimmed_element_gp


class SurfaceL2Quadrature:
    """Prepare reusable quadrature on the current geometry/discretization.

    Rebuild after refinement, trimming or control-point movement. Coefficients
    have shape (len(space.basis), n_components), indexed by global basis ID.
    Reference coefficients must use the same discretization, or pass a physical
    callback reference(x,y,z). No stored element integration data is modified.
    """
    def __init__(self, space, order=5):
        if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError('order must be a positive integer')
        self.n_basis = len(space.basis)
        self.entries = []
        for cell in space.elements:
            if getattr(cell, 'void', False) or not getattr(cell, 'active', True):
                continue
            temporary = copy.copy(cell)
            temporary.p = order-1
            if cell.trimmed:
                get_trimmed_element_gp(temporary, full_quad=True, quadrature_degree=2*order-2)
            else:
                get_untrimmed_element_gp(temporary, full_quad=True)
            points = temporary.gausspoints
            basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
            ids = np.array([b.id for b in basis], dtype=int)
            N, gradients = evaluate_basis_batch(basis, [(g.xi, g.eta) for g in points], cell)
            cps = np.asarray([b.cp for b in basis])
            tangents = np.einsum('qbd,bk->qdk', gradients, cps)
            jacobian = np.linalg.norm(np.cross(tangents[:, 0], tangents[:, 1]), axis=1)
            if not np.isfinite(jacobian).all() or np.any(jacobian <= 0):
                raise ValueError('Degenerate surface geometry in L2 integration')
            weights = jacobian*np.array([g.weight_inplane for g in points])
            self._store_entry(ids, N, gradients, weights, N@cps, tangents)

    def _store_entry(self, ids, N, gradients, weights, positions, tangents):
        self.entries.append((ids, N, weights, positions))

    def _sample(self, operator, coefficients):
        return operator@coefficients

    def error(self, coefficients, reference=None, relative=False):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.ndim == 1:
            coefficients = coefficients[:, None]
        if (coefficients.ndim != 2 or coefficients.shape[0] != self.n_basis
                or not np.isfinite(coefficients).all()):
            raise ValueError('Expected finite coefficients indexed by basis ID')
        if reference is not None and not callable(reference):
            reference = np.asarray(reference, dtype=float)
            if reference.ndim == 1:
                reference = reference[:, None]
            if reference.shape != coefficients.shape or not np.isfinite(reference).all():
                raise ValueError('Reference coefficients must match the field layout')
        total, norm = 0., 0.
        for ids, N, weights, positions in self.entries:
            field = self._sample(N, coefficients[ids])
            if reference is None:
                exact = np.zeros_like(field)
            elif callable(reference):
                exact = np.array([reference(*point) for point in positions], dtype=float)
                if exact.ndim == 1 and coefficients.shape[1] == 1:
                    exact = exact[:, None]
                if exact.shape != field.shape or not np.isfinite(exact).all():
                    raise ValueError('Reference callback must match field components')
            else:
                exact = self._sample(N, reference[ids])
            total += weights@np.sum((field-exact)**2, axis=1)
            norm += weights@np.sum(exact**2, axis=1)
        if relative:
            if norm <= 0:
                raise ValueError('Relative error requires a nonzero reference norm')
            return np.sqrt(total/norm)
        return np.sqrt(total)


class SurfaceDirectorL2Quadrature(SurfaceL2Quadrature):
    """Physical director-change L2 norm for KL and RM shell formulations.

    Input is the full shell coefficient array. Reference callbacks
    return three Cartesian director-change components. Coefficient references
    use the same geometry and formulation. No thickness scaling is applied.
    """
    def __init__(self, space, formulation='KL', order=5):
        if formulation not in ('KL', 'RM5_hier', 'RM6', 'RM5', 'RM5_v2'):
            raise ValueError('Unknown shell formulation for the director norm')
        self.formulation = formulation
        self.dof = 3 if formulation == 'KL' else 6 if formulation == 'RM6' else 5
        super().__init__(space, order)

    def _store_entry(self, ids, N, gradients, weights, positions, tangents):
        metric = np.einsum('qak,qbk->qab', tangents, tangents)
        contra = np.linalg.inv(metric)@tangents
        normal = np.cross(tangents[:, 0], tangents[:, 1])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        surface_gradient = np.einsum('qba,qak->qbk', gradients, contra)
        operator = np.zeros((len(N), 3, len(ids), self.dof))
        if self.formulation in ('KL', 'RM5_hier'):
            operator[:, :, :, :3] = -np.einsum('qbk,qj->qkbj', surface_gradient, normal)
        if self.formulation == 'RM5_hier':
            operator[:, :, :, 3:] = np.einsum('qb,qak->qkba', N, tangents)
        elif self.formulation in ('RM5', 'RM5_v2'):
            vectors = contra if self.formulation == 'RM5' else tangents
            operator[:, :, :, 3:] = np.einsum('qb,qak->qkba', N, vectors)
        elif self.formulation == 'RM6':
            rotation_map = np.stack([np.cross(axis, normal) for axis in np.eye(3)], axis=-1)
            operator[:, :, :, 3:] = np.einsum('qb,qka->qkba', N, rotation_map)
        self.entries.append((ids, operator.reshape(len(N), 3, -1), weights, positions))

    def _sample(self, operator, coefficients):
        return operator@coefficients.ravel()

    def error(self, coefficients, reference=None, relative=False):
        if np.shape(coefficients) != (self.n_basis, self.dof):
            raise ValueError('Director coefficients must match the full shell DOF layout')
        return super().error(coefficients, reference, relative)
