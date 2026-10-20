"""Physical displacement/director penalties for KL and hierarchical RM5 shells."""
import numpy as np
from IGAeasy.constraints.line_helpers import get_integration_points
from IGAeasy.differential_geometry.derivatives import get_full_geometric_data


def _directions(value):
    rows = np.eye(3) if value is None else np.asarray(value, dtype=float)
    if rows.size == 0:
        return np.empty((0, 3))
    rows = np.atleast_2d(rows)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError('Directions must be three-vectors')
    if not len(rows):
        return rows
    if not np.isfinite(rows).all() or np.any(np.linalg.norm(rows, axis=1) == 0):
        raise ValueError('Directions must be finite nonzero three-vectors')
    rows = rows/np.linalg.norm(rows, axis=1)[:, None]
    if not np.allclose(rows@rows.T, np.eye(len(rows)), rtol=0., atol=1e-12):
        raise ValueError('Directions must be mutually orthogonal')
    return rows


class WeakGradientShellPenalty:
    """Prescribe physical (u,delta_director) on oriented material polylines.

    KL uses three translation DOFs. RM5_hier uses translations plus two
    hierarchical director coefficients. prescribed(x,y,z) returns six values.
    penalty_u and penalty_d are explicit dimensional coefficients; no automatic
    Nitsche stabilization is claimed. This implements pure penalty only.
    Directions are global vectors (constant or callable(x,y,z) row matrices).
    Material is to the left of the parametric line, as for WeakRM6Boundary.
    """
    def __init__(self, line, prescribed, formulation, penalty_u=0., penalty_d=0.,
                 displacement_directions=None, director_directions=None):
        if formulation not in ('KL', 'RM5_hier'):
            raise ValueError('formulation must be KL or RM5_hier')
        if line.geom_type not in ('LineString', 'LinearRing') or line.is_empty or line.length == 0:
            raise ValueError('A nonempty oriented polyline is required')
        if (not np.isfinite(penalty_u) or not np.isfinite(penalty_d)
                or penalty_u < 0 or penalty_d < 0 or penalty_u+penalty_d <= 0):
            raise ValueError('Penalty coefficients must be nonnegative with at least one positive')
        self.line, self.prescribed, self.formulation = line, prescribed, formulation
        self.penalty_u, self.penalty_d = penalty_u, penalty_d
        self.displacement_directions = displacement_directions if callable(displacement_directions) else _directions(displacement_directions)
        self.director_directions = director_directions if callable(director_directions) else _directions(director_directions)

    def integrate_contribution(self, cell):
        basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
        dof = 3 if self.formulation == 'KL' else 5
        size = dof*len(basis)
        K, f = np.zeros((size, size)), np.zeros(size)
        coords = np.asarray(self.line.coords)
        lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        cumulative = np.cumsum(lengths)
        cps = np.asarray([b.cp for b in basis])
        for coordinate, weight in get_integration_points(self.line, cell):
            segment = min(np.searchsorted(cumulative, coordinate), len(lengths)-1)
            tangent = (coords[segment+1]-coords[segment])/lengths[segment]
            point = self.line.interpolate(coordinate)
            cov, contra, _, _, _, _ = get_full_geometric_data(cell, point.x, point.y)
            N = np.array([b(point.x, point.y, cell=cell) for b in basis])
            positions = N@cps
            value = np.asarray(self.prescribed(*positions), dtype=float)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError('prescribed must return six finite physical displacement/director components')
            translation, director = np.zeros((3, size)), np.zeros((3, size))
            for index, b in enumerate(basis):
                start = dof*index
                translation[:, start:start+3] = N[index]*np.eye(3)
                grad = b.grad(point.x, point.y, cell=cell)
                surface_gradient = grad[0]*contra[1]+grad[1]*contra[2]
                director[:, start:start+3] = -np.outer(surface_gradient, cov[3])
                if self.formulation == 'RM5_hier':
                    director[:, start+3:start+5] = N[index]*np.column_stack((cov[1], cov[2]))
            measure = weight*np.linalg.norm(tangent[0]*cov[1]+tangent[1]*cov[2])
            for operator, data, coefficient, directions in (
                    (translation, value[:3], self.penalty_u, self.displacement_directions),
                    (director, value[3:], self.penalty_d, self.director_directions)):
                if coefficient == 0:
                    continue
                projection = _directions(directions(*positions)) if callable(directions) else directions
                trace, prescribed = projection@operator, projection@data
                if operator is director:
                    tangent_map = projection@np.column_stack((contra[1], contra[2]))
                    attainable = tangent_map@np.linalg.lstsq(tangent_map, prescribed, rcond=None)[0]
                    if not np.allclose(attainable, prescribed, rtol=1e-10, atol=1e-12):
                        raise ValueError('Prescribed director components must be attainable in the tangent plane')
                K += coefficient*measure*trace.T@trace
                f += coefficient*measure*trace.T@prescribed
        return K, f
