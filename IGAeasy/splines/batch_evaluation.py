"""Batched tensor-product derivatives with cell-aware boundary traces."""
from functools import lru_cache
import numpy as np
from scipy.interpolate import BSpline
from IGAeasy.splines.univariate import _augment_knots


@lru_cache(maxsize=4096)
def _polynomial(knots, degree):
    augmented = _augment_knots(knots, degree)
    coefficients = np.zeros(len(augmented)-degree-1)
    coefficients[degree+1] = 1.
    return BSpline(augmented, coefficients, degree, extrapolate=False)


def evaluate_derivatives_batch(basis, points, cell, max_order=2):
    """Return {(du,dv): array(points,basis)} through total derivative order.

    Boundary points use the existing cell-aware evaluator to preserve sidedness.
    Cached polynomials depend only on immutable knot values, not mutable weights.
    """
    if isinstance(max_order, (bool, np.bool_)) or not isinstance(max_order, (int, np.integer)) or max_order < 0:
        raise ValueError('max_order must be a nonnegative integer')
    basis = list(basis)
    orders = [(total-r, r) for total in range(max_order+1) for r in range(total+1)]
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    tolerance = 8*np.finfo(float).eps*max(abs(cell.u_min), abs(cell.u_max), abs(cell.v_min), abs(cell.v_max),
                                        cell.u_max-cell.u_min, cell.v_max-cell.v_min)
    if not np.isfinite(points).all() or np.any(points[:, 0] < cell.u_min-tolerance) or np.any(points[:, 0] > cell.u_max+tolerance) or np.any(points[:, 1] < cell.v_min-tolerance) or np.any(points[:, 1] > cell.v_max+tolerance):
        raise ValueError('Batch points must lie inside the element')
    univariate, evaluated = {}, {}

    def evaluate(function):
        key = id(function)
        if key in evaluated:
            return evaluated[key]
        if getattr(function, 'trunc', False):
            data = {order: np.zeros(len(points)) for order in orders}
            for coefficient, child in zip(function.coefficient, function.children):
                if not child.active:
                    child_data = evaluate(child)
                    for order in orders:
                        data[order] += coefficient*child_data[order]
        else:
            directions = []
            for axis, knots in enumerate((function.knots_u, function.knots_v)):
                cache_key = (axis, tuple(knots), function.p)
                if cache_key not in univariate:
                    polynomial = _polynomial(tuple(knots), function.p)
                    x = points[:, axis]
                    univariate[cache_key] = [np.nan_to_num(polynomial(x, nu=r))
                        if r <= function.p else np.zeros(len(x)) for r in range(max_order+1)]
                directions.append(univariate[cache_key])
            u, v = directions
            weight = getattr(function, 'weight', 1.)
            data = {(a, b): weight*u[a]*v[b] for a, b in orders}
        evaluated[key] = data
        return data

    result = {order: np.empty((len(points), len(basis))) for order in orders}
    boundary = ((points[:, 0] <= cell.u_min) | (points[:, 0] >= cell.u_max)
                | (points[:, 1] <= cell.v_min) | (points[:, 1] >= cell.v_max))
    for index, function in enumerate(basis):
        data = evaluate(function)
        for order in orders:
            result[order][:, index] = data[order]
        for point_index in np.flatnonzero(boundary):
            u, v = points[point_index]
            for a, b in orders:
                result[a, b][point_index, index] = function(u, v, r1=a, r2=b, cell=cell)
    return result


def evaluate_basis_batch(basis, points, cell):
    """Return values (points,basis) and gradients (points,basis,2)."""
    data = evaluate_derivatives_batch(basis, points, cell, max_order=1)
    return data[0, 0], np.stack((data[1, 0], data[0, 1]), axis=-1)
