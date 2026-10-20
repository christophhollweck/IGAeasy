"""Analytic surface derivatives used by fourth-order shell boundary operators."""
from dataclasses import dataclass
import numpy as np
from IGAeasy.differential_geometry.surface import surface_normal, surface_metric


@dataclass
class SurfaceJet:
    """Parameter indices precede spatial indices; gamma[a,b,c]=Gamma^c_ab."""
    position: np.ndarray
    tangents: np.ndarray
    second: np.ndarray
    third: np.ndarray
    normal: np.ndarray
    normal_gradient: np.ndarray
    normal_hessian: np.ndarray
    jacobian: float
    jacobian_gradient: np.ndarray
    inverse_metric: np.ndarray
    inverse_metric_gradient: np.ndarray
    contravariant: np.ndarray
    contravariant_gradient: np.ndarray
    gamma: np.ndarray
    gamma_gradient: np.ndarray


def surface_jet(cell, u, v, basis_evaluations=None):
    """Evaluate a polynomial spline surface through third order, from this cell.

    No neighboring samples or finite differences are used. At knot lines the
    derivatives belong to the supplied cell. Rational weights are not supported
    by the underlying polynomial geometry evaluator.
    basis_evaluations may supply (basis, {(du,dv): point_values}) through order
    three, to reuse a batch on the same point and current geometry.
    """
    basis = list(cell.supported_b_splines) if basis_evaluations is None else basis_evaluations[0]
    cp = np.asarray([b.cp for b in basis], dtype=float)
    if basis_evaluations is None:
        values = np.asarray([b(u, v, cell=cell) for b in basis])
        gradient = np.asarray([b.grad(u, v, cell=cell) for b in basis])
        second_values = np.asarray([[*b.grad2(u, v, cell=cell), b.grad_mix(u, v, cell=cell)] for b in basis])
        third_values = np.asarray([b.grad3(u, v, cell=cell) for b in basis])
    else:
        data = basis_evaluations[1]
        values = data[0, 0]
        gradient = np.column_stack((data[1, 0], data[0, 1]))
        second_values = np.column_stack((data[2, 0], data[0, 2], data[1, 1]))
        third_values = np.column_stack((data[3, 0], data[2, 1], data[1, 2], data[0, 3]))
    tangents = gradient.T@cp
    second = np.empty((2, 2, 3))
    second[0, 0], second[1, 1], second[0, 1] = second_values.T@cp
    second[1, 0] = second[0, 1]
    third = np.empty((2, 2, 2, 3))
    coefficients = third_values.T@cp
    for a in range(2):
        for b in range(2):
            for c in range(2):
                third[a, b, c] = coefficients[a+b+c]
    cross = np.cross(*tangents)
    normal, jacobian = surface_normal(tangents)
    cross_gradient = np.asarray([np.cross(second[0, a], tangents[1])
        + np.cross(tangents[0], second[1, a]) for a in range(2)])
    jacobian_gradient = cross_gradient@normal
    normal_gradient = (cross_gradient-jacobian_gradient[:, None]*normal)/jacobian
    normal_hessian = np.empty((2, 2, 3))
    for a in range(2):
        for b in range(2):
            cross_second = (np.cross(third[0, a, b], tangents[1])
                + np.cross(second[0, a], second[1, b])
                + np.cross(second[0, b], second[1, a])
                + np.cross(tangents[0], third[1, a, b]))
            jacobian_second = (cross_gradient[a]@cross_gradient[b]
                + cross@cross_second-jacobian_gradient[a]*jacobian_gradient[b])/jacobian
            normal_hessian[a, b] = (cross_second/jacobian
                - (cross_gradient[a]*jacobian_gradient[b]+cross_gradient[b]*jacobian_gradient[a])/jacobian**2
                - normal*jacobian_second/jacobian
                + 2*normal*jacobian_gradient[a]*jacobian_gradient[b]/jacobian**2)
    _, inverse_metric, contravariant = surface_metric(tangents)
    inverse_metric_gradient = np.asarray([-inverse_metric@(second[:, a]@tangents.T
        + tangents@second[:, a].T)@inverse_metric for a in range(2)])
    contravariant_gradient = np.asarray([inverse_metric_gradient[a]@tangents
        + inverse_metric@second[:, a] for a in range(2)])
    gamma = np.einsum('ck,abk->abc', contravariant, second)
    gamma_gradient = np.asarray([np.einsum('ck,abk->abc', contravariant_gradient[a], second)
        + np.einsum('ck,abk->abc', contravariant, third[:, :, a]) for a in range(2)])
    return SurfaceJet(values@cp, tangents, second, third, normal, normal_gradient,
        normal_hessian, jacobian, jacobian_gradient, inverse_metric,
        inverse_metric_gradient, contravariant, contravariant_gradient, gamma, gamma_gradient)
