"""Surface tangents, frames and changes of basis."""
import numpy as np


def compute_jacobian_surface(cell, xi, eta):
    """Return the two physical tangents, using the supplied cell side."""
    basis = list(cell.supported_b_splines)
    gradients = np.asarray([b.grad(xi, eta, cell=cell) for b in basis])
    return tuple(gradients.T@np.asarray([b.cp for b in basis]))


def surface_normal(tangents):
    """Return the oriented unit normal and physical area Jacobian."""
    cross = np.cross(*tangents)
    jacobian = np.linalg.norm(cross)
    scale = np.prod(np.linalg.norm(tangents, axis=1))
    if not np.isfinite(jacobian) or jacobian <= np.finfo(float).eps*scale:
        raise ValueError('Surface parameterization must have independent finite tangents')
    return cross/jacobian, jacobian


def surface_metric(tangents):
    """Return covariant metric, inverse metric and contravariant tangents."""
    tangents = np.asarray(tangents)
    metric = tangents@tangents.T
    inverse = np.linalg.inv(metric)
    return metric, inverse, inverse@tangents


def orthonormal_basis(tangents, normal):
    first = tangents[0]-np.dot(tangents[0], normal)*normal
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first), normal


def resultant_transformations(a1, a2, e1, e2):
    """Map contravariant symmetric tensors to an orthonormal surface frame."""
    q = np.array([[np.dot(e1, a1), np.dot(e1, a2)],
                  [np.dot(e2, a1), np.dot(e2, a2)]])
    a, b = q[0]
    c, d = q[1]
    membrane_bending = np.array([[a*a, b*b, 2*a*b],
                                [c*c, d*d, 2*c*d],
                                [a*c, b*d, a*d+b*c]])
    return membrane_bending, q
