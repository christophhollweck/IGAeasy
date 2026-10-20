"""Cell-aware surface geometry and first variations of its tangent frame."""
import numpy as np
from IGAeasy.differential_geometry.surface import (
    compute_jacobian_surface, surface_normal, surface_metric, orthonormal_basis,
)
from IGAeasy.splines.batch_evaluation import evaluate_derivatives_batch


def _surface_variations(cell, u, v, geometry_derivatives=None):
    if geometry_derivatives is None:
        basis = list(cell.supported_b_splines)
        cps = np.asarray([b.cp for b in basis])
        data = evaluate_derivatives_batch(basis, [(u, v)], cell, max_order=2)
        a1, a2 = data[1, 0][0]@cps, data[0, 1][0]@cps
        a11, a22, a12 = (data[order][0]@cps for order in ((2, 0), (0, 2), (1, 1)))
    else:
        (a1, a2), (a11, a22, a12) = geometry_derivatives
    normal, jacobian = surface_normal((a1, a2))
    cross_gradient = np.array([np.cross(a11, a2)+np.cross(a1, a12),
                               np.cross(a12, a2)+np.cross(a1, a22)])
    normal_gradient = (cross_gradient-(cross_gradient@normal)[:, None]*normal)/jacobian
    return a1, a2, normal, a11, a22, a12, a12, *normal_gradient


def get_full_geometric_data(el, xi, eta, geometry_derivatives=None):
    """Return covariant/contravariant frames, variations, metrics and local axes.

    geometry_derivatives may supply ((X_u,X_v),(X_uu,X_vv,X_uv)) to reuse
    derivatives evaluated on the same cell and current geometry.
    """
    a1, a2, normal, a11, a22, a12, a21, n1, n2 = _surface_variations(
        el, xi, eta, geometry_derivatives)
    tangents = np.array([a1, a2])
    metric, _, contra = surface_metric(tangents)
    covariant = {1: a1, 2: a2, 3: normal}
    contravariant = {1: contra[0], 2: contra[1], 3: normal}
    variations = {1: {1: a11, 2: a12}, 2: {1: a21, 2: a22}, 3: {1: n1, 2: n2}}
    curvature = np.array([[a11@normal, a12@normal], [a21@normal, a22@normal]])
    frame = dict(enumerate(orthonormal_basis(tangents, normal), start=1))
    return covariant, contravariant, variations, metric, curvature, frame


def get_local_basis(el, xi, eta):
    a1, a2 = compute_jacobian_surface(el, xi, eta)
    normal, _ = surface_normal((a1, a2))
    return a1, a2, normal


def get_local_basis_mixed_variation(el, xi, eta):
    return sum((np.asarray(b.cp)*b.grad_mix(xi, eta, cell=el)
                for b in el.supported_b_splines), np.zeros(3))


def get_local_basis_variation(el, xi, eta):
    basis = list(el.supported_b_splines)
    second = np.asarray([b.grad2(xi, eta, cell=el) for b in basis])
    return tuple(second.T@np.asarray([b.cp for b in basis]))


def get_local_basis_and_variations(el, xi, eta, geometry_derivatives=None):
    """Return tangents, normal and their first parametric derivatives."""
    return _surface_variations(el, xi, eta, geometry_derivatives)
