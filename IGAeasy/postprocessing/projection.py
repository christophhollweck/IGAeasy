"""Quadrature-weighted global projection of scalar integration-point data."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr


def project_scalar_results(space, attribute='sigma_vM'):
    """Project a scalar GP attribute or callable(gp) with physical area weights."""
    rows, columns, entries, targets = [], [], [], []
    for cell in space.elements:
        if getattr(cell, 'void', False) or not getattr(cell, 'active', True):
            continue
        for gp in cell.gausspoints:
            basis = list(cell.supported_b_splines)
            derivatives = np.array([b.grad(gp.xi, gp.eta, cell=cell) for b in basis])
            tangents = derivatives.T @ np.array([b.cp for b in basis])
            weight = gp.weight_inplane*np.linalg.norm(np.cross(*tangents))
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError('Projection requires positive finite physical quadrature weights')
            value = attribute(gp) if callable(attribute) else getattr(gp, attribute, None)
            if value is None or np.ndim(value) != 0 or not np.isfinite(value):
                raise ValueError(f'Missing or invalid scalar result {attribute}')
            row = len(targets)
            for b in basis:
                rows.append(row)
                columns.append(b.id)
                entries.append(np.sqrt(weight)*b(gp.xi, gp.eta, cell=cell))
            targets.append(np.sqrt(weight)*value)
    if not targets:
        raise ValueError('No material integration points to project')
    matrix = coo_matrix((entries, (rows, columns)), shape=(len(targets), len(space.basis))).tocsr()
    active = np.flatnonzero(np.asarray(matrix.power(2).sum(axis=0)).ravel())
    result = lsqr(matrix[:, active], targets, atol=1e-12, btol=1e-12,
                  iter_lim=max(1000, 4*len(active)))
    if result[1] not in (0, 1, 2, 4, 5):
        raise RuntimeError(f'Scalar projection failed to converge: LSQR status {result[1]}')
    values = np.zeros(len(space.basis))
    values[active] = result[0]
    return values
