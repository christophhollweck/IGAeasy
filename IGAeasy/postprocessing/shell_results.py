"""Transform contravariant shell resultants to a local orthonormal frame."""
import numpy as np
from IGAeasy.differential_geometry.surface import resultant_transformations
from scipy.interpolate import RBFInterpolator
from IGAeasy.trimming.trim_elements import contains_material_point



def invalidate_shell_interpolators(cell):
    for key in ('N', 'M', 'Q', 'vM'):
        if hasattr(cell, key+'_interp'):
            delattr(cell, key+'_interp')


def reconstruct_shell_stress(N, M, Q, thickness, z):
    """Local stress order (11,22,33,12,23,13); Q/t is average shear."""
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError('thickness must be finite and positive')
    if not np.isfinite(z) or abs(z) > thickness/2:
        raise ValueError('z must lie within the physical thickness')
    inplane = np.asarray(N)/thickness + 12*np.asarray(M)*z/thickness**3
    shear = np.zeros(2) if Q is None else np.asarray(Q)/thickness
    return np.array([inplane[0], inplane[1], 0., inplane[2], shear[1], shear[0]])


def von_mises(stress):
    s1, s2, s3, t12, t23, t13 = stress
    return np.sqrt(((s1-s2)**2+(s2-s3)**2+(s3-s1)**2)/2
                   +3*(t12**2+t23**2+t13**2))


def interpolate_shell_results(space, quantity=None, available=('N', 'M', 'Q', 'vM')):
    if quantity is not None and quantity not in available:
        raise ValueError(f'Unsupported shell result {quantity!r}; available: {available}')
    selected = available if quantity is None else (quantity,)
    for cell in space.elements:
        for key in selected:
            if hasattr(cell, key+'_interp'):
                delattr(cell, key+'_interp')
        if getattr(cell, 'void', False) or not getattr(cell, 'active', True):
            continue
        points = np.array([(gp.xi, gp.eta) for gp in cell.gausspoints])
        if not len(points):
            raise ValueError('Shell result interpolation requires integration points')
        for key in selected:
            attribute = 'sigma_vM' if key == 'vM' else key
            values = [getattr(gp, attribute, None) for gp in cell.gausspoints]
            if any(value is None for value in values):
                raise ValueError(f'Missing {key} results; compute shell results before interpolation')
            values = np.asarray(values, dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f'Nonfinite {key} results')
            setattr(cell, key+'_interp', RBFInterpolator(points, values, kernel='linear', degree=0))


def evaluate_shell_result(space, xi, eta, quantity=None, available=('N', 'M', 'Q', 'vM')):
    key = 'vM' if quantity is None else quantity
    if key not in available:
        raise ValueError(f'Unsupported shell result {key!r}; available: {available}')
    for cell in space.elements:
        if contains_material_point(cell, xi, eta):
            interpolator = getattr(cell, key+'_interp', None)
            if interpolator is None:
                raise ValueError(f'{key} interpolation is missing; interpolate results first')
            return interpolator([[xi, eta]])
    raise ValueError(f'Point ({xi}, {eta}) lies outside the active material domain')
