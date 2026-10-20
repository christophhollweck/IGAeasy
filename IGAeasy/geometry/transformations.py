"""Rigid transformations of physical geometry and vector fields."""
import numpy as np


def rotation_matrix(angle, axis=(1., 2., 3.)):
    """Rodrigues rotation in radians about a finite nonzero spatial axis."""
    axis = np.asarray(axis, dtype=float)
    if not np.isfinite(angle) or axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
        raise ValueError('Rotation requires a finite angle and a nonzero three-vector axis')
    axis = axis/np.linalg.norm(axis)
    ax, ay, az = axis
    skew = np.array([[0., -az, ay], [az, 0., -ax], [-ay, ax, 0.]])
    return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*skew@skew
