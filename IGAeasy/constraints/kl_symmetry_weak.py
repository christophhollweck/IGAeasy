"""Weak reflection symmetry on a planar cut of a KL or hierarchical shell."""
import numpy as np
from IGAeasy.constraints.kl_clamped_weak import WeakKLClampedBoundary


class WeakKLSymmetryBoundary(WeakKLClampedBoundary):
    """Zero transverse-to-plane displacement and director on a reflection cut.

    The cut tangent and surface normal must lie in the reflection plane.
    Complementary force/moment data are zero by symmetry. This specialized
    condition is not an arbitrary partial KL component mask.
    """
    require_tangential_data = False

    def __init__(self, line, plane_normal, method='nitsche', safety_factor=4.):
        direction = np.asarray(plane_normal, dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
            raise ValueError('Reflection plane normal must be a finite nonzero three-vector')
        self.plane_normal = direction/np.linalg.norm(direction)
        super().__init__(line, lambda x, y, z: np.zeros(6), method=method, safety_factor=safety_factor)

    def integrate_contribution(self, cell):
        samples, energy, frames, size = self._boundary_data(cell)
        selected = []
        P = np.zeros((2, 6))
        P[0, :3] = P[1, 3:] = self.plane_normal
        for (measure, trace, flux, value), (_, tangent, conormal, _) in zip(samples, frames):
            normal = np.cross(conormal, tangent)
            if max(abs(normal@self.plane_normal), abs(tangent@self.plane_normal)) > 1e-9:
                raise ValueError('The edge is not a reflection cut of this surface')
            selected.append((measure, P@trace, P@flux, P@value))
        if not selected:
            return np.zeros((size, size)), np.zeros(size), 0.
        return self._assemble(cell, selected, size, energy_factor=energy)
