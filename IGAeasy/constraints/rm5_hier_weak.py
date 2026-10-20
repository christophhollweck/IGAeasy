"""Selected physical boundary conditions for the hierarchical RM5 shell."""
import numpy as np
from IGAeasy.constraints.kl_clamped_weak import WeakRM5HierClampedBoundary
from IGAeasy.constraints.shell_weak import trace_projector, select_director_data


class WeakRM5HierBoundary(WeakRM5HierClampedBoundary):
    """Penalty/Nitsche for selected physical (u,delta_director) directions.

    Hierarchical eta coefficients are not prescribed as physical rotations.
    Unlike KL, the director trace is independently variable through eta;
    unselected director components receive their natural moment condition.
    natural(x,y,z,conormal) optionally supplies physical (force,moment) components
    on the complementary trace. Default natural data are zero.
    directions may be constant rows or callable(x,y,z) rows, acting on (u,t*d).
    """
    boundary_order = staticmethod(lambda p: 2*p+1)
    require_tangential_data = False

    def __init__(self, line, prescribed, method='nitsche', penalty=None, safety_factor=4.,
                 components=('ux','uy','uz','dx','dy','dz'), directions=None, natural=None):
        super().__init__(line, prescribed, method=method, penalty=penalty, safety_factor=safety_factor)
        self.components = tuple(components)
        self.names = ('ux','uy','uz','dx','dy','dz')
        self.directions, self.natural = directions, natural
        if callable(directions):
            if self.components != self.names:
                raise ValueError('Use either components or directions')
        else:
            self.projector = trace_projector(self.components, self.names, directions)

    def integrate_contribution(self, cell):
        samples, energy, frames, size = self._boundary_data(cell)
        if not samples:
            return np.zeros((size,size)), np.zeros(size), 0.
        selected, load = [], np.zeros(size)
        t = cell.gausspoints[0].t
        for (measure, trace, flux, value), (position, tangent, conormal, _) in zip(samples, frames):
            P = (trace_projector(self.components, self.names, self.directions(*position))
                 if callable(self.directions) else self.projector)
            if selected and len(P) != selected[0][1].shape[0]:
                raise ValueError('The number of selected directions must remain constant along the boundary')
            normal = np.cross(conormal, tangent)
            prescribed = select_director_data(P, value, normal)
            selected.append((measure, P@trace, P@flux, prescribed))
            if self.natural is not None:
                data = np.asarray(self.natural(*position, conormal), dtype=float)
                if data.shape != (6,) or not np.isfinite(data).all():
                    raise ValueError('natural must return six finite physical force/moment values')
                data = data/np.array([1.,1.,1.,t,t,t])
                load += measure*trace.T@(data-P.T@(P@data))
        matrix, rhs, gamma = self._assemble(cell, selected, size, energy_factor=energy)
        return matrix, rhs+load, gamma
