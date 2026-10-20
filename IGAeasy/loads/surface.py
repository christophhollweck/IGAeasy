import numpy as np


class SurfaceLoad:
    def __init__(self, num_local_dofs, load_functions):
        """Accept component functions or one vector-valued function of (u,v)."""
        if not callable(load_functions):
            assert len(load_functions) == num_local_dofs, "Number of load functions must match num_local_dofs."
        self.num_local_dofs = num_local_dofs
        self.load_functions = load_functions

    def evaluate(self, u, v):
        values = (self.load_functions(u, v) if callable(self.load_functions)
                  else [f(u, v) for f in self.load_functions])
        values = np.asarray(values)
        if values.shape != (self.num_local_dofs,):
            raise ValueError('Surface load must return one value per local DOF')
        return values

    def integrate_contribution(self, cell, dof_per_cp):
        """
        Computes the surface load contribution to the local force vector f_el.
        Prepared Gauss basis values are reused when their element ID order matches.
        Recompute element integration data after refinement, as for surface tangents.
        """
        supported_bfs = list(cell.supported_b_splines)
        supported_bfs.sort(key=lambda bf: bf.id)
        n_cps = len(supported_bfs)
        f_el = np.zeros(n_cps * dof_per_cp)
        cached_order = np.array_equal(getattr(cell, 'bfnums', None), [bf.id for bf in supported_bfs])

        for iwip in cell.gausspoints:
            gpx = iwip.xi
            gpy = iwip.eta
            gw = iwip.weight_inplane
            load_values = self.evaluate(gpx, gpy)
            N = getattr(iwip, 'N_evals', None) if cached_order else None
            if np.shape(N) != (n_cps,):
                N = np.array([bf(gpx, gpy) for bf in supported_bfs])
            g1, g2 = iwip.surface_tangents
            detJ = np.linalg.norm(np.cross(g1, g2))

            f_el += (np.asarray(N)[:, None]*load_values[None, :dof_per_cp]*gw*detJ).ravel()

        return f_el
