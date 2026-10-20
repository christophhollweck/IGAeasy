import numpy as np

class PointLoad:
    def __init__(self, param_coords, load_vector):
        """
        Represents a single point load in parametric space.

        Parameters
        ----------
        param_coords : tuple[float, float]
            The (xi, eta) coordinates in parametric space.
        load_vector : array_like
            Load vector applied at the point (e.g. [fx, fy]).
        """
        coordinates = np.asarray(param_coords, dtype=float)
        values = np.asarray(load_vector, dtype=float)
        if coordinates.shape != (2,) or not np.isfinite(coordinates).all():
            raise ValueError('Point-load coordinates must contain two finite values')
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError('Point-load vector must be a nonempty finite vector')
        self.param_coords = tuple(coordinates)
        self.load_vector = values

    def compute_contribution(self, cell, dof_per_cp):
        """
        Compute the element vector contribution due to this point load.

        Parameters
        ----------
        cell : object
            The element/cell containing geometry info and a `.contains()` method,
            as well as a list of supported B-splines with callable basis functions.
        dof_per_cp : int
            Degrees of freedom per control point.

        Returns
        -------
        np.ndarray
            Local right-hand side vector f_el of the element.
        """
        xi, eta = self.param_coords
        supported_bfs = list(cell.supported_b_splines)
        supported_bfs.sort(key=lambda bf: bf.id)
        n_cps = len(supported_bfs)
        f_el = np.zeros(n_cps * dof_per_cp)

        from IGAeasy.trimming.trim_elements import contains_material_point

        if not contains_material_point(cell, xi, eta):
            return f_el

        # Shared edges/vertices belong to one material cell, also after refinement.
        owner = next((other for other in getattr(cell, '_mesh_elements', [cell])
                      if contains_material_point(other, xi, eta)), None)
        if owner is not cell:
            return f_el

        if self.load_vector.shape != (dof_per_cp,) or not np.isfinite(self.load_vector).all():
            raise ValueError('Point load must contain one finite value per DOF')

        N = np.array([bf(xi, eta, cell=cell) for bf in supported_bfs])

        for i in range(n_cps):
            for j in range(dof_per_cp):
                f_el[dof_per_cp * i + j] += N[i] * self.load_vector[j]

        return f_el
