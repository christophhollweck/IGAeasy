"""Collect strong coefficient constraints independently of geometric selection."""
import numpy as np


def control_point_dofs(ids, names, components, dof_per_cp=None):
    """Map named coefficient components to global control-point-major indices."""
    stride = len(components) if dof_per_cp is None else dof_per_cp
    columns = {name: i for i, name in enumerate(components)}
    selected = (names,) if isinstance(names, str) else tuple(names)
    if any(name not in columns or columns[name] >= stride for name in selected):
        raise ValueError('Selected component is not part of the DOF layout')
    return {stride*cp+columns[name] for cp in ids for name in selected}


class PrescribedDofs:
    """Constant spline-coefficient constraints with explicit component names.

    Nonconstant physical boundary fields require a separate trace projection;
    assigning their values at control points is not generally exact.
    """
    def __init__(self, components, n_control_points):
        self.components = tuple(components)
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError('Component names must be nonempty and unique')
        if isinstance(n_control_points, bool) or not isinstance(n_control_points, (int, np.integer)) or n_control_points < 0:
            raise ValueError('n_control_points must be a nonnegative integer')
        self.n_control_points = n_control_points
        self._values = {}

    def add(self, control_point_ids, components, value=0.):
        """Apply one constant value to the selected coefficients/components."""
        selected = (components,) if isinstance(components, str) else tuple(components)
        if not selected or any(name not in self.components for name in selected):
            raise ValueError(f'Unknown or empty component selection: {selected}')
        if np.ndim(value) != 0 or not np.isfinite(value):
            raise ValueError('Prescribed value must be a finite scalar')
        pending = {}
        for cp in control_point_ids:
            if isinstance(cp, (bool, np.bool_)) or not isinstance(cp, (int, np.integer)) or not 0 <= cp < self.n_control_points:
                raise ValueError(f'Invalid control point ID {cp}')
            for name in selected:
                dof = len(self.components)*int(cp)+self.components.index(name)
                if dof in self._values and self._values[dof] != float(value):
                    raise ValueError(f'Conflicting prescribed values for DOF {dof}')
                pending[dof] = float(value)
        self._values.update(pending)
        return self

    def arrays(self):
        """Return aligned sorted DOF indices and values for block partitioning."""
        indices = np.array(sorted(self._values), dtype=int)
        return indices, np.array([self._values[i] for i in indices], dtype=float)
