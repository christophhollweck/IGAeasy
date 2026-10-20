"""Linear coefficient constraints C u = d and a sparse multiplier solve."""
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, MatrixRankWarning


class LinearConstraints:
    def __init__(self, n_dofs):
        if isinstance(n_dofs, bool) or not isinstance(n_dofs, (int, np.integer)) or n_dofs < 1:
            raise ValueError('n_dofs must be a positive integer')
        self.n_dofs = n_dofs
        self._rows = []
        self._values = []

    @classmethod
    def for_control_points(cls, components, n_control_points):
        from IGAeasy.constraints.prescribed_dofs import PrescribedDofs
        layout = PrescribedDofs(components, n_control_points)
        conditions = cls(len(layout.components)*n_control_points)
        conditions.components = layout.components
        conditions.n_control_points = n_control_points
        return conditions

    def dof(self, control_point_id, component):
        if not hasattr(self, 'components'):
            raise ValueError('Use for_control_points() for named components')
        if isinstance(control_point_id, (bool, np.bool_)) or not isinstance(control_point_id, (int, np.integer)) or not 0 <= control_point_id < self.n_control_points:
            raise ValueError(f'Invalid control point ID {control_point_id}')
        if component not in self.components:
            raise ValueError(f'Unknown component {component!r}')
        return len(self.components)*int(control_point_id)+self.components.index(component)

    def prescribe(self, control_point_ids, components, value=0.):
        """Prescribe constant coefficient values; not arbitrary physical fields."""
        names = (components,) if isinstance(components, str) else tuple(components)
        equations = [{self.dof(cp, name): 1.} for cp in control_point_ids for name in names]
        return self._add_many(equations, value)

    def direction(self, control_point_ids, direction, value=0., components=('x', 'y', 'z')):
        """Enforce direction dot coefficient = value; direction is used as given.

        A physical direction must use global Cartesian displacement or RM6
        rotation components, not RM5 director coordinates.
        """
        vector = np.asarray(direction, dtype=float)
        if vector.shape != (len(components),) or not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError('Direction must be finite, nonzero and match the selected components')
        if len(set(components)) != len(components):
            raise ValueError('Direction components must be unique')
        equations = [{self.dof(cp, name): coefficient for name, coefficient in zip(components, vector)}
                     for cp in control_point_ids]
        return self._add_many(equations, value)

    def tie(self, first_id, second_id, components, offset=0.):
        """Enforce u(second)-u(first)=offset componentwise."""
        if first_id == second_id:
            raise ValueError('A tie requires two different control point IDs')
        names = (components,) if isinstance(components, str) else tuple(components)
        equations = [{self.dof(second_id, name): 1., self.dof(first_id, name): -1.} for name in names]
        return self._add_many(equations, offset)

    def _add_many(self, equations, value):
        # Validate all equations before mutating the collection.
        if not equations:
            raise ValueError('Constraint selection is empty')
        temporary = LinearConstraints(self.n_dofs)
        for equation in equations:
            temporary.add(equation, value)
        self._rows.extend(temporary._rows)
        self._values.extend(temporary._values)
        return self

    def add(self, coefficients, value=0.):
        """Add an equation from {global_dof: coefficient}."""
        if np.ndim(value) != 0 or not np.isfinite(value):
            raise ValueError('Constraint value must be a finite scalar')
        row = {}
        for dof, coefficient in coefficients.items():
            if isinstance(dof, (bool, np.bool_)) or not isinstance(dof, (int, np.integer)) or not 0 <= dof < self.n_dofs:
                raise ValueError(f'Invalid DOF {dof}')
            if np.ndim(coefficient) != 0 or not np.isfinite(coefficient):
                raise ValueError('Coefficients must be finite scalars')
            if coefficient != 0:
                row[int(dof)] = float(coefficient)
        if not row:
            raise ValueError('Constraint must have at least one nonzero coefficient')
        self._rows.append(row)
        self._values.append(float(value))
        return self

    def matrices(self):
        rows, columns, data = [], [], []
        for index, equation in enumerate(self._rows):
            for dof, value in equation.items():
                rows.append(index)
                columns.append(dof)
                data.append(value)
        return sparse.csr_matrix((data, (rows, columns)), shape=(len(self._rows), self.n_dofs)), np.array(self._values)


def solve_linear_constraints(K, f, constraints):
    """Solve [K C.T; C 0] [u, lambda] = [f, d].

    All prescribed DOFs must be supplied as equations on the original system.
    Returns displacement and multipliers; constraint forces are -C.T @ lambda.
    This indefinite system requires a direct solver, not conjugate gradients.
    Redundant equations or remaining rigid motions raise an error.
    """
    K = sparse.csc_matrix(K)
    f = np.asarray(f, dtype=float)
    C, d = constraints.matrices()
    n = constraints.n_dofs
    if K.shape != (n, n) or f.shape != (n,):
        raise ValueError('System and constraint dimensions do not match')
    if not np.isfinite(K.data).all() or not np.isfinite(f).all():
        raise ValueError('System contains nonfinite values')
    system = sparse.bmat([[K, C.T], [C, sparse.csc_matrix((len(d), len(d)))]], format='csc')
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        try:
            result = spsolve(system, np.r_[f, d])
        except MatrixRankWarning as error:
            raise ValueError('Singular constrained system: check redundant equations and rigid motions') from error
    if not np.isfinite(result).all():
        raise ValueError('Constrained solve returned nonfinite values')
    return result[:n], result[n:]
