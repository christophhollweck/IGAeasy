import typing
import numpy as np
from IGAeasy.splines.basis_identity import basis_key
from IGAeasy.splines.univariate import (
    _evaluate_univariate_b_spline, _find_knot_interval, _augment_knots,
)

Vector = typing.List[float]

class BSpline:
    def __init__(self, p: int, id: int, level: int,
                 knots_u: Vector, knots_v: Vector,
                 cp: typing.Sequence[float],
                 end_u: bool = False, end_v: bool = False) -> None:

        self.p = p
        self.knots_u = np.array(knots_u, dtype=np.float64)
        self.norm_knots_u = self.knots_u - self.knots_u[0]

        self.knots_v = np.array(knots_v, dtype=np.float64)
        self.norm_knots_v = self.knots_v - self.knots_v[0]
        self.knots_u.setflags(write=False)
        self.knots_v.setflags(write=False)
        self._basis_key = basis_key(p, self.knots_u, self.knots_v)

        self.cp = np.array(cp, dtype=np.float64)
        self.end_u = end_u
        self.end_v = end_v
        self.level = level
        self.loc_id = id
        self.id = None

        self.trunc = False
        self.coefficient = []
        self.children = []

        self.active = None
        self.trimmed = False
        self.void = False
        self.displacements = np.zeros(3)
        self.scalar_value = 0.0
        self.elements_of_support = []

    def __call__(self, u: float, v: float, r1=0, r2=0, cell=None) -> float:
        u_norm = u - self.knots_u[0]
        v_norm = v - self.knots_v[0]

        if cell is not None and not self.trunc:
            knotspan_norm_u = (cell.u_min - self.knots_u[0], cell.u_max - self.knots_u[0])
            knotspan_norm_v = (cell.v_min - self.knots_v[0], cell.v_max - self.knots_v[0])
        elif cell is not None and self.trunc:
            cell = next((child for child in cell.children if child.contains(u, v)), cell)
        else:
            knotspan_norm_u = None
            knotspan_norm_v = None

        if not self.trunc:
            Bu = _evaluate_univariate_b_spline(u_norm, self.norm_knots_u, self.p, self.end_u, r1, knotspan_norm_u)
            Bv = _evaluate_univariate_b_spline(v_norm, self.norm_knots_v, self.p, self.end_v, r2, knotspan_norm_v)
            return Bu * Bv
        else:
            return sum(c * b(u, v, r1, r2, cell=cell) for c, b in zip(self.coefficient, self.children)
                       if not b.active)

    def grad(self, u, v, cell=None) -> np.ndarray:
        return np.array([
            self(u, v, r1=1, r2=0, cell=cell),
            self(u, v, r1=0, r2=1, cell=cell)
        ])

    def grad_mix(self, u, v, cell=None) -> float:
        return self(u, v, r1=1, r2=1, cell=cell)

    def grad2(self, u, v, cell=None) -> np.ndarray:
        return np.array([
            self(u, v, r1=2, r2=0, cell=cell),
            self(u, v, r1=0, r2=2, cell=cell)
        ])

    def grad3(self, u, v, cell=None) -> np.ndarray:
        from IGAeasy.splines.basis_derivatives import third_derivatives
        return third_derivatives(self, u, v, cell=cell)


    @property
    def knot_average(self) -> typing.Tuple[float, float]:
        """
        Returns the knot average for this BSpline (the Greville point).

        :return: the knot average (u, v).
        """

        if self.p == 0:
            return np.mean(self.knots_u), np.mean(self.knots_v)
        return np.mean(self.knots_u[1:-1]), np.mean(self.knots_v[1:-1])


    def __eq__(self, other: object) -> bool:
        """
        Compare exact polynomial degree and local knot vectors.

        :param other: BSpline to compare against
        :return: true or false
        """
        if not isinstance(other, BSpline):
            return False
        return self.basis_key == other.basis_key

    @property
    def basis_key(self):
        return self._basis_key

    def __hash__(self):
        return hash(self.basis_key)

    def __repr__(self):
        return (f"Bspline(loc_id={self.loc_id}, id={self.id}, active={self.active}, trunc ={self.trunc}, p={self.p}, knots_u={self.knots_u.tolist()}, "
                f"knots_v={self.knots_v.tolist()}, cp={self.cp}, level={self.level})")
