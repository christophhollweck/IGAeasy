import typing
import numpy as np
from IGAeasy.splines.basis_identity import basis_key
from IGAeasy.splines.control_point_fields import control_point_fields
from IGAeasy.splines.univariate import _evaluate_univariate_b_spline as _shared_evaluate

Vector = typing.List[float]


def _evaluate_univariate_b_spline(x, knots, degree, r=0, endpoint=None, knotspan_norm=None):
    """Compatibility wrapper preserving LR's positional derivative argument."""
    if endpoint is None:
        endpoint = bool(np.all(np.asarray(knots)[-(degree + 1):] == knots[-1]))
    return _shared_evaluate(x, knots, degree, endpoint=endpoint, r=r,
                            knotspan_norm=knotspan_norm)

class BSpline:
    """
    Represents a single weighted tensor product B-spline with associated methods and fields.
    """

    def __init__(self, p: int, knots_u: Vector, knots_v: Vector, cp: Vector, weight: float = 1) -> None:

        self.p = p
        self.knots_u = np.array(knots_u, dtype=np.float64)
        self.norm_knots_u = self._normalize_knots(self.knots_u)
        self.knots_v = np.array(knots_v, dtype=np.float64)
        self.norm_knots_v = self._normalize_knots(self.knots_v)
        self.knots_u.setflags(write=False)
        self.knots_v.setflags(write=False)
        self._basis_key = basis_key(p, self.knots_u, self.knots_v)
        self.weight = weight
        self.cp = np.array(cp, dtype=np.float64)
        self.displacements = np.array([0.0,0.0,0.0], dtype=np.float64)
        self.scalar_value = 0.0
        self.elements_of_support = []
        self.level = None
        self.id = None
        self.trimmed = False
        self.void = False
        self._hash = self._compute_hash()

    def _normalize_knots(self, knots: np.ndarray) -> np.ndarray:
        """
        Normalize the knot vector so that it starts from 0.

        :param knots: Original knot vector as a numpy array
        :return: Normalized knot vector as a numpy array
        """
        return knots - knots[0]

    def get_AABB(self):
        u_min, u_max = self.knots_u[0], self.knots_u[-1]
        v_min, v_max = self.knots_v[0], self.knots_v[-1]
        return [[u_min, u_max], [v_min, v_max]]

    def __call__(self, u: float, v: float, r1=0, r2=0, cell=None) -> float:
        """
        Evaluates the BSpline basis function at the parametric point (u, v).

        :param u: u component
        :param v: v component
        :return: B(u, v)
        """
        u_norm = u - self.knots_u[0]
        v_norm = v - self.knots_v[0]
        span_u = None if cell is None else (cell.u_min - self.knots_u[0], cell.u_max - self.knots_u[0])
        span_v = None if cell is None else (cell.v_min - self.knots_v[0], cell.v_max - self.knots_v[0])
        return self.weight * _evaluate_univariate_b_spline(
            u_norm, self.norm_knots_u, self.p, r1, knotspan_norm=span_u
        ) * _evaluate_univariate_b_spline(
            v_norm, self.norm_knots_v, self.p, r2, knotspan_norm=span_v
        )


    def grad(self, X, Y, cell=None):
        return np.array([self(X, Y, r1=1, r2=0, cell=cell), self(X, Y, r1=0, r2=1, cell=cell)])

    def grad_mix(self, X, Y, cell=None):
        return self(X, Y, r1=1, r2=1, cell=cell)

    def grad2(self, X, Y, cell=None):
        return np.array([self(X, Y, r1=2, r2=0, cell=cell), self(X, Y, r1=0, r2=2, cell=cell)])

    def grad3(self, X, Y, cell=None):
        from IGAeasy.splines.basis_derivatives import third_derivatives
        return third_derivatives(self, X, Y, cell=cell)

    def add_to_support_if_intersects(self, element: "Element") -> bool:
        """
        Returns true if the given element intersects the support of this BSpline, and
        adds element to the list of elements of support.

        :param element: element in consideration
        :return: true or false
        """

        if self.intersects(element):
            self.elements_of_support.append(element)
            return True
        else:
            return False


    def intersects(self, element: "Element") -> bool:
        """
        Returns true if the support of b_spline intersects the element with positive area.

        :param b_spline: b_spline whose support is to be checked
        :param element: element whose domain is to be checked
        :return: true or false
        """

        intersection_u = min(self.knots_u[-1], element.u_max) - max(self.knots_u[0], element.u_min)
        intersection_v = min(self.knots_v[-1], element.v_max) - max(self.knots_v[0], element.v_min)

        if intersection_u <= 0 or intersection_v <= 0:
            return False
        else:
            return True

    @property
    def knot_average(self) -> typing.Tuple[float, float]:
        """
        Returns the knot average for this BSpline (the Greville point).

        :return: the knot average (u, v).
        """

        if self.p == 0:
            return np.mean(self.knots_u), np.mean(self.knots_v)
        return np.mean(self.knots_u[1:-1]), np.mean(self.knots_v[1:-1])

    def update_weights(self, other: "BSpline") -> None:
        """
        Updates the weights during splitting.
        """
        w = self.weight + other.weight
        for field, shape in control_point_fields([self, other]).items():
            value = (np.asarray(getattr(self, field, np.zeros(shape))) * self.weight +
                     np.asarray(getattr(other, field, np.zeros(shape))) * other.weight) / w
            setattr(self, field, float(value) if not shape else value.copy())
        self.weight = w

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

    def _compute_hash(self):
        return hash(self.basis_key)

    def __hash__(self):
        """
        Return the precomputed hash value.
        """
        return self._hash

    def __repr__(self):
        return (f"Bspline(id={self.id}, p={self.p}, knots_u={self.knots_u.tolist()}, "
                f"knots_v={self.knots_v.tolist()}, cp={self.cp}, weight={self.weight}, level={self.level})")
