import typing
import numpy as np
from scipy.special import roots_legendre
# import gauss_point

class Element:

    def __init__(self, u_min: float, u_max: float, v_min: float, v_max: float, p: int, id: int, level: int = 0) -> None:

        """
        Initialize a rectangular Element defined by corners (u_min, v_min) and (u_max, v_max).

        :param u_min: lower u-bound
        :param u_max: upper u-bound
        :param v_min: lower v-bound
        :param v_max: upper v-bound
        :param p: polynomial degree
        :param id: local element ID
        :param level: refinement level
        """

        self.u_min = u_min
        self.v_min = v_min
        self.u_max = u_max
        self.v_max = v_max
        self.area = (u_max - u_min) * (v_max - v_min)
        self.center = ((u_min + u_max) / 2, (v_min + v_max) / 2)
        self.p = p
        self.level = level
        self.id = None
        self.loc_id = id
        self.active = None
        self.trimmed = False
        self.void = False
        self.children = []
        self.supported_b_splines_init = []
        self.supported_b_splines = []
        self.gausspoints = []
        self.BEO = None
        self.AABB = [[u_min, u_max],[v_min, v_max]]
        self.supported_b_splines_list = []

    def contains(self, u: float, v: float, eps: float = 1e-14) -> bool:
        """
        Returns True if this element contains the point (u, v), including edge tolerance.

        :param u: u_component
        :param v: v_component
        :param eps: tolerance
        """
        return (self.u_min - eps <= u <= self.u_max + eps) and \
            (self.v_min - eps <= v <= self.v_max + eps)

    def __eq__(self, other: 'Element') -> bool:
        """
        Checks whether the two elements are equal within a tolerance.

        :param other: element to compare
        :return: true or false
        """
        tol = 1.0e-14
        return abs(self.u_min - other.u_min) < tol and abs(self.u_max - other.u_max) < tol and abs(
            self.v_min - other.v_min) < tol and abs(self.v_max - other.v_max) < tol

    def __hash__(self):
        return hash(tuple((self.u_min, self.u_max, self.v_min, self.v_max)))

    def __repr__(self):
        return (f"Element(loc_id={self.loc_id}, id={self.id}, level={self.level}, u_min={self.u_min}, u_max={self.u_max}, "
                f"v_min={self.v_min}, v_max={self.v_max}, p={self.p})")

