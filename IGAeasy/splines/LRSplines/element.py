import numpy as np
from scipy.special import roots_legendre

class Element:

    def __init__(self, u_min: float, u_max: float, v_min: float, v_max: float, p: int, id: int, level: int = 0) -> None:
        """
        Initialize an Element (a rectangle) with lower left corner (u_min, v_min)
        and upper right corner (u_max, v_max)

        :param u_min: lower left u_component
        :param v_min: lower left v_component
        :param u_max: upper right u_component
        :param v_max: upper right v_component
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
        self.trimmed = False
        self.void = False
        self.supported_b_splines = []
        self.gausspoints = []
        self.BEO = None



    def contains(self, u: float, v: float) -> bool:
        """
        Returns True if this element contains the point (u, v)

        :param u: u_component
        :param v: v_component
        :return:
        """

        return self.u_min <= u <= self.u_max and self.v_min <= v <= self.v_max

    def add_supported_b_spline(self, b_spline):
        """
        Adds a B-spline to the list of supported B-splines.

        :param b_spline: B-spline to add
        """
        self.supported_b_splines.append(b_spline)


    def intersects(self, other: 'Element') -> bool:
        """
        Returns true if this element intersects the other element with *positive* area.

        :param other: the element to check intersection with.
        :return: true or false
        """
        intersection_u = min(other.u_max, self.u_max) - max(other.u_min, self.u_min)
        intersection_v = min(other.v_max, self.v_max) - max(other.v_min, self.v_min)

        if intersection_u <= 0 or intersection_v <= 0:
            return False
        else:
            return True

    def split(self, axis: int, split_value: float) -> ["Element", "Element"]:
        """
        Splits the element into two, resizing into the left half, and returning both the left and right halves.

        :return: a tuple with the left and right half of the element.
        """

        # Create new elements for the split
        left_element = None
        right_element = None

        if axis == 0:  # vertical split
            if not self.u_min < split_value < self.u_max:
                return None, None  # No split if the split_value is out of bounds

            # Create the right element (after the split_value)
            right_element = Element(split_value, self.u_max, self.v_min, self.v_max, self.p, self.id,
                                    level=self.level + 0.5)
            # Create the left element (before the split_value)
            left_element = Element(self.u_min, split_value, self.v_min, self.v_max, self.p, self.id,
                                   level=self.level + 0.5)

        elif axis == 1:  # horizontal split
            if not self.v_min < split_value < self.v_max:
                return None, None  # No split if the split_value is out of bounds

            # Create the top element (after the split_value)
            right_element = Element(self.u_min, self.u_max, split_value, self.v_max, self.p, self.id,
                                    level=self.level + 0.5)
            # Create the bottom element (before the split_value)
            left_element = Element(self.u_min, self.u_max, self.v_min, split_value, self.p, self.id,
                                   level=self.level + 0.5)

        # Return both the left and right elements
        return left_element, right_element

    def get_AABB(self):
        return [[self.u_min, self.u_max], [self.v_min, self.v_max]]

    def get_gp_gw_2d(self, full_quad=False):
        """
        calculates the Gauss quadrature points and weights for numerical integration on an untrimmed 2D
        cell defined by an AABB.
        As the weights are also transformed, the determinat of the mapping btw. the parameter and gauss space
        ( 0.25 * (xi_i+1 - xi_i) * (eta_i+1 - eta_i) is already included within the weights and must not be further
        included in the numerical integration procedure)

        :param
        - aabb: Axis-Aligned Bounding Box, a 2D list [[xmin, xmax], [ymin, ymax]]
        - num_gauss_points: Number of Gauss points for each dimension
        :return
        - gauss_points: List of 2D Gauss points in the form [[x1, y1], [x2, y2], ...]
        - weights: List of corresponding Gauss weights
        """
        #self.AABB = [[self.u_min, self.u_max], [self.v_min, self.v_max]]
        # Get Gauss points and weights for 1D integration
        if not full_quad:
            gauss_points_1d, weights_1d = roots_legendre(self.p)
        else:
            gauss_points_1d, weights_1d = roots_legendre(self.p+1)

        # Transform 1D Gauss points and weights to the interval [a, b]
        def transform_1d(gauss_points, weights, a, b):
            return [(b - a) / 2 * x + (a + b) / 2 for x in gauss_points], [(b - a) / 2 * w for w in weights]

        # Transform 1D Gauss points and weights for each dimension
        AABB = [[self.u_min, self.u_max], [self.v_min, self.v_max]]
        gauss_points_x, weights_x = transform_1d(gauss_points_1d, weights_1d, AABB[0][0], AABB[0][1])
        gauss_points_y, weights_y = transform_1d(gauss_points_1d, weights_1d, AABB[1][0], AABB[1][1])

        # Create 2D Gauss points and weights using tensor product
        gauss_points = [[x, y] for x in gauss_points_x for y in gauss_points_y]
        weights = [wx * wy for wx in weights_x for wy in weights_y]

        for point, weight in zip(gauss_points, weights):
            point_with_weight = point + [weight]
            self.gausspoints.append(gauss_point(point[0], point[1], weight))


    def __eq__(self, other: 'Element') -> bool:
        """
        Checks whether the two elements are equal within a tolerance.

        :param other: element to compare
        :return: true or false
        """
        tol = 1.0e-10
        return abs(self.u_min - other.u_min) < tol and abs(self.u_max - other.u_max) < tol and abs(
            self.v_min - other.v_min) < tol and abs(self.v_max - other.v_max) < tol

    def __hash__(self):
        return hash(tuple((self.u_min, self.u_max, self.v_min, self.v_max)))

    def __repr__(self):
        return (f"id={self.id}, u_min={self.u_min}, u_max={self.u_max}, "
                f"v_min={self.v_min}, v_max={self.v_max}, p={self.p}), level={self.level})")

