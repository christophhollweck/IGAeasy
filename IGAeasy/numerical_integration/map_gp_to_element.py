import numpy as np
from scipy.special import roots_legendre
from IGAeasy.numerical_integration import gauss_inplane


# def assign_gp_and_mat(RS, material_model, E, t, rho, full_quad=False):
#     # assign Integrationpoints to elements
#     for element in RS.elements:
#         if element.void == False and element.trimmed == False:
#             get_untrimmed_element_gp(element, full_quad=full_quad)
#         elif element.void == True:
#             continue
#         elif element.trimmed == True:
#             get_trimmed_element_gp(element, full_quad=full_quad)
#         # assign material to IP
#         for gp in element.gausspoints:
#             gp.material_model = material_model
#             gp.t = t
#             gp.E = E
#             gp.rho = rho

def assign_gp_and_properties(RS, material_model=None, properties=None, full_quad=False,
                             quadrature_degree=None, quadrature_order=None):
    """
    Assigns integration points and material/properties to elements.

    :param RS: The geometry or mesh object
    :param material_model: An optional material model object
    :param properties: A dictionary with arbitrary key-value pairs to assign to each Gauss point
    :param full_quad: For rectangles, p+1 points per direction if True,
        p points if False. The same points serve stiffness and mass integration.
        Exactness requires affine geometry and constant material coefficients.
    :param quadrature_degree: Optional total-degree override for triangle integration.
    :param quadrature_order: Optional number of Gauss points per rectangle direction.
    """
    if properties is None:
        properties = {}

    for element in RS.elements:
        if element.void:
            element.gausspoints = []
            continue
        elif element.trimmed:
            get_trimmed_element_gp(element, full_quad=full_quad, quadrature_degree=quadrature_degree,
                                   quadrature_order=quadrature_order)
        else:
            get_untrimmed_element_gp(element, full_quad=full_quad, quadrature_order=quadrature_order)

        for gp in element.gausspoints:
            if material_model is not None:
                gp.material_model = material_model
            for key, value in properties.items():
                setattr(gp, key, value)


def get_untrimmed_element_gp(element, full_quad=False, quadrature_order=None):
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
    # Get Gauss points and weights for 1D integration
    if quadrature_order is not None:
        if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, (int, np.integer)) or quadrature_order < 1:
            raise ValueError('quadrature_order must be a positive integer')
        gauss_points_1d, weights_1d = roots_legendre(quadrature_order)
    elif not full_quad:
        gauss_points_1d, weights_1d = roots_legendre(element.p)
    else:
        gauss_points_1d, weights_1d = roots_legendre(element.p + 1)

    # Transform 1D Gauss points and weights to the interval [a, b]
    def transform_1d(gauss_points, weights, a, b):
        return [(b - a) / 2 * x + (a + b) / 2 for x in gauss_points], [(b - a) / 2 * w for w in weights]

    # Transform 1D Gauss points and weights for each dimension
    AABB = [[element.u_min, element.u_max], [element.v_min, element.v_max]]
    gauss_points_x, weights_x = transform_1d(gauss_points_1d, weights_1d, AABB[0][0], AABB[0][1])
    gauss_points_y, weights_y = transform_1d(gauss_points_1d, weights_1d, AABB[1][0], AABB[1][1])

    # Create 2D Gauss points and weights using tensor product
    gauss_points = [[x, y] for x in gauss_points_x for y in gauss_points_y]
    weights = [wx * wy for wx in weights_x for wy in weights_y]

    element.gausspoints = [gauss_inplane.gauss_point(point[0], point[1], weight)
                          for point, weight in zip(gauss_points, weights)]


def get_trimmed_element_gp(element, full_quad=False, quadrature_degree=None, quadrature_order=None):
    """
    Calculates the Gauss points for trimmed elements based on their material polygons.

    :param element: The element for which Gauss points are calculated
    :param full_quad: If True, uses an extended number of quadrature points
    :return: None, modifies element.gausspoints in-place
    """
    element.gausspoints = []
    area = element.area
    eps = 1e-14
    if np.sum([poly.area for poly in element.material_polygons])/area > 1-eps:
        get_untrimmed_element_gp(element, full_quad=full_quad, quadrature_order=quadrature_order)
        # early return if area ratio > 0.99
        return

    # flag small elements as void and return
    if np.sum([poly.area for poly in element.material_polygons])/area < eps:
        element.void = True
        # Ignore this element for routines but it is not flagged as void for a proper Visualization
        return

    for poly, triangles in zip(element.coarse_material_polygons, element.coarse_material_triangles):
        # if trimmed element is convex and 4 corners
        if quadrature_degree is None and not poly.interiors and len(list(poly.exterior.coords)) == 5 and is_convex(poly):
            gauss_points, weights = map_quad_gp(element, poly, full_quad=full_quad)

            for point, weight in zip(gauss_points, weights):
                element.gausspoints.append(gauss_inplane.gauss_point(point[0], point[1], weight))

        else:
            for triangle in triangles:
                gauss_points, weights = map_triangles_gp(element, triangle, full_quad=full_quad,
                                                        quadrature_degree=quadrature_degree)

                if gauss_points is not None and weights is not None:
                    for point, weight in zip(gauss_points, weights):
                        element.gausspoints.append(gauss_inplane.gauss_point(point[0], point[1], weight))


def map_triangles_gp(element, triangle, full_quad=False, quadrature_degree=None):
    """Integrate a parameter-space triangle with prescribed total-degree accuracy.

    Full integration covers tensor-product mass and stiffness integrands (4p)
    for affine geometry and constant coefficients. Reduced mode covers the
    stiffness degree 4p-2. An explicit degree overrides either setting.
    """
    degree = quadrature_degree if quadrature_degree is not None else (
        4 * element.p if full_quad else max(0, 4 * element.p - 2))
    vertices = np.asarray(triangle.exterior.coords[:-1], dtype=float)
    return get_triangle_quadrature(vertices, degree)

def map_quad_gp(element, polygon, full_quad=False):
    """
    Maps Gauss points from the reference quadrilateral [-1,1] x [-1,1] to a physical quadrilateral.

    :param polygon: List of four corner points defining the quadrilateral [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    :return: List of mapped Gauss points and their corresponding weights
    """

    if polygon.interiors:
        raise ValueError("Quadrilateral mapping cannot integrate polygons with holes; triangulate them first.")
    coords = np.array(polygon.exterior.coords[:-1])  # Remove duplicate last point
    if coords.shape != (4, 2) or not polygon.is_valid or not is_convex(polygon) or polygon.area <= 0:
        raise ValueError("Quadrilateral mapping requires a valid convex four-corner polygon.")

    edges = np.roll(coords, -1, axis=0) - coords
    axis_aligned = np.all((edges[:, 0] == 0) | (edges[:, 1] == 0))
    # A bilinear map gives each reference direction the integrand's total
    # degree; its determinant adds at most one. Axis-aligned rectangles
    # retain the ordinary tensor-product rule.
    n = element.p + int(full_quad) if axis_aligned else 2*element.p + int(full_quad)
    gauss_points_1d, weights_1d = roots_legendre(n)


    # Tensor product to get 2D Gauss points and weights
    gauss_pts = np.array([[xi, eta] for xi in gauss_points_1d for eta in gauss_points_1d])
    weights = np.array([wx * wy for wx in weights_1d for wy in weights_1d])

    def shape_functions(xi, eta):
        """Bilinear shape functions"""
        N1 = 0.25 * (1 - xi) * (1 - eta)
        N2 = 0.25 * (1 + xi) * (1 - eta)
        N3 = 0.25 * (1 + xi) * (1 + eta)
        N4 = 0.25 * (1 - xi) * (1 + eta)
        return np.array([N1, N2, N3, N4])

    def jacobian(xi, eta):
        """Computes the Jacobian determinant at a given Gauss point."""
        dN_dxi = np.array([
            [-0.25 * (1 - eta),  0.25 * (1 - eta),  0.25 * (1 + eta), -0.25 * (1 + eta)],
            [-0.25 * (1 - xi),  -0.25 * (1 + xi),   0.25 * (1 + xi),  0.25 * (1 - xi)]
        ])
        J = dN_dxi @ coords  # 2x2 Jacobian matrix
        return np.linalg.det(J)

    mapped_points = []
    weighted_weights = []

    for xi, eta, w in zip(gauss_pts[:, 0], gauss_pts[:, 1], weights):
        N = shape_functions(xi, eta)
        x_mapped = np.dot(N, coords[:, 0])
        y_mapped = np.dot(N, coords[:, 1])
        mapped_points.append([x_mapped, y_mapped])

        # Compute Jacobian determinant and weight the quadrature weight
        detJ = jacobian(xi, eta)
        weighted_weights.append(w * abs(detJ))  # Ensure positive weights

    return np.array(mapped_points), np.array(weighted_weights)


def triangle_area(triangle):
    """Computes the area of a triangle."""
    A, B, C = np.array(triangle)
    return 0.5 * abs(np.linalg.det([[B[0] - A[0], C[0] - A[0]], [B[1] - A[1], C[1] - A[1]]]))


def get_triangle_quadrature(triangle, poly_degree):
    """Positive Gauss-product rule via a Duffy map, exact to total degree d."""
    if isinstance(poly_degree, (bool, np.bool_)) or not isinstance(poly_degree, (int, np.integer)) or poly_degree < 0:
        raise ValueError("Quadrature degree must be a nonnegative integer.")
    vertices = np.asarray(triangle, dtype=float)
    if vertices.shape != (3, 2) or not np.all(np.isfinite(vertices)):
        raise ValueError("Expected three finite 2D triangle vertices.")
    A, B, C = vertices
    det = abs(np.linalg.det(np.column_stack((B-A, C-A))))
    if det == 0:
        raise ValueError("Cannot integrate a degenerate triangle.")
    # Under r=a, s=(1-a)*b, the Jacobian adds one degree in a.
    a, wa = roots_legendre((poly_degree + 3) // 2)
    b, wb = roots_legendre((poly_degree + 2) // 2)
    a, b, wa, wb = (a+1)/2, (b+1)/2, wa/2, wb/2
    r, t = np.meshgrid(a, b, indexing='ij')
    s = (1-r)*t
    points = A + r.reshape(-1, 1)*(B-A) + s.reshape(-1, 1)*(C-A)
    weights = (wa[:, None]*wb[None, :]*(1-r)*det).ravel()
    return points, weights

def barycentric_to_cartesian(triangle, bary):
    """Converts barycentric coordinates to Cartesian coordinates."""
    A, B, C = np.array(triangle)
    l1, l2, l3 = bary
    return l1 * A + l2 * B + l3 * C

def is_convex(polygon):
    """Checks if the polygon is convex."""
    coords = list(polygon.exterior.coords)
    n = len(coords)
    sign = 0

    for i in range(n - 2):
        # Calculate the vectors for adjacent edges
        p1, p2, p3 = np.array(coords[i]), np.array(coords[i + 1]), np.array(coords[i + 2])
        # Calculate the cross product of vectors p1p2 and p2p3
        cross_product = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])

        # Check the sign of the cross product
        if cross_product > 0:
            current_sign = 1
        elif cross_product < 0:
            current_sign = -1
        else:
            current_sign = 0

        # If the sign changes, the polygon is not convex
        if sign == 0:
            sign = current_sign
        elif sign != current_sign and current_sign != 0:
            return False

    return True






