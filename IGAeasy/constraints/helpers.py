import numpy as np
from shapely.geometry import LineString, LinearRing
import math

def find_control_point_ids_on_edges(RS, xmin=False, xmax=False, ymin=False, ymax=False):
    # Step 1: Find min and max values for x and y among all control points
    control_points = np.array([b.cp for b in RS.basis])
    x_min_val = control_points[:, 0].min()
    x_max_val = control_points[:, 0].max()
    y_min_val = control_points[:, 1].min()
    y_max_val = control_points[:, 1].max()

    control_point_ids = set()  # Set to store the IDs of control points on the edges

    # Step 2: Loop over all control points and check if they are on the specified edges
    for b in RS.basis:
        cp = b.cp  # Access the control point (x, y, z)
        cp_id = b.id  # Access the ID of the control point

        # Initialize a flag to check if the control point is on the edge
        on_edge = False

        # Check for each edge based on the boolean inputs and np.isclose
        if xmin and np.isclose(cp[0], x_min_val):
            on_edge = True
        if xmax and np.isclose(cp[0], x_max_val):
            on_edge = True
        if ymin and np.isclose(cp[1], y_min_val):
            on_edge = True
        if ymax and np.isclose(cp[1], y_max_val):
            on_edge = True

        # If the control point is on one of the edges, add its ID to the set
        if on_edge:
            control_point_ids.add(cp_id)

    return control_point_ids



def find_control_point_ids_in_box(RS,
                                   xmin=None, xmax=None,
                                   ymin=None, ymax=None,
                                   zmin=None, zmax=None):
    """
    Finds control point IDs within a specified bounding box.
    If a bound is None, the corresponding global min/max value is used.

    Parameters:
        RS: Spline or basis object with RS.basis (each with b.cp and b.id)
        xmin, xmax, ymin, ymax, zmin, zmax: float or None

    Returns:
        Set of control point IDs within the box.
    """

    control_points = np.array([b.cp for b in RS.basis])

    # Fallback to global bounds if not specified
    x_min_val = control_points[:, 0].min() if xmin is None else xmin
    x_max_val = control_points[:, 0].max() if xmax is None else xmax
    y_min_val = control_points[:, 1].min() if ymin is None else ymin
    y_max_val = control_points[:, 1].max() if ymax is None else ymax
    z_min_val = control_points[:, 2].min() if zmin is None else zmin
    z_max_val = control_points[:, 2].max() if zmax is None else zmax

    control_point_ids = set()

    for b in RS.basis:
        cp = b.cp
        cp_id = b.id

        if (x_min_val <= cp[0] <= x_max_val and
            y_min_val <= cp[1] <= y_max_val and
            z_min_val <= cp[2] <= z_max_val):
            control_point_ids.add(cp_id)

    return control_point_ids



def get_dofs_from_cps(cps, loc_dofs, dof_per_cp=6):
    from IGAeasy.constraints.prescribed_dofs import control_point_dofs
    return control_point_dofs(cps, loc_dofs, ('x', 'y', 'z', 'rx', 'ry', 'rz'), dof_per_cp)


def split_linestring(line):
    """
    Splits a closed LineString or LinearRing into straight subsegments.
    """
    if line.is_empty:
        return []

    # Convert LinearRing to LineString
    if isinstance(line, LinearRing):
        line = LineString(line)

    coords = list(line.coords)

    # Remove redundant last point if ring is closed
    # if coords[0] == coords[-1] and len(coords) > 2:
    #     coords = coords[:-1]

    return [LineString([coords[i], coords[i+1]]) for i in range(len(coords) - 1)]



def compute_taylor_term(derivative: float, dx: float, n_k: int, dy: float, k: int) -> float:
    """Compute a single term in a 2D Taylor expansion."""
    return derivative * (dx ** n_k) * (dy ** k) / (math.factorial(n_k) * math.factorial(k))

def compute_taylor_expansion_contribution(u, v, cell, dx_vec):
    """
    Computes the Taylor approximation of all supported basis functions at (u + dx, v + dy).

    Parameters:
    - u, v: float, reference coordinates
    - supported_bfs: list of callable basis functions, each bf(u, v, r1, r2) returns ∂^{r1+r2} / ∂xi^{r1} ∂eta^{r2}
    - dx_vec: np.array([dx, dy]), displacement in parametric space
    - basis_function_order: int, maximum derivative order

    Returns:
    - H_sum_vec: np.array, Taylor expansion values for all basis functions at (u + dx, v + dy)
    """
    supported_bfs = list(cell.supported_b_splines)
    supported_bfs.sort(key=lambda bf: bf.id)
    n_nodes = len(supported_bfs)
    H_sum_vec = np.zeros(n_nodes)

    p_min = min([bf.p for bf in supported_bfs])
    n_max = p_min * 2 + 1
    # Zeroth-order term
    N = np.array([bf(u, v, r1=0, r2=0, cell=cell) for bf in supported_bfs])

    for i, bf in enumerate(supported_bfs):

        H_taylor_term = 0.0
        # for n in range(1, n_max):
        # for n in range(1, p_min + 1):
        for n in range(1, p_min * 2+1):
            for k in range(n + 1):  # k = eta derivatives, n-k = xi derivatives
                n_k = n - k
                if n_k > p_min or k > p_min:
                    derivative = 0
                else:
                    derivative = bf(u, v, r1=n_k, r2=k, cell=cell)
                H_taylor_term += compute_taylor_term(derivative, dx_vec[0], n_k, dx_vec[1], k)
        H_sum_vec[i] = H_taylor_term + N[i]

    return H_sum_vec


def compute_gradient_taylor_expansion_contribution(u, v, cell, dx_vec):
    """
    Computes the gradient of the Taylor approximation of all supported basis functions
    at (u + dx, v + dy) with respect to dx and dy.

    Parameters:
    - u, v: float, reference coordinates
    - supported_bfs: list of callable basis functions, each bf(u, v, r1, r2) returns ∂^{r1+r2} / ∂xi^{r1} ∂eta^{r2}
                     and must have attribute 'p' indicating its order.
    - dx_vec: np.array([dx, dy]), displacement in parametric space

    Returns:
    - grad_H_sum: np.array shape (2, n_nodes), gradient of Taylor expansion w.r.t. dx and dy
    """
    supported_bfs = list(cell.supported_b_splines)
    supported_bfs.sort(key=lambda bf: bf.id)
    n_nodes = len(supported_bfs)
    grad_H_sum = np.zeros((2, n_nodes))  # [∂H/∂dx, ∂H/∂dy]

    p_min = min([bf.p for bf in supported_bfs])
    n_max = (p_min * 2 + 1) if p_min in [1, 2] else (p_min + 1)
    # n_max = (2*p_min + 1)

    for i, bf in enumerate(supported_bfs):
        if bf.p != p_min:
            raise ValueError("All basis functions must have the same order for gradient computation. ERROR SBM")
        H_taylor_dx = 0.0
        H_taylor_dy = 0.0

        for n in range(2, n_max):
        # for n in range(2, p_min + 1):
            for k in range(n):  # for ∂^n / ∂xi^{n-k} ∂eta^{k}
                n_k = n - 1 - k
                if n_k+1 > p_min or k > p_min:
                    deriv = 0
                else:
                    deriv = bf(u, v, r1=n_k + 1, r2=k, cell=cell)
                H_taylor_dx += compute_taylor_term(deriv, dx_vec[0], n_k, dx_vec[1], k)

            for k in range(n):  # for ∂^n / ∂xi^{n-k} ∂eta^{k+1}
                n_k = n - 1 - k
                if n_k > p_min or k+1 > p_min:
                    deriv = 0
                else:
                    deriv = bf(u, v, r1=n_k, r2=k + 1, cell=cell)
                H_taylor_dy += compute_taylor_term(deriv, dx_vec[0], n_k, dx_vec[1], k)

        # Add first-order derivatives directly
        grad_H_sum[0, i] = bf(u, v, r1=1, r2=0, cell=cell) + H_taylor_dx
        grad_H_sum[1, i] = bf(u, v, r1=0, r2=1, cell=cell) + H_taylor_dy

    return grad_H_sum
