import numpy as np
import copy


def eval_element_scalar_field(el, xi, eta, field_component=None):
    """
    Evaluate scalar field at (xi, eta) in a given element.
    """
    value = 0.0
    for bf in el.supported_b_splines:
        coefficient = bf.scalar_value if field_component is None else bf.displacements[field_component]
        value += coefficient * bf(xi, eta, cell=el)
    return value


def eval_element_scalar_grad(el, xi, eta, field_component=None):
    """
    Evaluate scalar gradient at (xi, eta) in a given element.
    Returns (du/dxi, du/deta).
    """
    grad = np.zeros(2)
    for bf in el.supported_b_splines:
        coefficient = bf.scalar_value if field_component is None else bf.displacements[field_component]
        grad += coefficient * bf.grad(xi, eta, cell=el)
    return grad


def compute_L2_H1_errors(RS, exact_solution_func, exact_grad_func=None, quadrature_order=None,
                         relative=True, field_component=None):
    """
    Compute physical L2 and H1 errors for a planar XY geometry.

    With exact_grad_func=None, only L2 is computed and H1 is returned as None.
    relative=False returns absolute norms. field_component selects a component
    of bf.displacements; None uses bf.scalar_value.

    :param RS: Refined space or mesh object.
    :param exact_solution_func: Callable u(x, y) in physical coordinates.
    :param exact_grad_func: Callable returning physical gradient (du/dx, du/dy).
    :return: (L2_rel_error, H1_rel_error)
    :param quadrature_order: Optional independent Gauss point count per direction
        for rectangles; triangles use total degree 2*order-2. Existing element
        integration data is preserved.
    """
    L2_error_sq = 0.0
    L2_norm_sq = 0.0
    H1_error_sq = 0.0
    H1_norm_sq = 0.0

    if quadrature_order is not None and (isinstance(quadrature_order, (bool, np.bool_)) or
            not isinstance(quadrature_order, (int, np.integer)) or quadrature_order < 1):
        raise ValueError("quadrature_order must be a positive integer.")
    for element in RS.elements:
        if element.void or not getattr(element, 'active', True):
            continue
        points = element.gausspoints
        if quadrature_order is not None:
            from IGAeasy.numerical_integration.map_gp_to_element import get_untrimmed_element_gp, get_trimmed_element_gp
            temporary = copy.copy(element)
            temporary.p = quadrature_order - 1
            if element.trimmed:
                get_trimmed_element_gp(temporary, full_quad=True, quadrature_degree=2*quadrature_order-2)
            else:
                get_untrimmed_element_gp(temporary, full_quad=True)
            points = temporary.gausspoints
        basis = list(element.supported_b_splines)
        cps = np.array([b.cp for b in basis])
        if cps.shape[1] > 2 and not np.allclose(cps[:, 2], cps[0, 2], rtol=0, atol=1e-14):
            raise ValueError("These error norms require a planar XY geometry; embedded surfaces need surface norms.")
        coefficients = np.array([b.scalar_value if field_component is None else b.displacements[field_component]
                                 for b in basis])
        from IGAeasy.splines.batch_evaluation import evaluate_basis_batch
        batch_values, batch_gradients = evaluate_basis_batch(
            basis, [(gp.xi, gp.eta) for gp in points], element)
        for point_index, gp in enumerate(points):
            xi, eta = gp.xi, gp.eta
            w = gp.weight_inplane

            N = batch_values[point_index]
            u_num = N @ coefficients
            derivatives = batch_gradients[point_index].T
            position = N @ cps[:, :2]
            J = derivatives @ cps[:, :2]
            determinant = np.linalg.det(J)
            if not np.isfinite(determinant) or determinant == 0:
                raise ValueError("Degenerate geometry Jacobian in error integration.")
            w *= abs(determinant)
            u_ex = exact_solution_func(*position)

            # L2 contributions
            diff_u = u_num - u_ex
            L2_error_sq += diff_u ** 2 * w
            L2_norm_sq += u_ex ** 2 * w

            if exact_grad_func is None:
                continue
            grad_num = np.linalg.solve(J, derivatives @ coefficients)
            grad_ex = exact_grad_func(*position)

            # H1 contributions
            diff_grad = np.array(grad_num) - np.array(grad_ex)
            grad_norm_sq = np.dot(grad_ex, grad_ex)
            grad_error_sq = np.dot(diff_grad, diff_grad)

            H1_error_sq += (diff_u ** 2 + grad_error_sq) * w
            H1_norm_sq += (u_ex ** 2 + grad_norm_sq) * w

    if not relative:
        return np.sqrt(L2_error_sq), np.sqrt(H1_error_sq) if exact_grad_func is not None else None
    if L2_norm_sq == 0 or (exact_grad_func is not None and H1_norm_sq == 0):
        raise ValueError("Relative error is undefined for a zero reference norm.")
    L2_rel_error = np.sqrt(L2_error_sq) / np.sqrt(L2_norm_sq)
    H1_rel_error = np.sqrt(H1_error_sq) / np.sqrt(H1_norm_sq) if exact_grad_func is not None else None

    return L2_rel_error, H1_rel_error

def plot_pointwise_error(RS, exact_solution_func, exact_grad_func=None,
                         type="pointwise_value", relative=False, save_path=None):
    """Compatibility entry point; plotting lives in postprocessing."""
    from IGAeasy.postprocessing.error_plots import plot_pointwise_error as plot
    return plot(RS, exact_solution_func, exact_grad_func, type, relative, save_path)
