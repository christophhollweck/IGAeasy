"""Pointwise plots of scalar value and gradient errors."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.ticker import Formatter, MaxNLocator
from IGAeasy.error_norms.norms_poisson import eval_element_scalar_field, eval_element_scalar_grad


class SimpleDecimalFormatter(Formatter):
    def __call__(self, x, pos=None):
        if np.isclose(x, 0):
            return "0"
        else:
            s = f"{x:.2f}".rstrip('0').rstrip('.')
            return s

def plot_pointwise_error(RS, exact_solution_func, exact_grad_func=None,
                         type="pointwise_value", relative=False, save_path=None):
    """
    Plot pointwise error at grid points per element.
    Always shows plot window. Optionally saves figure as PDF.

    Parameters
    ----------
    RS : object
        Refined space or mesh object.
    exact_solution_func : callable
        Function u(x, y) for exact solution evaluation.
    exact_grad_func : callable, optional
        Function returning grad_u(x, y) as (du/dx, du/dy).
        Required if type == "pointwise_gradient".
    type : str
        Error type. Use "pointwise_value" or "pointwise_gradient".
    relative : bool
        If True, plot relative error.
    save_path : str or Path, optional
        File path to save the plot. If None, only shows the plot.
    """

    # Compute global L2 norm of exact solution.
    L2_norm_sq = 0.0

    for element in RS.elements:
        if element.void:
            continue

        for gp in element.gausspoints:
            xi, eta = gp.xi, gp.eta
            w = gp.weight_inplane
            u_ex = exact_solution_func(xi, eta)
            L2_norm_sq += u_ex ** 2 * w

    L2_norm = np.sqrt(L2_norm_sq)

    points = []
    errors = []

    x_end = RS.ku[-1]
    y_end = RS.kv[-1]

    for element in RS.elements:
        if element.void:
            continue

        u_vals = np.linspace(element.u_min, element.u_max, 10)
        v_vals = np.linspace(element.v_min, element.v_max, 10)

        for x in u_vals:
            for y in v_vals:
                if abs(x - x_end) < 1e-13:
                    x -= 1e-8

                if abs(y - y_end) < 1e-13:
                    y -= 1e-8

                u_num = eval_element_scalar_field(element, x, y)
                u_ex = exact_solution_func(x, y)

                if type == "pointwise_value":
                    diff = abs(u_num - u_ex)

                    if relative:
                        ref = L2_norm if L2_norm > 1e-13 else 1.0
                        normed_err = diff / ref
                    else:
                        normed_err = diff

                elif type == "pointwise_gradient":
                    if exact_grad_func is None:
                        raise ValueError(
                            "exact_grad_func must be provided for pointwise_gradient error."
                        )

                    grad_num = eval_element_scalar_grad(element, x, y)
                    grad_ex = exact_grad_func(x, y)

                    diff_vec = np.array(grad_num) - np.array(grad_ex)
                    norm_diff = np.linalg.norm(diff_vec)

                    if relative:
                        grad_ref_norm = np.linalg.norm(grad_ex)
                        ref = grad_ref_norm if grad_ref_norm > 1e-13 else 1.0
                        normed_err = norm_diff / ref
                    else:
                        normed_err = norm_diff

                else:
                    raise ValueError(
                        "type must be 'pointwise_value' or 'pointwise_gradient'."
                    )

                points.append([x, y])
                errors.append(normed_err)

    points = np.array(points)
    errors = np.array(errors)

    if points.shape[0] == 0:
        raise ValueError("No points collected for error plotting.")

    # Triangulation based on point coordinates.
    tri = Triangulation(points[:, 0], points[:, 1])

    size = 30

    plt.rcParams.update({
        # Disable external LaTeX rendering.
        # Matplotlib's internal mathtext is sufficient for the labels used here.
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": size,
        "font.size": size,
        "legend.fontsize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
    })

    # Determine scale factor for normalization.
    err_max = errors.max()

    if err_max < 1e-13:
        exponent = 0
    else:
        exponent = int(np.floor(np.log10(err_max)))

    scale_factor = 10 ** exponent
    errors_normed = errors / scale_factor

    from matplotlib.ticker import MaxNLocator

    max_norm = np.max(errors_normed)

    if max_norm < 1e-13:
        max_norm = 1.0

    locator = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5], prune=None)
    cbar_ticks = locator.tick_values(0, max_norm)

    # Keep only valid colorbar ticks.
    cbar_ticks = cbar_ticks[
        (cbar_ticks >= 0.0) & (cbar_ticks <= max_norm + 1e-13)
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    tpc = ax.tricontourf(
        tri,
        errors_normed,
        levels=100,
        cmap="viridis",
    )

    for coll in tpc.collections:
        coll.set_rasterized(True)

    if type == "pointwise_value":
        if relative:
            norm_label = r"$\frac{|u - u_h|}{\|u\|_{L^2}}$"
        else:
            norm_label = r"$|u - u_h|$"
    else:
        if relative:
            norm_label = r"$\frac{\|\nabla u - \nabla u_h\|}{\|\nabla u\|}$"
        else:
            norm_label = r"$\|\nabla u - \nabla u_h\|$"

    cbar = fig.colorbar(
        tpc,
        ax=ax,
        fraction=0.045,
        pad=0.05,
        ticks=cbar_ticks,
    )

    cbar.ax.yaxis.set_major_formatter(SimpleDecimalFormatter())

    # Show scale factor only if exponent != 0.
    if exponent != 0:
        cbar.ax.set_title(
            r"$\times 10^{" + f"{exponent}" + "}$",
            fontsize=30,
            pad=20,
        )
    else:
        cbar.ax.set_title("", fontsize=30, pad=20)

    ax.set_title(norm_label, fontsize=size, pad=30)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    ax.set_aspect("equal")

    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%g"))

    for element in RS.elements:
        if element.void:
            continue

        corners = [
            (element.u_min, element.v_min),
            (element.u_min, element.v_max),
            (element.u_max, element.v_max),
            (element.u_max, element.v_min),
            (element.u_min, element.v_min),
        ]

        x_c, y_c = zip(*corners)

        ax.plot(
            x_c,
            y_c,
            color="white",
            linewidth=0.5,
        )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()
