"""Convergence study for manufactured curved gradient shells."""
import numpy as np
import matplotlib.pyplot as plt
from IGAeasy.manufactured_solutions.shell_verification import solve_gradient_patch as solve_level


p = 3
n = 2
n_refinements = 3
family = 'THB'
trimmed = True
angle = .6
formulation = 'KL'
curvature = .2
stabilization = 'scalar'
reference_kind = 'parabolic'
boundary_method = 'nitsche'
boundary_safety_factor = None
boundary_kind = 'clamped'




def main(show=True, *, n=n, p=p, n_refinements=n_refinements, family=family,
         trimmed=trimmed, angle=angle, formulation=formulation, boundary_kind=boundary_kind,
         boundary_method=boundary_method, boundary_safety_factor=boundary_safety_factor):
    rows = []
    for level in range(n_refinements+1):
        current = n*2**level
        print(f'Computing {current} x {current}...', flush=True)
        RS, solution, row = solve_level(current, family, trimmed, angle, formulation=formulation,
                                       curvature=curvature, stabilization=stabilization,
                                       reference_kind=reference_kind, degree=p, method=boundary_method,
                                       safety_factor=boundary_safety_factor, boundary_kind=boundary_kind)
        rows.append(row)
        rate = np.log2(rows[-2]['l2']/row['l2']) if level else np.nan
        print(f"h={row['h']:.5f}, L2={row['l2']:.6e}, director L2={row['director_l2']:.6e}, rate={rate:.3f}, gamma_u={row['gamma_u_max']:.3e}, gamma_d={row['gamma_d_max']:.3e}, residual={row['residual']:.3e}", flush=True)
        if level == 0:
            from IGAeasy.postprocessing.visualization import plot_solution
            plotter = plot_solution(RS, deformation_scale=50., show=show, off_screen=not show)
            if not show:
                plotter.render()
                plotter.close()
    h = np.array([row['h'] for row in rows])
    errors = np.array([row['l2'] for row in rows])
    director_errors = np.array([row['director_l2'] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].loglog(h, errors, 'o-', label=f'{formulation} {boundary_method}')
    axes[0].loglog(h, errors[-1]*(h/h[-1])**(p+1), '--',
                   label=f'Expected h^{p+1}' if formulation == 'KL' else f'h^{p+1} comparison')
    axes[0].set_ylabel('Physical displacement L2 error')
    axes[1].loglog(h, director_errors, 'o-', label=f'{formulation} {boundary_method}')
    axes[1].loglog(h, director_errors[-1]*(h/h[-1])**p, '--', label=f'h^{p} comparison')
    axes[1].set_ylabel('Physical director L2 error')
    for axis in axes:
        axis.set_xlabel('Element size h')
        axis.grid(True, which='both', alpha=.3)
        axis.legend()
    fig.tight_layout()
    if show:
        plt.show()
    else:
        fig.canvas.draw()
    return rows


if __name__ == '__main__':
    main()
