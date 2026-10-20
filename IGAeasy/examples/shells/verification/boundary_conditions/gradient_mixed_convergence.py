"""Curved KL shell with weak displacements and natural normal moments."""
import numpy as np
import matplotlib.pyplot as plt
from IGAeasy.manufactured_solutions.shell_verification import solve_gradient_patch as solve_level

n = 2
p = 3
n_refinements = 3
family = 'THB'
trimmed = True
angle = .6
curvature = .2
methods = ['nitsche', 'penalty']
normal_rotation = False


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    comparison = None
    geometry_shown = False
    for method in methods:
        rows = []
        for level in range(n_refinements+1):
            current = n*2**level
            print(f'{method}: {current} x {current}...', flush=True)
            RS, _, row = solve_level(current, family=family, trimmed=trimmed,
                angle=angle, curvature=curvature, degree=p, boundary_kind='mixed',
                normal_rotation=normal_rotation, method=method,
                safety_factor=400. if method == 'penalty' else 4.)
            rows.append(row)
            print(f"L2={row['l2']:.6e}, director L2={row['director_l2']:.6e}, residual={row['residual']:.3e}", flush=True)
            if not geometry_shown:
                from IGAeasy.postprocessing.visualization import plot_solution
                plot_solution(RS, deformation_scale=50.)
                geometry_shown = True
        if comparison is None:
            comparison = rows
        for ax, key in zip(axes, ('l2', 'director_l2')):
            ax.loglog([r['h'] for r in rows], [r[key] for r in rows], 'o-', label=method)
    h = np.array([r['h'] for r in comparison])
    for ax, key, order in zip(axes, ('l2', 'director_l2'), (min(p+1, 2*p-2), p)):
        ax.loglog(h, comparison[-1][key]*(h/h[-1])**order, '--', label=f'h^{order} comparison')
        ax.set_xlabel('Element size h')
        ax.set_ylabel(key)
        ax.grid(True, which='both')
        ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
