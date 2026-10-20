"""Convergence study for manufactured curved rm shells."""
import numpy as np
import matplotlib.pyplot as plt
from IGAeasy.manufactured_solutions.shell_verification import solve_rm_patch as solve_level

p = 3
n = 2
n_refinements = 3
family = 'THB'
formulations = ['RM6', 'RM5']
methods = ['nitsche', 'penalty']
trimmed = True
local_refinement = False
angle = .6
curvature = .2
penalty_multiplier = 100.
director_amplitude = .002




def main(show=True, *, formulations=formulations, methods=methods, family=family,
         n_refinements=n_refinements, trimmed=trimmed):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    results = []
    geometry_shown = False
    comparison = None
    for formulation in formulations:
        for method in methods:
            rows = []
            for level in range(n_refinements+1):
                current = n*2**level
                print(f'{formulation}, {method}: {current} x {current}...', flush=True)
                RS, _, row = solve_level(current, formulation, method, family, p,
                                         curvature=curvature, angle=angle, trimmed=trimmed,
                                         local_refinement=local_refinement,
                                         director_amplitude=director_amplitude,
                                         penalty_multiplier=penalty_multiplier)
                rate = np.log2(rows[-1]['l2']/row['l2']) if rows else np.nan
                rows.append(row)
                print(f"L2={row['l2']:.3e}, director L2={row['director_l2']:.3e}, rate={rate:.2f}", flush=True)
                if not geometry_shown:
                    from IGAeasy.postprocessing.visualization import plot_solution
                    plotter = plot_solution(RS, deformation_scale=50., show=show, off_screen=not show)
                    if not show:
                        plotter.render()
                        plotter.close()
                    geometry_shown = True
            if comparison is None:
                comparison = rows
            results.extend(dict(row, formulation=formulation, method=method) for row in rows)
            for ax, key in zip(axes, ('l2', 'director_l2')):
                ax.loglog([row['h'] for row in rows], [row[key] for row in rows],
                          'o-', label=f'{formulation}, {method}')
    h = np.array([row['h'] for row in comparison])
    for ax, key in zip(axes, ('l2', 'director_l2')):
        ax.loglog(h, comparison[-1][key]*(h/h[-1])**(p+1), '--', label=f'h^{p+1} comparison')
    for ax, label in zip(axes, ('Displacement L2 error', 'Physical director L2 error')):
        ax.set_xlabel('Element size h')
        ax.set_ylabel(label)
        ax.grid(True, which='both')
        ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    else:
        fig.canvas.draw()
    return results


if __name__ == '__main__':
    main()
