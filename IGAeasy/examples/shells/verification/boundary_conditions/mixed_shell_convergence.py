"""Doubly curved shells with weak displacement and natural director moments."""
import matplotlib.pyplot as plt

from IGAeasy.examples.shells.verification.manufactured.paraboloid_convergence import solve_level
from IGAeasy.postprocessing.visualization import plot_solution


# --- Geometry & discretization ---
curvature = (.4, -.2)
mixed_terms = (1.2, -.8)
net_sizes = (2, 4, 8, 16)
formulations = ('RM5', 'RM5_hier', 'KL')
methods = ('nitsche', 'penalty')
penalty_factor = 100.  # Fixed multiplier: penalty has a finite boundary approximation error.
domain = None
# from IGAeasy.examples.shells.verification.trimmed.all_formulations_convergence import domain
local_refinements = 0


def main(show=True, sizes=net_sizes, trim_domain=domain, local_steps=local_refinements, support='mixed'):
    # --- Analytical field and matching natural moments on every boundary ---
    results = {}
    for formulation in formulations:
        for method in methods:
            rows = []
            for n in sizes:
                print(f'{formulation}, {method}: {n} x {n}...', flush=True)
                space, row = solve_level(
                    n, formulation, boundary_method=method, boundary_kind=support,
                    surface_curvature=curvature, mixed=mixed_terms,
                    director_amplitude=.0003, enriched=True, domain=trim_domain,
                    local_refinements=local_steps, refinement_region=(0., 0., .2, .2),
                    penalty_factor=penalty_factor
                )
                rows.append(row)
                print(f'  L2={row["l2"]:.6e}, director L2={row["director_l2"]:.6e}, '
                      f'residual={row["residual"]:.3e}', flush=True)
            results[f'{formulation} {method}'] = rows

    # --- Physical displacement and director errors ---
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    figure.suptitle('Normal displacement supports' if support == 'normal'
                   else 'Weak displacements and natural director moments')
    for axis, key in zip(axes, ('l2', 'director_l2')):
        for name, rows in results.items():
            axis.loglog([r['h'] for r in rows], [r[key] for r in rows], 'o-', label=name)
        axis.set(xlabel='Background element size h', ylabel=key)
        axis.grid(True, which='both')
        axis.legend()
    if show:
        plt.show()

        # --- Last computed displacement field ---
        plot_solution(space, deformation_scale=50.)
    return results


if __name__ == '__main__':
    main()
