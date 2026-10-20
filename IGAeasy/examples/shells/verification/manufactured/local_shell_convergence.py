"""Check manufactured shell solutions under repeated local refinement."""
import matplotlib.pyplot as plt

from IGAeasy.examples.shells.verification.manufactured.paraboloid_convergence import solve_level
from IGAeasy.postprocessing.visualization import plot_solution


# --- Geometry & discretization ---
curvature = (.4, -.2)
mixed_terms = (1.2, -.8)
initial_elements = 8
refinement_steps = (0, 1, 2)
refinement_region = (.4, .4, .6, .6)
families = ('THB', 'LR')
formulations = ('RM5', 'RM5_hier', 'KL')

domain = None
# Uncomment to include the moderate outer trim and inner hole.
# from IGAeasy.examples.shells.verification.trimmed.all_formulations_convergence import domain
deformation_scale = 50.


def main(show=True):
    # --- Solve each mesh independently with the same physical reference ---
    results = {}
    for family in families:
        for formulation in formulations:
            rows = []
            for step in refinement_steps:
                print(f'{family} {formulation}: local refinement {step}...', flush=True)
                space, row = solve_level(
                    initial_elements, formulation, family=family,
                    surface_curvature=curvature, mixed=mixed_terms,
                    director_amplitude=.0003, enriched=True, domain=domain,
                    stabilization='split', local_refinements=step,
                    refinement_region=refinement_region
                )
                rows.append(row)
                print(f'  basis={row["n_basis"]}, cells={row["n_elements"]}, '
                      f'L2={row["l2"]:.6e}, director L2={row["director_l2"]:.6e}, '
                      f'residual={row["residual"]:.3e}', flush=True)
            results[f'{family} {formulation}'] = rows

    # --- Error versus problem size ---
    # There is no single shrinking h: the unmarked part of the mesh stays coarse.
    figure, axis = plt.subplots()
    for name, rows in results.items():
        axis.loglog([r['n_basis'] for r in rows], [r['l2'] for r in rows], 'o-', label=name)
    axis.set(xlabel='Number of active basis functions', ylabel='Displacement L2 error',
             title='Local shell refinement')
    axis.grid(True, which='both')
    axis.legend()

    if show:
        plt.show()

        # --- Last computed mesh and physical displacement field ---
        plot_solution(space, deformation_scale=deformation_scale)
    return results


if __name__ == '__main__':
    main()
