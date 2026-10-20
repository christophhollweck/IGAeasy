"""Nitsche mesh convergence using the complete single-formulation setups."""
import matplotlib.pyplot as plt
from IGAeasy.examples.shells.benchmarks.pinched_hemisphere import rm6, rm5, rm5_hierarchical, kirchhoff_love
from IGAeasy.postprocessing.visualization import plot_solution

net_sizes = (8, 16, 32)
# RM point-load responses are finite-mesh comparisons, not finite-limit references.
setups = {'RM6': rm6, 'RM5': rm5, 'RM5_hier': rm5_hierarchical, 'KL': kirchhoff_love}


def main(show=True, sizes=net_sizes):
    results = {}
    geometry_shown = False
    for name, setup in setups.items():
        rows = []
        for n in sizes:
            print(f'{name}, Nitsche: {n} x {n}...', flush=True)
            space, row = setup.run(n_elements=n, method='nitsche', show=False)
            rows.append(row)
            print(row, flush=True)
            if show and not geometry_shown:
                plot_solution(space, deformation_scale=setup.deformation_scale)
                geometry_shown = True
        results[name] = rows

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for name, rows in results.items():
        axes[0].plot([r['n'] for r in rows], [r['radial_a'] for r in rows], 'o-', label=name)
        axes[1].loglog([r['n'] for r in rows], [r['benchmark_difference'] for r in rows], 'o-', label=name)
    axes[0].axhline(rm5.reference_displacement, color='gray', linestyle='--', label='Published KL extrapolation')
    for axis, label in zip(axes, ('Radial displacement', 'Difference from published KL value')):
        axis.set(xlabel='Elements per direction', ylabel=label)
        axis.grid(True, which='both')
        axis.legend()
    if show:
        plt.show()
    else:
        plt.close(figure)
    return results


if __name__ == '__main__':
    main()
