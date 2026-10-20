"""Nitsche mesh convergence using the complete single-formulation setups."""
import matplotlib.pyplot as plt
from IGAeasy.examples.shells.benchmarks.simply_supported_plate import rm6, rm5, rm5_hierarchical, kirchhoff_love
from IGAeasy.postprocessing.visualization import plot_solution

net_sizes = (2, 4, 8, 16)
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

    figure, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for name, rows in results.items():
        axes[0].plot([r['n'] for r in rows], [r['displacement'] for r in rows], 'o-', label=name)
        axes[1].loglog([r['n'] for r in rows], [r['benchmark_difference'] for r in rows], 'o-', label=name)
        axes[2].loglog([1/r['n'] for r in rows], [r['l2'] for r in rows], 'o-', label=name)
    axes[0].axhline(rm5.reference_displacement, color='gray', linestyle='--', label='RM Navier')
    axes[0].axhline(kirchhoff_love.reference_displacement, color='black', linestyle=':', label='KL Navier')
    for axis, label in zip(axes, ('Benchmark displacement', 'Difference from respective reference')):
        axis.set(xlabel='Elements per direction', ylabel=label)
        axis.grid(True, which='both')
        axis.legend()
    rows = next(iter(results.values()))
    h = [1/row['n'] for row in rows]
    axes[2].loglog(h, [rows[-1]['l2']*(value/h[-1])**4 for value in h], 'k--', label='h^4 comparison')
    axes[2].set(xlabel='Element size h', ylabel='Displacement L2 error')
    axes[2].grid(True, which='both')
    axes[2].legend()
    if show:
        plt.show()
    else:
        plt.close(figure)
    return results


if __name__ == '__main__':
    main()
