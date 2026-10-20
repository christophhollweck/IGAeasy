"""The same saddle rectangle represented by a fitted patch and a trimmed patch."""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from IGAeasy.examples.shells.verification.trimmed.manufactured_hole_rm5 import run
from IGAeasy.postprocessing.visualization import plot_solution

net_sizes = (4, 8, 16)
domain = box(1/12, 1/12, 11/12, 11/12)


def main(show=True, sizes=net_sizes):
    results = {}
    figure, axis = plt.subplots()
    for fitted in (False, True):
        name = 'Fitted patch' if fitted else 'Trimmed background'
        rows = []
        for n in sizes:
            # Original parameter coordinates are preserved: geometry and loads coincide.
            space, row = run(n, show=False, material_domain=domain, fitted_rectangle=fitted)
            rows.append(row)
            print(name, row, flush=True)
            if show and n == sizes[0] and not fitted:
                plot_solution(space, deformation_scale=50.)
        results[name] = rows
        axis.loglog([row['h'] for row in rows], [row['l2'] for row in rows], 'o-', label=name)
    h = np.array([row['h'] for row in rows])
    axis.loglog(h, rows[-1]['l2']*(h/h[-1])**4, 'k--', label='h^4 comparison')
    axis.set(xlabel='Element size in original parameter coordinates', ylabel='Displacement L2 error')
    axis.grid(True, which='both')
    axis.legend()
    if show:
        plt.show()
    else:
        plt.close(figure)
    return results


if __name__ == '__main__':
    main()
