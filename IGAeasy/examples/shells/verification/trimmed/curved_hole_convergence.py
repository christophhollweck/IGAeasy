"""Mesh convergence of the manufactured curved-hole verification setup."""
import numpy as np
import matplotlib.pyplot as plt
from IGAeasy.examples.shells.verification.trimmed.manufactured_hole_rm5 import run, deformation_scale
from IGAeasy.postprocessing.visualization import plot_solution


def main(show=True, sizes=(4, 8, 16)):
    rows = []
    for n in sizes:
        print(f'RM5, curved hole: {n} x {n}...', flush=True)
        space, row = run(n, show=False)
        rows.append(row)
        print(row, flush=True)
    figure, axis = plt.subplots()
    h = np.array([row['h'] for row in rows])
    axis.loglog(h, [row['l2'] for row in rows], 'o-', label='Displacement L2')
    axis.loglog(h, rows[-1]['l2']*(h/h[-1])**4, 'k--', label='h^4 comparison')
    axis.set(xlabel='Background element size', ylabel='L2 error')
    axis.grid(True, which='both')
    axis.legend()
    if show:
        plt.show()
        plot_solution(space, deformation_scale=deformation_scale)
    else:
        plt.close(figure)
    return rows


if __name__ == '__main__':
    main()
