"""Trimmed flat cantilever against its analytical RM tip and displacement field."""
import matplotlib.pyplot as plt
from IGAeasy.examples.shells.trimmed.flat_cantilever_rm5 import run


def main(show=True, sizes=(4, 8, 16)):
    rows = []
    for n in sizes:
        _, row = run(n, show=show and n == sizes[-1])
        rows.append(row)
        print(row, flush=True)
    figure, axis = plt.subplots()
    axis.loglog([1/row['n'] for row in rows], [row['absolute_error'] for row in rows],
                'o-', label='Tip error')
    axis.loglog([1/row['n'] for row in rows], [row['l2'] for row in rows],
                's-', label='Displacement L2')
    axis.set(xlabel='Background mesh size', ylabel='Absolute error',
             title='Exact cubic field: errors are limited by roundoff, not mesh size')
    axis.grid(True, which='both')
    axis.legend()
    if show:
        plt.show()
    else:
        plt.close(figure)
    return rows


if __name__ == '__main__':
    main()
