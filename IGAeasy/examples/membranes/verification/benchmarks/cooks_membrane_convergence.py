"""Compare fitted and trimmed Cook panels against a rounded numerical reference band."""
import matplotlib.pyplot as plt
from IGAeasy.examples.membranes.benchmarks import cooks_membrane as fitted
from IGAeasy.examples.membranes.trimmed import cooks_membrane as trimmed


def main(show=True, sizes=(4, 8, 16)):
    figure, axis = plt.subplots()
    results = {}
    for name, setup in (('Fitted', fitted), ('Trimmed', trimmed)):
        rows = []
        for n in sizes:
            _, row = setup.run(n, show=False)
            rows.append(row)
            print(name, row, flush=True)
        results[name] = rows
        axis.plot(sizes, [row['normalized_displacement'] for row in rows], 'o-', label=name)
    axis.axhspan(23.90, 23.96, color='gray', alpha=.2, label='Rounded numerical reference range')
    axis.set(xlabel='Elements per parameter direction', ylabel='Midpoint u_y E t / P',
             title="Cook's membrane: same physical panel, different meshes")
    axis.grid(True)
    axis.legend()
    if show:
        plt.show()
    else:
        plt.close(figure)
    return results


if __name__ == '__main__':
    main()
