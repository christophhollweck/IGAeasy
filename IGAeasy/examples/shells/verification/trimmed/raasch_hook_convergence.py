"""Width-trimmed Raasch hook: RM5 versus a numerical 3D benchmark."""
import matplotlib.pyplot as plt
from IGAeasy.examples.shells.trimmed.raasch_hook_rm5 import run
from IGAeasy.examples.shells.benchmarks.raasch_hook.rm5 import run as run_fitted


def main(show=True, levels=(0, 1, 2)):
    rows = []
    figure, axis = plt.subplots()
    for name, setup in (('Trimmed RM5', run), ('Fitted RM5', run_fitted)):
        values = []
        for level in levels:
            _, row = setup(level, show=False)
            row['discretization'] = name
            rows.append(row)
            values.append(row['displacement'])
            print(row, flush=True)
        axis.plot(levels, values, 'o-', label=name)
    axis.axhline(rows[0]['reference_displacement'], color='black', linestyle='--',
                 label='Numerical 3D reference')
    axis.set(xlabel='Global refinement level', ylabel='Loaded-edge midpoint displacement')
    axis.grid(True)
    axis.legend()
    if show:
        plt.show()
    else:
        plt.close(figure)
    return rows


if __name__ == '__main__':
    main()
