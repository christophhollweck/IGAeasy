import numpy as np
from scipy.interpolate import BSpline
from scipy.special import roots_legendre


def gradient_space_diagnostic(n_elements=4, p=3):
    """One-dimensional probe of eta=grad(w), the bending-free shear mode.

    This tests a discrete space property, not the full shell convergence order.
    """
    if not isinstance(n_elements, int) or n_elements < 2 or not isinstance(p, int) or p < 2:
        raise ValueError('Use at least two elements and degree at least two')
    breaks = np.linspace(0., 1., n_elements+1)
    knots = np.r_[np.zeros(p+1), breaks[1:-1], np.ones(p+1)]
    count = len(knots)-p-1
    nodes, weights = roots_legendre(p+2)
    points = np.concatenate([a+(nodes+1)*(b-a)/2 for a, b in zip(breaks[:-1], breaks[1:])])
    weights = np.concatenate([weights*(b-a)/2 for a, b in zip(breaks[:-1], breaks[1:])])
    coefficient = np.eye(count)[count//2]
    target = BSpline(knots, coefficient, p).derivative()
    data = target(points)
    spaces = {
        'same': (knots, p),
        'derivative': (knots[1:-1], p-1),
        'lower_continuity': (np.r_[np.zeros(p+1), np.repeat(breaks[1:-1], 2), np.ones(p+1)], p),
    }
    result = {'target': target, 'projections': {}, 'errors': {}}
    for name, (field_knots, degree) in spaces.items():
        matrix = BSpline.design_matrix(points, field_knots, degree).toarray()
        scaled = np.sqrt(weights)[:, None]*matrix
        fitted = np.linalg.lstsq(scaled, np.sqrt(weights)*data, rcond=None)[0]
        result['projections'][name] = BSpline(field_knots, fitted, degree)
        result['errors'][name] = np.sqrt(weights@((matrix@fitted-data)**2))
    return result


def main():
    import matplotlib.pyplot as plt
    result = gradient_space_diagnostic()
    points = np.linspace(0., 1., 500)
    plt.plot(points, result['target'](points), color='black', label='Derivative of cubic C2 displacement basis')
    for name, field in result['projections'].items():
        print(f"{name}: L2 projection error={result['errors'][name]:.6e}")
        plt.plot(points, field(points), '--', label=name)
    plt.xlabel('Parameter u')
    plt.ylabel('Hierarchical coefficient for zero physical director')
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
