"""Hierarchical RM5: weak displacement and natural physical director moments."""
from IGAeasy.examples.shells.verification.boundary_conditions import gradient_clamped_convergence as study

n = 2
p = 3
n_refinements = 3
family = 'THB'
trimmed = True
angle = .6
methods = ['penalty', 'nitsche']


def main(show=True):
    result = {}
    for method in methods:
        print(f'RM5_hier mixed: {method}', flush=True)
        result[method] = study.main(show=show, n=n, p=p, n_refinements=n_refinements,
            family=family, trimmed=trimmed, angle=angle, formulation='RM5_hier',
            boundary_kind='mixed_hier', boundary_method=method,
            boundary_safety_factor=400. if method == 'penalty' else None)
    return result


if __name__ == '__main__':
    main()
