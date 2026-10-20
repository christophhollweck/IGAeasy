"""Affine membrane patch on rotated shells with shrinking material strips."""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box, LineString
from IGAeasy.trimming.Brep import BRep
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.elements import RMshell6p, RMshell5p, RMshell5p_v2
from IGAeasy.constraints.shell_weak import WeakRM6Boundary, WeakRM5Boundary
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.solver.direct_solver import solve
from IGAeasy.solver.linear_system import prepare, equilibrate
from IGAeasy.preconditioning import ILUPreconditioner
from scipy.sparse.linalg import gmres
from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature, SurfaceDirectorL2Quadrature

n = 2
p = 2
family = 'THB'
angle = .6
curvature = 0.  # Nonzero curvature selects the RM6 rigid-motion reference.
fractions = [1., .1, .01, .001, .0001, .00001]
formulations = ['RM6', 'RM5', 'RM5_v2']
methods = ['nitsche', 'penalty']
penalty_multiplier = 100.
E, nue, t = 100., .3, .1


def run_case(fraction, formulation='RM6', method='nitsche', family='THB',
             degree=2, n_elements=2, angle=.6, local_refinement=False, iterative=True,
             curvature=0.):
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('fraction must lie in (0, 1]')
    if formulation not in ('RM6', 'RM5', 'RM5_v2') or family not in ('THB', 'LR'):
        raise ValueError('Unknown formulation or spline family')
    if method not in ('penalty', 'nitsche'):
        raise ValueError('method must be penalty or nitsche')
    if not np.isfinite(curvature) or not np.isfinite(angle):
        raise ValueError('curvature and angle must be finite')
    if curvature and formulation != 'RM6':
        raise ValueError('The curved rigid-motion reference in this study uses RM6')
    geo = get_geometry.surface(n_elements, n_elements, 1, 1, 1, 1,
                              degree-1, degree, distortion=0, n_waves=0)
    RS = (HierarchicalSpace(*geo.knot_vectors, degree, geo.control_points, 2)
          if family == 'THB' else LRSpline(degree, *geo.knot_vectors, geo.control_points))
    R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                  [-np.sin(angle), 0., np.cos(angle)]])
    for b in RS.basis:
        cp = np.asarray(b.cp).copy()
        cp[2] = curvature*cp[0]*cp[1]
        b.cp = R@cp
    if local_refinement:
        chosen = min(RS.basis, key=lambda b: np.linalg.norm(np.asarray(b.cp)-R@np.array([.5, .5, 0.])))
        if family == 'THB':
            RS.refine({0: [chosen.loc_id]})
        else:
            RS.refine_by_structured_mesh([chosen])
            RS.get_connectivity()
    domain = box(1-1/n_elements-fraction/n_elements, 0., 1., 1.)
    RS.BReps = [BRep(1, polygon=domain, outerloop=True)]
    trim(RS)
    assign_gp_and_properties(RS, material_model=LinearElasticMaterial(E, nue, 1.),
                             properties={'t': t, 'rho': 1.}, full_quad=True)
    module = {'RM6': RMshell6p, 'RM5': RMshell5p, 'RM5_v2': RMshell5p_v2}[formulation]
    shell = module.RMShell()
    dof = shell.dof_per_cp
    rule = np.polynomial.legendre.leggauss(3)
    omega = R@np.array([.003, -.002, .004] if curvature else [0., 0., -.0025])

    def displacement(x, y, z):
        if curvature:
            return R@np.array([.01, -.02, .03])+np.cross(omega, [x, y, z])
        local = R.T@np.array([x, y, z])
        return R@np.array([.01*local[0]+.002*local[1], -.003*local[0]+.02*local[1], 0.])

    def director_reference(x, y, z):
        local = R.T@np.array([x, y, z])
        normal = np.array([-curvature*local[1], -curvature*local[0], 1.])
        normal = R@(normal/np.linalg.norm(normal))
        return np.cross(omega, normal) if dof == 6 else np.zeros(3)

    def field(x, y, z):
        return np.concatenate((displacement(x, y, z), omega if dof == 6 else np.zeros(3)))

    line = LineString(domain.exterior.coords)
    boundary = (WeakRM6Boundary(line, field, rule, method=method) if dof == 6 else
                WeakRM5Boundary(line, field, formulation=formulation, method=method))
    gamma_values, cut_fractions = [], []
    multiplier = penalty_multiplier if method == 'penalty' else 1.
    for cell in RS.elements:
        if cell.void or not getattr(cell, 'active', True):
            continue
        if dof == 6:
            shell.element_routine(cell, rule)
        else:
            shell.element_routine(cell)
        addition, force, gamma = boundary.integrate_contribution(cell)
        cell.K_el += multiplier*addition
        cell.f_el += multiplier*force
        gamma_values.append(multiplier*gamma)
        if cell.trimmed:
            area = sum(polygon.area for polygon in cell.material_polygons)
            cut_fractions.append(area/((cell.u_max-cell.u_min)*(cell.v_max-cell.v_min)))
    K = MatrixAssembler(dof_per_cp=dof).assemble(RS, 'K_el', use_sparse=True)
    f = VectorAssembler(dof_per_cp=dof).assemble(RS, 'f_el')
    coefficients = solve(K, f, scaling=True).reshape(len(RS.basis), dof)
    exact = np.zeros_like(coefficients)
    for b in RS.basis:
        exact[b.id, :3] = displacement(*b.cp)
        if dof == 6:
            exact[b.id, 3:] = omega
    reduced, rhs, active, _, _ = prepare(K, f, tol=0.)
    scaled, scaled_rhs, back_scale = equilibrate(reduced, rhs, scaling=True)
    scaled_solution = coefficients.ravel()[active]/back_scale
    norm = SurfaceL2Quadrature(RS, order=degree+3)
    director_norm = SurfaceDirectorL2Quadrature(RS, formulation, order=degree+3)
    row = dict(formulation=formulation, method=method, family=family,
               degree=degree, angle=angle, curvature=curvature, local_refinement=local_refinement,
               requested_fraction=fraction, min_fraction=min(cut_fractions, default=1.),
               gamma_max=max(gamma_values), active_dofs=len(active),
               l2=norm.error(coefficients[:, :3], displacement),
               director_l2=director_norm.error(coefficients, director_reference),
               patch_residual=np.linalg.norm(K@exact.ravel()-f)/np.linalg.norm(f),
               residual=np.linalg.norm(K@coefficients.ravel()-f)/np.linalg.norm(f),
               scaled_residual=np.linalg.norm(scaled@scaled_solution-scaled_rhs)/np.linalg.norm(scaled_rhs),
               scaled_condition=np.linalg.cond(scaled.toarray()))
    if iterative:
        ilu = ILUPreconditioner(scaled)
        history = []
        y, info = gmres(scaled, scaled_rhs, M=ilu.as_linear_operator(), rtol=1e-10,
                       atol=0., restart=80, maxiter=20,
                       callback=history.append, callback_type='pr_norm')
        computed = np.zeros(coefficients.size)
        computed[active] = back_scale*y
        computed = computed.reshape(coefficients.shape)
        row.update(iterations=len(history), iterative_info=info,
                   iterative_l2=norm.error(computed[:, :3], displacement),
                   iterative_director_l2=director_norm.error(computed, director_reference),
                   solution_difference=norm.error(computed[:, :3], coefficients[:, :3]),
                   director_difference=director_norm.error(computed, coefficients),
                   iterative_residual=np.linalg.norm(K@computed.ravel()-f)/np.linalg.norm(f))
    return RS, coefficients, row


if __name__ == '__main__':
    rows = []
    for formulation in formulations:
        for method in methods:
            for fraction in fractions:
                print(f'Preparing {formulation}, {method}, fraction={fraction:.1e}...', flush=True)
                _, _, row = run_case(fraction, formulation, method, family, p, n, angle,
                                     curvature=curvature)
                rows.append(row)
                print(f"  L2={row['l2']:.3e}, director L2={row['director_l2']:.3e}, "
                      f"gamma={row['gamma_max']:.3e}, scaled cond={row['scaled_condition']:.3e}, "
                      f"scaled residual={row['scaled_residual']:.3e}", flush=True)
                print(f"  ILU/GMRES: iterations={row['iterations']}, info={row['iterative_info']}, "
                      f"L2 difference={row['solution_difference']:.3e}, "
                      f"director difference={row['director_difference']:.3e}", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for formulation in formulations:
        for method in methods:
            selected = [row for row in rows if row['formulation'] == formulation and row['method'] == method]
            x = [row['requested_fraction'] for row in selected]
            for ax, key, label in zip(axes.flat, ('l2', 'director_l2', 'gamma_max', 'scaled_condition'),
                                      ('Displacement L2 error', 'Director L2 error', 'Maximum penalty', 'Scaled condition number')):
                ax.loglog(x, [max(row[key], np.finfo(float).tiny) for row in selected],
                          'o-', label=f'{formulation}, {method}')
                ax.set_xlabel('Requested material fraction in cut column')
                ax.set_ylabel(label)
                ax.grid(True, which='both')
                ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
