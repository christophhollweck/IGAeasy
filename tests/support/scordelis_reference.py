"""Scordelis-Lo roof: strong and weak diaphragms for RM and KL shells."""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import block_diag, kron, eye
from shapely.geometry import LineString, Point
from IGAeasy.trimming.Brep import BRep
from IGAeasy.geometry import get_geometry
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.elements.RMshell6p import RMShell
from IGAeasy.elements import RMshell5p, RMshell5p_v2, RMshell5p_hier, KLshell3p
from IGAeasy.constraints.rm5_hier_weak import WeakRM5HierBoundary
from IGAeasy.constraints.kl_mixed_weak import WeakKLMixedBoundary
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.numerical_integration.costum_rule import gauss_lobatto
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.constraints.shell_weak import WeakRM6Boundary, WeakRM5Boundary
from IGAeasy.constraints.boundary_selection import select_patch_boundary
from IGAeasy.constraints.dirichlet import block_partitioning_dbc
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.loads import SurfaceLoad
from IGAeasy.solver.direct_solver import solve
from IGAeasy.postprocessing import postprocessing
from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature

n_elx = 2
n_ely = 2
p = 2
n_refinements = 3  # 4 x 4, 8 x 8, 16 x 16, 32 x 32
formulations = ['RM6', 'RM5', 'RM5_hier', 'KL']
with_hole = False  # The rounded Scordelis benchmark applies only without a hole.
angles = [0., .6]
methods = ['strong', 'penalty', 'nitsche']
penalty_multiplier = 100.
safety_factor = 4.
t = .25
E = 4.32e8
nue = 0.
load = -90.
reference_uz = -.3024  # Rounded RM benchmark; fitted geometry is held fixed.
plot_geometry = True



def displacement_norm(RS, coefficients, dof=6):
    return RS.study_error_quadrature.error(coefficients.reshape(-1, dof)[:, :3])


def boundary_norm(RS, coefficients, boundaries, dof=6):
    # Penalty with unit dimensional coefficient is the squared selected trace norm.
    total = 0.
    for cell in RS.elements:
        if cell.void or not getattr(cell, 'active', True):
            continue
        ids = [dof*b.id+j for b in sorted(cell.supported_b_splines, key=lambda b: b.id) for j in range(dof)]
        local = coefficients[ids]
        for boundary in boundaries:
            matrix, _, _ = boundary.integrate_contribution(cell)
            total += local@matrix@local
    return np.sqrt(max(total, 0.))


def run_formulation(formulation='RM6', show=True, hole=False, refinements=None, make_plots=True,
                    method_names=None, rotation_angles=None, family='THB', show_geometry=None,
                    rotation_axes=None, n_elx=n_elx, n_ely=n_ely, degree=p):
    if family not in ('THB','LR'):
        raise ValueError('family must be THB or LR')
    selected_methods = methods if method_names is None else list(method_names)
    selected_angles = angles if rotation_angles is None else list(rotation_angles)
    if (not selected_methods or any(m not in ('strong', 'penalty', 'nitsche', 'penalty_free') for m in selected_methods)
            or ('strong' in selected_methods and selected_methods[0] != 'strong')):
        raise ValueError('Select supported methods; strong must come first when included')
    if not selected_angles or not np.isfinite(selected_angles).all():
        raise ValueError('Select at least one finite rotation angle')
    selected_axes = [(1., 2., 3.)]*len(selected_angles) if rotation_axes is None else list(rotation_axes)
    if len(selected_axes) != len(selected_angles):
        raise ValueError('Provide one rotation axis per angle')
    rotations = [rotation_matrix(angle, axis) for angle, axis in zip(selected_angles, selected_axes)]
    results = []
    geometry = get_geometry.scordelis_lo_roof(n_elx, n_ely, degree, full=True)
    if (plot_geometry if show_geometry is None else show_geometry) and show:
        uv = np.array([[u, v] for v in np.linspace(0., 1., 21) for u in np.linspace(0., 1., 21)])
        points = geometry.evaluate(uv).reshape(21, 21, 3)
        points = points@rotations[0].T
        figure = plt.figure()
        axis = figure.add_subplot(projection='3d')
        axis.plot_wireframe(points[:, :, 0], points[:, :, 1], points[:, :, 2])
        axis.set_box_aspect(np.ptp(points.reshape(-1, 3), axis=0))
        axis.set_title('Untrimmed Scordelis-Lo roof')
    rule = gauss_lobatto(3)
    refinements = n_refinements if refinements is None else refinements
    if formulation not in ('RM6', 'RM5', 'RM5_v2', 'RM5_hier', 'KL'):
        raise ValueError('Unsupported shell formulation')
    if formulation in ('KL','RM5_hier') and 'penalty_free' in selected_methods:
        raise ValueError('KL/RM5_hier roof supports penalty and symmetric Nitsche only')
    dof = 6 if formulation == 'RM6' else (3 if formulation == 'KL' else 5)
    if formulation == 'KL':
        shell = KLshell3p.KLShell()
    else:
        shell = {'RM6':RMShell, 'RM5':RMshell5p.RMShell, 'RM5_v2':RMshell5p_v2.RMShell,
                 'RM5_hier':RMshell5p_hier.RMShell}[formulation]()
    benchmark_reference = -.3006 if formulation == 'KL' else reference_uz

    def boundary(line, directions, method='nitsche', penalty=None):
        callback = lambda x, y, z: np.zeros(6)
        if formulation == 'RM6':
            return WeakRM6Boundary(line, callback, rule, method=method,
                directions=directions, safety_factor=safety_factor, penalty=penalty)
        if formulation == 'RM5_hier':
            return WeakRM5HierBoundary(line, callback, method=method,
                directions=directions, safety_factor=safety_factor, penalty=penalty)
        if formulation == 'KL':
            return WeakKLMixedBoundary(line, callback, method=method,
                displacement_directions=directions[:,:3], safety_factor=safety_factor, penalty=penalty)
        return WeakRM5Boundary(line, callback, formulation=formulation, method=method,
            directions=directions, safety_factor=safety_factor, penalty=penalty)
    reference_solutions = {}
    for orientation, (angle, R) in enumerate(zip(selected_angles, rotations)):
        control_points = np.asarray(geometry.control_points)@R.T
        RS = (HierarchicalSpace(*geometry.knot_vectors, degree, control_points, refinements+1, 'h')
              if family == 'THB' else LRSpline(degree, *geometry.knot_vectors, control_points))
        if hole:
            RS.BReps = [BRep(1, polygon=Point(.5, .5).buffer(.08, quad_segs=16), outerloop=False)]
        vertical = R@np.array([0., 0., 1.])
        loads = SurfaceLoad(dof, [lambda u, v, value=load*component: value for component in vertical]
                            + [lambda u, v: 0.]*(dof-3))
        directions = np.column_stack((R[:, [0, 2]].T, np.zeros((2, 3))))
        lines = [LineString([(0., 0.), (1., 0.)]), LineString([(1., 1.), (0., 1.)])]
        for level in range(refinements+1):
            start = time.perf_counter()
            if level:
                if family == 'THB':
                    functions = RS.get_funcs_in_rectangle(level-1, [[[0., 1.], [0., 1.]]])
                    RS.refine({level-1: functions})
                else:
                    RS.refine_by_structured_mesh(list(RS.basis))
                    RS.get_connectivity()
            trim(RS)
            assign_gp_and_properties(RS, material_model=LinearElasticMaterial(E, nue, 1.),
                                     properties={'t': t, 'rho': 1.}, full_quad=True)
            cells = [cell for cell in RS.elements if getattr(cell, 'active', True) and not cell.void]
            RS.study_error_quadrature = SurfaceL2Quadrature(RS, order=degree+3)
            bulk = {}
            for cell in cells:
                if formulation == 'RM6':
                    shell.element_routine(cell, rule)
                else:
                    shell.element_routine(cell)
                cell.f_el += loads.integrate_contribution(cell, dof)
                bulk[id(cell)] = cell.K_el.copy(), cell.f_el.copy()
            end_ids = select_patch_boundary(RS, 'vmin') | select_patch_boundary(RS, 'vmax')
            corner = min(select_patch_boundary(RS, 'umin') & select_patch_boundary(RS, 'vmin'))
            block = R if dof == 3 else block_diag((R, R if dof == 6 else np.eye(2)))
            transform = kron(eye(len(RS.basis)), block, format='csr')
            solutions = {}
            print(f'{formulation}, hole={hole}, angle={angle:.3f}, mesh={2*n_elx*2**level}: bulk setup {time.perf_counter()-start:.2f}s', flush=True)
            for method in selected_methods:
                boundary_terms = {}
                if method != 'strong':
                    boundaries = [boundary(line, directions, method=method) for line in lines]
                    for cell in cells:
                        cell.K_el = bulk[id(cell)][0]
                        boundary_terms[id(cell)] = [boundary.integrate_contribution(cell) for boundary in boundaries]
                for cell in cells:
                    Kbulk, fbulk = bulk[id(cell)]
                    cell.K_el, cell.f_el = Kbulk.copy(), fbulk.copy()
                    for matrix, force, _ in boundary_terms.get(id(cell), []):
                        factor = penalty_multiplier if method == 'penalty' else 1.
                        cell.K_el += factor*matrix
                        cell.f_el += factor*force
                K = MatrixAssembler(dof_per_cp=dof).assemble(RS, 'K_el', use_sparse=True)
                f = VectorAssembler(dof_per_cp=dof).assemble(RS, 'f_el')
                local_K, local_f = transform.T@K@transform, transform.T@f
                fixed = {dof*corner+1}  # Only remove the remaining longitudinal rigid translation.
                if method == 'strong':
                    fixed.update(dof*i+j for i in end_ids for j in (0, 2))
                reduced, rhs, free = block_partitioning_dbc(K=local_K, f=local_f, prescribed_DOFs=fixed)
                local_solution = np.zeros(len(f))
                local_solution[free] = solve(reduced, rhs, scaling='auto')
                residual = np.linalg.norm(reduced@local_solution[free]-rhs)/np.linalg.norm(rhs)
                displacement = transform@local_solution
                postprocessing.update_cps(RS, displacement, dof_per_cp=dof, dimensions=3)
                uz = vertical@RS.eval_displacement(1., .5)
                solutions[method] = displacement
                relative_error = (displacement_norm(RS, displacement-solutions['strong'], dof)
                    /displacement_norm(RS, solutions['strong'], dof)) if 'strong' in solutions else np.nan
                trace_boundaries = [boundary(line, directions, method='penalty', penalty=1.) for line in lines]
                trace = boundary_norm(RS, displacement, trace_boundaries, dof)
                rotated_difference = 0.
                coefficient_difference = 0.
                if orientation == 0:
                    reference_solutions[level, method] = local_solution
                else:
                    difference = local_solution-reference_solutions[level, method]
                    coefficient_difference = np.linalg.norm(difference)/np.linalg.norm(local_solution)
                    rotated_difference = displacement_norm(RS, transform@difference, dof)/displacement_norm(RS, displacement, dof)
                    if max(rotated_difference, coefficient_difference) > 1e-7:
                        raise RuntimeError('Spatial rotation changed the physical solution.')
                benchmark = f', benchmark difference={abs(uz-benchmark_reference):.3e}' if not hole else ''
                comparison = f', relative L2 vs strong={relative_error:.3e}' if 'strong' in solutions else ''
                print(f'  {method}: uz={uz:.10e}{benchmark}{comparison}, '
                      f'boundary L2={trace:.3e}, residual={residual:.3e}, '
                      f'rotation difference={rotated_difference:.3e}', flush=True)
                results.append(dict(angle=angle, orientation=orientation, residual=residual,
                                    method=method, n=2*n_elx*2**level, uz=uz,
                                    error=relative_error, trace=trace, rotation_difference=rotated_difference,
                                    coefficient_rotation_difference=coefficient_difference,
                                    benchmark_error=abs(uz-benchmark_reference) if not hole else np.nan))
    if make_plots:
        plot_convergence(results, formulation, benchmark_reference, hole=hole, show=show)
    return results


def plot_convergence(results, formulation, benchmark_reference, hole=False, show=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    orientations = dict.fromkeys(row['orientation'] for row in results)
    methods = dict.fromkeys(row['method'] for row in results)
    for orientation in orientations:
        for method in methods:
            rows = [row for row in results if row['orientation'] == orientation and row['method'] == method]
            x = [row['n'] for row in rows]
            label = f"{method}, orientation={orientation+1}, angle={rows[0]['angle']:.1f}"
            axes[0].plot(x, [row['uz'] for row in rows], 'o-', label=label)
            if method != 'strong':
                axes[1].loglog(x, [row['error'] for row in rows], 'o-', label=label)
                axes[2].loglog(x, [row['trace'] for row in rows], 'o-', label=label)
    if not hole:
        axes[0].axhline(benchmark_reference, color='k', linestyle='--', label='rounded benchmark')
    for axis, label in zip(axes, ['Free-edge midpoint displacement', 'Relative L2 difference from strong',
                                   'L2 diaphragm displacement error']):
        axis.set_xlabel('Elements per direction')
        axis.set_ylabel(label)
        axis.grid(True, which='both', alpha=.3)
        axis.legend(fontsize=8)
    plt.tight_layout()
    fig.suptitle(f'{formulation}: diaphragm conditions'+(' (roof with hole)' if hole else ''))
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(show=True):
    summaries = {}
    for formulation in formulations:
        summaries[formulation] = run_formulation(formulation, show=show, hole=with_hole)
    return summaries


if __name__ == '__main__':
    main(show='--no-show' not in sys.argv)
