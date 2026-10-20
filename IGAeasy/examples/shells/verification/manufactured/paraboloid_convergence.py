"""Doubly curved shells: manufactured solutions, convergence and field plots."""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box, LineString
from shapely.geometry.polygon import orient
from IGAeasy.trimming.Brep import BRep

from IGAeasy.geometry import get_geometry
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.elements import RMshell5p, RMshell5p_v2, RMshell5p_hier, KLshell3p
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.constraints.shell_weak import WeakRM5Boundary
from IGAeasy.constraints.kl_clamped_weak import WeakKLClampedBoundary, WeakRM5HierClampedBoundary
from IGAeasy.constraints.kl_mixed_weak import WeakKLMixedBoundary
from IGAeasy.constraints.rm5_hier_weak import WeakRM5HierBoundary
from IGAeasy.loads import SurfaceLoad
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.postprocessing.postprocessing import update_cps
from IGAeasy.solver.direct_solver import solve
from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature, SurfaceDirectorL2Quadrature
from IGAeasy.manufactured_solutions.rm_paraboloid_reference import ParaboloidRMReference
from IGAeasy.manufactured_solutions.gradient_paraboloid_reference import GradientParaboloidReference
from IGAeasy.postprocessing.visualization import plot_solution


# --- Discretization & boundary conditions ---
net_sizes = (2, 4, 8, 16)
formulations = ('RM5', 'RM5_hier', 'KL')
method = 'nitsche'  # Change to 'penalty' to compare the boundary approximation.
p = 3
# --- Geometry & material properties ---
curvature = (1.0, -1.0)  # Equal signs give a cap; opposite signs give a saddle.
mixed_terms = (0., 0.)  # Nonzero values add x**2*y and x*y**2 geometry terms.
angle = .6  # Rotation about (1,2,3), away from the coordinate axes.
E = 100.   # Young's modulus
nue = .3   # Poisson's ratio
t = .1     # Shell thickness

# --- Reference solution & visualization ---
amplitude = .001  # Small displacements keep the linear shell model appropriate.
hierarchical_eta = (.0003, -.0002)  # Additional director coefficients for RM5_hier.
rm_director_amplitude = .0003  # Covariant director field for the enriched RM5 case.
deformation_scale = 50.
samples_per_element = 5


def solve_level(n, formulation='RM5', boundary_method=method, rotation_angle=angle,
                surface_curvature=curvature, director_amplitude=0., enriched=False,
                mixed=(0., 0.), domain=None, family='THB', stabilization='scalar',
                local_refinements=0, refinement_region=(.25, .25, .75, .75),
                boundary_kind='clamped', penalty_factor=100.):

    # --- Geometry & reference solution ---
    rotation = rotation_matrix(rotation_angle)
    if boundary_kind not in ('clamped', 'mixed', 'normal'):
        raise ValueError('boundary_kind must be clamped, mixed or normal')
    if formulation in ('KL', 'RM5_hier'):
        reference = GradientParaboloidReference(
            formulation, surface_curvature, E, nue, t, amplitude, rotation,
            eta=hierarchical_eta, mixed=mixed
        )
    else:
        reference = ParaboloidRMReference(
            surface_curvature, E, nue, t, amplitude, rotation, centered=True,
            director_amplitude=director_amplitude, enriched=enriched, mixed=mixed
        )

    geometry = get_geometry.polynomial_shell(n, n, p, surface_curvature, mixed=mixed)
    cps = geometry.control_points@rotation.T
    if family == 'THB':
        RS = HierarchicalSpace(*geometry.knot_vectors, p, cps)
    elif family == 'LR':
        RS = LRSpline(p, *geometry.knot_vectors, cps)
    else:
        raise ValueError('Use THB or LR')

    # --- Optional local refinement, before trimming and integration ---
    # Mark by parametric Greville points; THB and LR may create different meshes.
    if not isinstance(local_refinements, (int, np.integer)) or local_refinements < 0:
        raise ValueError('local_refinements must be a nonnegative integer')
    u0, v0, u1, v1 = refinement_region
    for _ in range(local_refinements):
        marked = [basis for basis in RS.basis
                  if u0 <= np.mean(basis.knots_u[1:-1]) <= u1
                  and v0 <= np.mean(basis.knots_v[1:-1]) <= v1]
        if not marked:
            raise ValueError('The refinement region contains no basis Greville points')
        if family == 'THB':
            levels = {}
            for basis in marked:
                levels.setdefault(basis.level, []).append(basis.loc_id)
            RS.refine(levels)
        else:
            RS.refine_by_structured_mesh(marked)
            RS.get_connectivity()

    # --- Optional trimming ---
    if domain is not None:
        domain = orient(domain, sign=1.)  # Material stays left of outer and hole boundaries.
        RS.BReps = [BRep(1, polygon=domain, outerloop=True)]
        trim(RS)
    else:
        domain = box(0., 0., 1., 1.)

    # --- Material & integration ---
    material = LinearElasticMaterial(E, nue, 1., kappa=5/6)

    assign_gp_and_properties(
        RS,
        material_model=material,
        properties={'t': t, 'rho': 1.},
        full_quad=True
    )

    # --- Shell formulation ---
    if formulation == 'KL':
        shell = KLshell3p.KLShell()
    else:
        module = {'RM5': RMshell5p, 'RM5_v2': RMshell5p_v2, 'RM5_hier': RMshell5p_hier}[formulation]
        shell = module.RMShell()
    dof = shell.dof_per_cp

    # --- Boundary conditions & loads ---
    # Prescribe the exact physical displacement and director along the full boundary.
    lines = [LineString(ring.coords) for ring in (domain.exterior, *domain.interiors)]
    if boundary_kind in ('mixed', 'normal'):
        # All displacement components are essential; director moments are natural.
        boundary_options = {'method': boundary_method,
                            'safety_factor': 4.*penalty_factor if boundary_method == 'penalty' else 4.}
        if formulation == 'KL':
            options = ({'displacement_directions': 'normal',
                        'traction': lambda x, y, z, mu: reference.traction(x, y, z, mu, normal_only=True)}
                       if boundary_kind == 'normal' else {})
            boundaries = [WeakKLMixedBoundary(line, reference.field, **boundary_options,
                normal_moment=lambda x, y, z, mu: mu@reference.moment(x, y, z, mu), **options) for line in lines]
        else:
            boundary_class = WeakRM5HierBoundary if formulation == 'RM5_hier' else WeakRM5Boundary
            options = {} if formulation == 'RM5_hier' else {'formulation': formulation}
            if boundary_kind == 'normal':
                options['directions'] = lambda x, y, z: np.r_[reference.normal(x, y, z), np.zeros(3)][None, :]
            components = (('ux', 'uy', 'uz', 'dx', 'dy', 'dz') if boundary_kind == 'normal'
                          else ('ux', 'uy', 'uz'))
            boundaries = [boundary_class(line, reference.field, **boundary_options,
                components=components,
                natural=lambda x, y, z, mu: np.r_[reference.traction(x, y, z, mu)
                    if boundary_kind == 'normal' else np.zeros(3), reference.moment(x, y, z, mu)],
                **options) for line in lines]
        load = (SurfaceLoad(dof, reference.body) if formulation in ('KL', 'RM5_hier') else
                SurfaceLoad(dof, lambda u, v: reference.body(u, v, formulation)))
    elif formulation in ('KL', 'RM5_hier'):
        boundary_class = WeakKLClampedBoundary if formulation == 'KL' else WeakRM5HierClampedBoundary
        boundaries = [boundary_class(line, reference.field, method=boundary_method,
                                     stabilization=stabilization) for line in lines]
        load = SurfaceLoad(dof, reference.body)
    else:
        boundaries = [WeakRM5Boundary(line, reference.field, formulation=formulation, method=boundary_method)
                      for line in lines]
        load = SurfaceLoad(dof, lambda u, v: reference.body(u, v, formulation))

    # Keep the actual condition objects available for support visualization.
    if boundary_kind == 'normal':
        for boundary in boundaries:
            boundary.normal_only = True
    RS.boundary_conditions = boundaries

    # --- Element-wise computations of K_el and f_el ---
    for cell in RS.elements:
        if cell.void or not getattr(cell, 'active', True):
            continue
        shell.element_routine(cell)
        force = load.integrate_contribution(cell, dof)

        # Compute the boundary bound before modifying the bulk stiffness.
        contributions = [boundary.integrate_contribution(cell) for boundary in boundaries]
        for Kbc, fbc, _ in contributions:
            cell.K_el += Kbc
            cell.f_el += fbc
        cell.f_el += force

    # --- Global system assembly ---
    K = MatrixAssembler(dof_per_cp=dof).assemble(RS, 'K_el', use_sparse=True)
    f = VectorAssembler(dof_per_cp=dof).assemble(RS, 'f_el')

    # --- Solve the linear system ---
    coefficients = solve(K, f, scaling=True).reshape(-1, dof)

    # --- Evaluate physical error norms ---
    # Use higher-order quadrature to assess the error independently of bulk integration.
    norm = SurfaceL2Quadrature(RS, order=p+3)
    director_norm = SurfaceDirectorL2Quadrature(RS, formulation, order=p+3)

    error = norm.error(coefficients[:, :3], reference.displacement)
    director_error = director_norm.error(coefficients, lambda x, y, z: reference.field(x, y, z)[3:])
    residual = np.linalg.norm(K@coefficients.ravel()-f)/np.linalg.norm(f)

    # --- Update displacement fields for visualization ---
    update_cps(RS, coefficients.ravel(), dof_per_cp=dof, formulation=formulation)

    fractions = [sum(polygon.area for polygon in cell.material_polygons)
                 /((cell.u_max-cell.u_min)*(cell.v_max-cell.v_min)) if cell.trimmed else 1.
                 for cell in RS.elements if not cell.void and getattr(cell, 'active', True)]
    return RS, dict(h=1/n, l2=error, director_l2=director_error, residual=residual,
                   min_parametric_fraction=min(fractions), n_basis=len(RS.basis),
                   n_elements=sum(not cell.void for cell in RS.elements))


def main(show=True, sizes=net_sizes, surface_curvature=curvature, mixed=mixed_terms,
         formulation_names=formulations, domain=None, family='THB', stabilization='scalar'):

    # --- Convergence study ---
    results = {}
    for formulation in formulation_names:
        rows = []

        for n in sizes:
            print(f'{formulation}: preparing {n} x {n} elements...', flush=True)
            RS, row = solve_level(
                n, formulation, surface_curvature=surface_curvature, mixed=mixed,
                director_amplitude=rm_director_amplitude, enriched=True,
                domain=domain, family=family, stabilization=stabilization
            )
            rows.append(row)

            rate = (np.log(rows[-2]['l2']/row['l2'])/np.log(rows[-2]['h']/row['h'])
                    if len(rows) > 1 else float('nan'))
            print(f'{formulation}, {n} x {n}: L2={row["l2"]:.6e}, rate={rate:.2f}, '
                  f'director L2={row["director_l2"]:.3e}, residual={row["residual"]:.3e}', flush=True)
            if domain is not None:
                print(f'  Minimum parametric material fraction: {row["min_parametric_fraction"]:.3f}', flush=True)

        results[formulation] = rows

    # --- Convergence plot ---
    figure, axis = plt.subplots()
    for name, rows in results.items():
        axis.loglog([r['h'] for r in rows], [r['l2'] for r in rows], 'o-', label=name)

    h = np.array([r['h'] for r in rows])
    axis.loglog(h, rows[0]['l2']*(h/h[0])**(p+1), 'k--', label=f'h^{p+1}')
    axis.set(xlabel='h', ylabel='Displacement L2 error', title='Manufactured shell convergence')
    axis.grid(True, which='both')
    axis.legend()

    if show:
        plt.show()

        # --- One geometry/result view; colors retain physical displacement values ---
        plot_solution(RS, deformation_scale=deformation_scale,
                      samples_per_element=samples_per_element)

    return results


if __name__ == '__main__':
    main()
