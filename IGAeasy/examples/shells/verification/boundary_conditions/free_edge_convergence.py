"""Curved RM5/KL strip: one weak clamp and three naturally loaded edges."""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import orient

from IGAeasy.geometry.get_geometry import polynomial_shell
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.trimming.Brep import BRep
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.elements.KLshell3p import KLShell
from IGAeasy.elements.RMshell5p import RMShell
from IGAeasy.constraints.kl_clamped_weak import WeakKLClampedBoundary
from IGAeasy.constraints.shell_weak import WeakRM5Boundary
from IGAeasy.loads import SurfaceLoad, ShellEdgeLoad
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.solver.direct_solver import solve
from IGAeasy.postprocessing.postprocessing import update_cps
from IGAeasy.postprocessing.visualization import plot_solution
from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature, SurfaceDirectorL2Quadrature
from IGAeasy.manufactured_solutions.kl_parabolic_reference import ParabolicKLReference, monomial_coefficient
from IGAeasy.manufactured_solutions.rm_parabolic_reference import ParabolicRMShearReference


net_sizes = (2, 4, 8, 16)
formulations = ('RM5', 'KL')
angle = .6
deformation_scale = 10.
# Slanted free edges give KL nonzero twisting moments and physical corner work.
trimmed_domain = Polygon([(.1, .1), (.88, .1), (.86, .9), (.1, .9)])


def reference_balance(reference, domain, norm, formulation):
    """Independent continuum force/moment resultants and strain energy."""
    total, root = np.zeros(6), np.zeros(6)
    exact_energy = 0.
    R, c = reference.rotation, reference.curvature
    for _, _, weights, positions in norm.entries:
        for weight, position in zip(weights, positions):
            u, v, _ = R.T@position
            body = reference.body(u, v)
            moment = np.cross(position, body[:3])
            if formulation == 'RM5':
                normal = R@np.array([-2*c*u, 0., 1.])/np.sqrt(1+4*c*c*u*u)
                director_force = R@np.array([body[3], body[4], 2*c*u*body[3]])
                moment += np.cross(normal, director_force)
            total += weight*np.r_[body[:3], moment]
            exact_energy += weight*reference.energy_density(u, v)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    corners = np.asarray(domain.exterior.coords)
    for start, end in zip(corners[:-1], corners[1:]):
        direction = end-start
        for s, weight in zip((nodes+1)/2, weights/2):
            u, v = start+s*direction
            position = R@np.array([u, v, c*u*u])
            normal = R@np.array([-2*c*u, 0., 1.])/np.sqrt(1+4*c*c*u*u)
            tangent = R@np.array([direction[0], direction[1], 2*c*u*direction[0]])
            measure = weight*np.linalg.norm(tangent)
            conormal = np.cross(tangent/np.linalg.norm(tangent), normal)
            force, moment = np.split(reference.edge_data(*position, conormal), 2)
            resultant = measure*np.r_[force, np.cross(position, force)+np.cross(normal, moment)]
            total += resultant
            if np.allclose([start[0], end[0]], domain.bounds[0]):
                root += resultant
    return total, root, exact_energy


def solve_level(n, formulation='KL', family='THB', rotation_angle=angle,
                domain=None, local_refinement=False, power=4, method='nitsche'):
    if formulation not in ('KL', 'RM5') or family not in ('THB', 'LR'):
        raise ValueError('Choose KL/RM5 and THB/LR')

    # --- Exact parabolic geometry and independent continuum loads ---
    degree = 3
    rotation = rotation_matrix(rotation_angle, (1., 2., 3.))
    reference = (ParabolicKLReference(power=power, rotation=rotation) if formulation == 'KL'
                 else ParabolicRMShearReference('RM5', power=power, rotation=rotation))
    geometry = polynomial_shell(n, n, degree, (reference.curvature, 0.), centered=False)
    cps = geometry.control_points@rotation.T
    space = (HierarchicalSpace(*geometry.knot_vectors, degree, cps) if family == 'THB'
             else LRSpline(degree, *geometry.knot_vectors, cps))
    if local_refinement:
        selected = min(space.basis, key=lambda b: np.linalg.norm(rotation.T@b.cp-np.array([.7, .3, .098])))
        if family == 'THB':
            space.refine({0: [selected.loc_id]})
        else:
            space.refine_by_structured_mesh([selected])
            space.get_connectivity()
    material_domain = orient(domain if domain is not None else Polygon([(0., 0.), (1., 0.), (1., 1.), (0., 1.)]))
    if domain is not None:
        space.BReps = [BRep(1, polygon=material_domain, outerloop=True)]
        trim(space)

    # --- One root clamp; all other traces remain unconstrained ---
    corners = list(material_domain.exterior.coords)
    edges = [LineString([a, b]) for a, b in zip(corners[:-1], corners[1:])]
    root = next(edge for edge in edges if np.allclose(np.asarray(edge.coords)[:, 0], material_domain.bounds[0]))
    natural_edges = [edge for edge in edges if edge is not root]
    shell = KLShell() if formulation == 'KL' else RMShell()
    dof = shell.dof_per_cp
    material = LinearElasticMaterial(reference.E, reference.nu, 1.)
    assign_gp_and_properties(space, material, {'t': reference.thickness, 'rho': 1.}, full_quad=True)
    boundary_type = WeakKLClampedBoundary if formulation == 'KL' else WeakRM5Boundary
    boundary = boundary_type(root, reference.field, method=method,
                             safety_factor=400. if method == 'penalty' else 4.)
    edge_loads = [ShellEdgeLoad(edge, reference.edge_data, formulation) for edge in natural_edges]
    body_load = SurfaceLoad(dof, reference.body)
    space.boundary_conditions = [boundary]

    # --- Body, raw force and director-moment work, then the single weak clamp ---
    for cell in space.elements:
        if cell.void or not getattr(cell, 'active', True):
            continue
        shell.element_routine(cell)
        cell.K_bulk = cell.K_el
        cell.f_el += body_load.integrate_contribution(cell, dof)
        for load in edge_loads:
            cell.f_el += load.integrate_contribution(cell)
        cell.f_load = cell.f_el.copy()
        matrix, force, _ = boundary.integrate_contribution(cell)
        cell.K_boundary, cell.f_boundary = matrix, force
        cell.K_el = cell.K_bulk+matrix
        cell.f_el += force

    # --- Sparse solve and independent displacement/director norms ---
    K = MatrixAssembler(dof).assemble(space, 'K_el', use_sparse=True)
    f = VectorAssembler(dof).assemble(space, 'f_el')
    coefficients = solve(K, f, scaling=True).reshape(-1, dof)
    norm = SurfaceL2Quadrature(space, order=degree+3)
    director_norm = SurfaceDirectorL2Quadrature(space, formulation, order=degree+3)
    K_boundary = MatrixAssembler(dof).assemble(space, 'K_boundary', use_sparse=True)
    f_boundary = VectorAssembler(dof).assemble(space, 'f_boundary')
    f_load = VectorAssembler(dof).assemble(space, 'f_load')
    solution = coefficients.ravel()
    reaction = f_boundary-K_boundary@solution
    energy = sum(.5*coefficients[cell.bfnums].ravel()@cell.K_bulk@coefficients[cell.bfnums].ravel()
                 for cell in space.elements if not cell.void and getattr(cell, 'active', True))
    external_work, boundary_work = solution@f_load, solution@reaction
    balance, exact_root, exact_energy = reference_balance(reference, material_domain, norm, formulation)
    # Rigid-rotation virtual work is exact for KL. Curved RM5 director coordinates
    # are sampled at Greville points here, so their moment recovery is approximate.
    reaction_moment = np.zeros(3)
    for b in space.basis:
        value = reaction.reshape(-1, dof)[b.id]
        reaction_moment += np.cross(b.cp, value[:3])
        if formulation == 'RM5':
            u = np.mean(b.knots_u[1:-1])
            c = reference.curvature
            normal = rotation@np.array([-2*c*u, 0., 1.])/np.sqrt(1+4*c*c*u*u)
            director_force = rotation@np.array([value[3], value[4], 2*c*u*value[3]])
            reaction_moment += np.cross(normal, director_force)
    row = dict(n=n, h=1/n, l2=norm.error(coefficients[:, :3], lambda *x: reference.field(*x)[:3]),
               director_l2=director_norm.error(coefficients, lambda *x: reference.field(*x)[3:]),
               residual=np.linalg.norm(K@coefficients.ravel()-f)/np.linalg.norm(f),
               min_parametric_fraction=min(sum(p.area for p in cell.material_polygons)
                   /((cell.u_max-cell.u_min)*(cell.v_max-cell.v_min)) if cell.trimmed else 1.
                   for cell in space.elements if not cell.void and getattr(cell, 'active', True)))
    row.update(internal_energy=energy, energy_error=abs(energy-exact_energy),
               energy_balance=abs(2*energy-external_work-boundary_work)/max(2*energy, 1e-300),
               physical_force_balance=np.linalg.norm(balance[:3]), physical_moment_balance=np.linalg.norm(balance[3:]),
               reaction_force_error=np.linalg.norm(reaction.reshape(-1, dof)[:, :3].sum(axis=0)-exact_root[:3]),
               virtual_reaction_moment_error=np.linalg.norm(reaction_moment-exact_root[3:]))
    if power <= degree:
        exact = np.zeros_like(coefficients)
        for b in space.basis:
            value = monomial_coefficient(b.knots_u[1:-1], power)
            exact[b.id, :3] = rotation@([0., 0., reference.amplitude*value] if formulation == 'KL'
                                       else [0., reference.amplitude*value, 0.])
            if formulation == 'RM5':
                exact[b.id, 4] = reference.director_amplitude*value
        row['patch_residual'] = np.linalg.norm(K@exact.ravel()-f)/np.linalg.norm(f)
    update_cps(space, coefficients.ravel(), dof_per_cp=dof, formulation=formulation)
    return space, row


def main(show=True, sizes=net_sizes):
    results = {}
    for formulation in formulations:
        rows = []
        for n in sizes:
            print(f'{formulation}, natural edges: {n} x {n}...', flush=True)
            space, row = solve_level(n, formulation)
            rows.append(row)
            print(row, flush=True)
        results[formulation] = rows

    # --- Norm convergence and one geometry window, with deformation toggle ---
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for name, rows in results.items():
        for axis, key in zip(axes, ('l2', 'director_l2')):
            axis.loglog([r['h'] for r in rows], [r[key] for r in rows], 'o-', label=name)
            axis.set(xlabel='Background element size h', ylabel=key)
            axis.grid(True, which='both')
            axis.legend()
    if show:
        plot_solution(space, deformation_scale=deformation_scale)
        plt.show()
    else:
        plt.close(figure)
    return results


if __name__ == '__main__':
    main()
