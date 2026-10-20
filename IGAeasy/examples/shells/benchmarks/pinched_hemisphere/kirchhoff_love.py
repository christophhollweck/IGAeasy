"""Pinched open hemisphere: planar symmetry cuts, weak supports and rotation."""
import numpy as np
from scipy.sparse import kron, eye
from shapely.geometry import LineString

from IGAeasy.geometry import get_geometry
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.elements import KLshell3p
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.constraints.kl_symmetry_weak import WeakKLSymmetryBoundary
from IGAeasy.constraints.boundary_selection import select_patch_boundary
from IGAeasy.constraints.dirichlet import block_partitioning_dbc
from IGAeasy.loads import PointLoad
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.postprocessing.postprocessing import update_cps
from IGAeasy.solver.direct_solver import solve
from IGAeasy.postprocessing.visualization import plot_solution


# --- Geometry & material ---
radius = 10.
opening_degrees = 18.
E = 6.825e7
nue = .3
t = .04
base_elements = 8  # Fixed fitted geometry; subsequent meshes only insert knots.

# --- Loads and symmetry ---
# Each equatorial load is shared by two quarters: unit quarter load, full load 2.
quarter_load = 1.
angle = .6
rotation_axis = (1., 2., 3.)  # Geometry, loads and supports rotate together.
penalty_factor = 100.

# Published extrapolated KL value for the OPEN hemisphere; not an exact solution.
reference_displacement = .093597
deformation_scale = 20.


def run(n_elements=8, method='nitsche', show=True, rotation_angle=angle,
        geometry_elements=base_elements):
    n = n_elements
    # --- Fixed, regular geometry and nested mesh sequence ---
    if method not in ('strong', 'penalty', 'nitsche'):
        raise ValueError('Use strong, penalty or nitsche')
    if method == 'strong':
        raise ValueError('This formulation uses weak reflection conditions')
    ratio = n/geometry_elements
    if ratio < 1 or not np.isclose(np.log2(ratio), round(np.log2(ratio))):
        raise ValueError('n must be a dyadic refinement of geometry_elements')
    geometry = get_geometry.open_hemisphere(geometry_elements, radius, opening_degrees)
    for _ in range(round(np.log2(ratio))):
        for direction in (0, 1):
            knots = np.unique(geometry.knot_vectors[direction])
            geometry.insert_knots(direction, (knots[:-1]+knots[1:])/2)
    rotation = rotation_matrix(rotation_angle, rotation_axis)
    space = HierarchicalSpace(*geometry.knot_vectors, 3, geometry.control_points@rotation.T)

    shell = KLshell3p.KLShell()
    dof = shell.dof_per_cp
    assign_gp_and_properties(space, material_model=LinearElasticMaterial(E, nue, 1., kappa=5/6),
                             properties={'t': t, 'rho': 1.}, full_quad=True)

    # --- Reflection planes: transverse displacement and director vanish ---
    # The equator and opening remain free. No ring support is introduced.
    boundaries = []
    cuts = ((LineString([(0., 1.), (0., 0.)]), 1),
            (LineString([(1., 0.), (1., 1.)]), 0))
    if method != 'strong':
        factor = 4.*penalty_factor if method == 'penalty' else 4.
        for line, plane_axis in cuts:
            plane_normal = rotation[:, plane_axis]
            condition = WeakKLSymmetryBoundary(line, plane_normal, method=method, safety_factor=factor)
            boundaries.append(condition)
    space.boundary_conditions = boundaries

    # --- Bulk and boundary assembly ---
    for cell in space.elements:
        shell.element_routine(cell)
        contributions = [bc.integrate_contribution(cell) for bc in boundaries]
        for matrix, force, _ in contributions:
            cell.K_el += matrix
            cell.f_el += force

    # Apply each corner force once; summing over containing cells duplicates point loads.
    for point, force in (((0., 0.), quarter_load*rotation[:, 0]),
                         ((1., 0.), -quarter_load*rotation[:, 1])):
        cell = next(cell for cell in space.elements if cell.contains(*point))
        cell.f_el += PointLoad(point, np.r_[force, np.zeros(dof-3)]).compute_contribution(cell, dof)

    K = MatrixAssembler(dof).assemble(space, 'K_el', use_sparse=True)
    f = VectorAssembler(dof).assemble(space, 'f_el')

    # --- Solve in the benchmark frame to impose one vertical rigid-translation gauge ---
    block = rotation
    transform = kron(eye(len(space.basis)), block, format='csr')
    local_K, local_f = transform.T@K@transform, transform.T@f
    corner = select_patch_boundary(space, 'umin') & select_patch_boundary(space, 'vmin')
    fixed = {dof*min(corner)+2}
    reduced, rhs, free = block_partitioning_dbc(local_K, local_f, sorted(fixed), np.zeros(len(fixed)))
    local_solution = np.zeros(len(f))
    local_solution[free] = solve(reduced, rhs, scaling=True)
    solution = np.asarray(transform@local_solution).reshape(-1, dof)
    update_cps(space, solution.ravel(), dof_per_cp=dof, formulation='KL')

    # --- Radial displacement and independent geometry diagnostics ---
    radial_a = rotation[:, 0]@space.eval_displacement(0., 0.)
    radial_b = rotation[:, 1]@space.eval_displacement(1., 0.)
    samples = np.array([(u, v) for v in np.linspace(0., 1., 41) for u in np.linspace(0., 1., 41)])
    radius_error = np.max(np.abs(np.linalg.norm(geometry.evaluate(samples), axis=1)-radius))
    residual = np.linalg.norm((local_K@local_solution-local_f)[free])/np.linalg.norm(local_f[free])
    results = dict(n=n, radial_a=radial_a, radial_b=radial_b, residual=residual,
                       radius_error=radius_error, reference_displacement=reference_displacement,
                       relative_benchmark_difference=abs(radial_a-reference_displacement)/abs(reference_displacement),
                       benchmark_difference=abs(radial_a-reference_displacement))

    # --- One result window; deformation and overlays can be toggled ---
    if show:
        plot_solution(space, deformation_scale=deformation_scale)

    return space, results


if __name__ == '__main__':
    _, results = run()
    print(results)
