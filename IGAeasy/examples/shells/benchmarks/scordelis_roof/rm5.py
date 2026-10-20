"""Scordelis-Lo roof with RM5: a complete single-mesh setup."""
import numpy as np
from scipy.sparse import block_diag, kron, eye
from shapely.geometry import LineString

from IGAeasy.geometry import get_geometry
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.constraints.boundary_selection import select_patch_boundary
from IGAeasy.constraints.dirichlet import block_partitioning_dbc
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.postprocessing.postprocessing import update_cps
from IGAeasy.solver.direct_solver import solve
from IGAeasy.postprocessing.visualization import plot_solution
from IGAeasy.elements.RMshell5p import RMShell
from IGAeasy.constraints.shell_weak import WeakRM5Boundary
from IGAeasy.loads import SurfaceLoad


# --- Mesh, material and loading ---
n_elements = 8
degree = 2
base_elements = 4  # Fit once; refinement preserves this geometry.
E = 4.32e8  # Young's modulus
nue = 0.  # Poisson's ratio
thickness = .25
vertical_load = -90.
angle = .6
rotation_axis = (1., 2., 3.)
deformation_scale = 20.
reference_displacement = -0.3024  # Rounded numerical benchmark, not an exact solution.


def run(n_elements=n_elements, method='nitsche', show=True, rotation_angle=angle):
    # --- Fixed full-roof geometry and nested knot insertion ---
    if method not in ('nitsche', 'penalty', 'strong'):
        raise ValueError('Use nitsche, penalty or strong')
    ratio = n_elements/base_elements
    if ratio < 1 or not np.isclose(np.log2(ratio), round(np.log2(ratio))):
        raise ValueError('Use a dyadic refinement of base_elements')
    geometry = get_geometry.scordelis_lo_roof(base_elements//2, base_elements//2, degree, full=True)
    for _ in range(round(np.log2(ratio))):
        for direction in (0, 1):
            knots = np.unique(geometry.knot_vectors[direction])
            geometry.insert_knots(direction, (knots[:-1]+knots[1:])/2)
    rotation = rotation_matrix(rotation_angle, rotation_axis)
    space = HierarchicalSpace(*geometry.knot_vectors, degree, geometry.control_points@rotation.T)

    # --- Material and full bulk quadrature ---
    shell = RMShell()
    dof = shell.dof_per_cp
    assign_gp_and_properties(space, material_model=LinearElasticMaterial(E, nue, 1.),
                             properties={'t': thickness, 'rho': 1.}, full_quad=True)

    # --- End diaphragms: local x/z displacement fixed, longitudinal motion free ---
    lines = [LineString([(0., 0.), (1., 0.)]), LineString([(1., 1.), (0., 1.)])]
    directions = np.column_stack((rotation[:, [0, 2]].T, np.zeros((2, 3))))
    factor = 400. if method == 'penalty' else 4.
    boundaries = [] if method == 'strong' else [
        WeakRM5Boundary(line, lambda x, y, z: np.zeros(6),
            directions=directions, method=method, safety_factor=factor) for line in lines]
    space.boundary_conditions = boundaries
    vertical = rotation[:, 2]
    load = SurfaceLoad(dof, lambda u, v: np.r_[vertical_load*vertical, np.zeros(dof-3)])

    # --- Element stiffness, load and weak boundary contributions ---
    for cell in space.elements:
        shell.element_routine(cell)
        cell.f_el += load.integrate_contribution(cell, dof)
        # Every boundary estimate uses the unchanged bulk stiffness.
        contributions = [boundary.integrate_contribution(cell) for boundary in boundaries]
        for matrix, force, _ in contributions:
            cell.K_el += matrix
            cell.f_el += force

    # --- Assemble and remove the remaining longitudinal rigid translation ---
    K = MatrixAssembler(dof).assemble(space, 'K_el', use_sparse=True)
    f = VectorAssembler(dof).assemble(space, 'f_el')
    block = block_diag((rotation, eye(2)))
    transform = kron(eye(len(space.basis)), block, format='csr')
    local_K, local_f = transform.T@K@transform, transform.T@f
    corner = min(select_patch_boundary(space, 'umin') & select_patch_boundary(space, 'vmin'))
    fixed = {dof*corner+1}
    if method == 'strong':
        ends = select_patch_boundary(space, 'vmin') | select_patch_boundary(space, 'vmax')
        fixed.update(dof*i+j for i in ends for j in (0, 2))
    reduced, rhs, free = block_partitioning_dbc(local_K, local_f, prescribed_DOFs=sorted(fixed))
    local_solution = np.zeros(len(f))
    local_solution[free] = solve(reduced, rhs, scaling=True)
    coefficients = np.asarray(transform@local_solution).reshape(-1, dof)
    update_cps(space, coefficients.ravel(), dof_per_cp=dof, formulation='RM5')

    # --- Free-edge midpoint response in the original vertical direction ---
    displacement = vertical@space.eval_displacement(1., .5)
    residual = np.linalg.norm(reduced@local_solution[free]-rhs)/np.linalg.norm(rhs)
    results = dict(n=n_elements, displacement=displacement, residual=residual,
                   reference_displacement=reference_displacement,
                   relative_benchmark_difference=abs(displacement-reference_displacement)/abs(reference_displacement),
                   benchmark_difference=abs(displacement-reference_displacement))

    # --- One result window; deformation and overlays can be toggled ---
    if show:
        plot_solution(space, deformation_scale=deformation_scale)

    return space, results


if __name__ == '__main__':
    _, results = run()
    print(results)
