"""Width-trimmed Raasch hook with a numerical tip-displacement reference."""
import numpy as np
from shapely.geometry import box, LineString
from IGAeasy.geometry.get_geometry import raasch_hook
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.trimming.Brep import BRep
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.elements.RMshell5p import RMShell
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.constraints.shell_weak import WeakRM5Boundary
from IGAeasy.loads import LineLoad
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.solver.direct_solver import solve
from IGAeasy.postprocessing.postprocessing import update_cps
from IGAeasy.postprocessing.visualization import plot_solution

# --- Geometry, material and load (inch, pound) ---
refinements = 0
angle = .6
E = 3300.
nue = .35
thickness = 2.
width = 20.
end_force = 1.
deformation_scale = 2.
reference_displacement = 5.020  # Numerical 3D reference, not an exact shell solution.
# https://abaqus.uclouvain.be/English/SIMACAEBMKRefMap/simabmk-c-raasch.htm
domain = box(0., 1/12, 1., 11/12)
root_line = LineString([(0., 11/12), (0., 1/12)])
end_line = LineString([(1., 1/12), (1., 11/12)])


def run(refinements=refinements, method='nitsche', show=True, rotation_angle=angle):
    # --- Trim an extended, fixed two-arc background geometry ---
    rotation = rotation_matrix(rotation_angle, (1., 2., 3.))
    geometry = raasch_hook(refinements)
    space = HierarchicalSpace(*geometry.knot_vectors, 3, geometry.control_points@rotation.T)
    space.BReps = [BRep(1, polygon=domain, outerloop=True)]
    trim(space)

    # --- Full material quadrature and end shear along the strip width ---
    assign_gp_and_properties(space, material_model=LinearElasticMaterial(E, nue, 1., kappa=5/6),
                             properties={'t': thickness, 'rho': 1.}, full_quad=True)
    shell = RMShell()
    dof = shell.dof_per_cp
    traction = np.r_[end_force/width*rotation[:, 2], 0., 0.]
    load = LineLoad(dof, [lambda s, value=value: value for value in traction], end_line, id=1)
    boundaries = [WeakRM5Boundary(root_line, lambda x, y, z: np.zeros(6), method=method,
                                  safety_factor=400. if method == 'penalty' else 4.)]
    space.boundary_conditions = boundaries

    # --- Element contributions and sparse global assembly ---
    cells = [cell for cell in space.elements if cell.active and not cell.void]
    for cell in cells:
        shell.element_routine(cell)
        cell.f_el += load.integrate_contribution(cell, dof)[1]
        matrix, force, _ = boundaries[0].integrate_contribution(cell)
        cell.K_el += matrix
        cell.f_el += force
    K = MatrixAssembler(dof).assemble(space, 'K_el', use_sparse=True)
    f = VectorAssembler(dof).assemble(space, 'f_el')
    coefficients = solve(K, f, scaling=True).reshape(-1, dof)
    update_cps(space, coefficients.ravel(), dof_per_cp=dof, formulation='RM5')

    # --- Loaded-edge midpoint response in the original load direction ---
    displacement = rotation[:, 2]@space.eval_displacement(1., .5)
    parameters = np.linspace(0., 1., 1001)
    points = geometry.evaluate(np.column_stack((parameters, np.full_like(parameters, .5))))
    join = (14*np.pi/3)/(14*np.pi/3+46*5*np.pi/6)
    centers = np.where((parameters <= join)[:, None], [0., 14.], [30*np.sqrt(3), -16.])
    radii = np.where(parameters <= join, 14., 46.)
    radius_error = np.max(np.abs(np.linalg.norm(points[:, :2]-centers, axis=1)-radii))
    results = dict(refinements=refinements, displacement=displacement,
                   reference_displacement=reference_displacement,
                   benchmark_difference=abs(displacement-reference_displacement),
                   relative_benchmark_difference=abs(displacement-reference_displacement)/reference_displacement,
                   radius_error=radius_error,
                   residual=np.linalg.norm(K@coefficients.ravel()-f)/np.linalg.norm(f))
    if show:
        plot_solution(space, deformation_scale=deformation_scale)
    return space, results


if __name__ == '__main__':
    _, results = run()
    print(results)
