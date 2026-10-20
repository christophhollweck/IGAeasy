"""Uniformly loaded simply supported unit plate: RM5_hier."""
import numpy as np
from shapely.geometry import LineString

from IGAeasy.geometry import get_geometry
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.postprocessing.postprocessing import update_cps
from IGAeasy.solver.direct_solver import solve
from IGAeasy.postprocessing.visualization import plot_solution
from IGAeasy.elements.RMshell5p_hier import RMShell
from IGAeasy.constraints.rm5_hier_weak import WeakRM5HierBoundary
from IGAeasy.loads import SurfaceLoad
from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature
from IGAeasy.examples.shells.benchmarks.simply_supported_plate.reference import navier_center, navier_solution

n_elements = 8
degree = 3
E = 10920.  # Young's modulus
nue = .3  # Poisson's ratio
thickness = .1
pressure = -1.
angle = .6
rotation_axis = (1., 2., 3.)
deformation_scale = 30.
reference_parameters = dict(E=E, nue=nue, thickness=thickness, pressure=pressure)
reference_displacement = navier_center(include_shear=True, **reference_parameters)


def run(n_elements=n_elements, method='nitsche', show=True, rotation_angle=angle):
    if method not in ('nitsche', 'penalty'):
        raise ValueError('This example uses nitsche or penalty')

    # --- Unit square, rotated together with the load and supports ---
    geometry = get_geometry.surface(n_elements, n_elements, 1., 1., 1., 1.,
                                    degree-1, degree, distortion=0., n_waves=0)
    rotation = rotation_matrix(rotation_angle, rotation_axis)
    space = HierarchicalSpace(*geometry.knot_vectors, degree, geometry.control_points@rotation.T)

    # --- Isotropic material and full bulk quadrature ---
    shell = RMShell()
    dof = shell.dof_per_cp
    assign_gp_and_properties(space, material_model=LinearElasticMaterial(E, nue, 1., kappa=5/6),
                             properties={'t': thickness, 'rho': 1.}, full_quad=True)

    # --- Hard simple supports: translations and tangential director fixed ---
    edges = [(LineString([(0., 1.), (0., 0.)]), 0),
             (LineString([(1., 0.), (1., 1.)]), 0),
             (LineString([(0., 0.), (1., 0.)]), 1),
             (LineString([(1., 1.), (0., 1.)]), 1)]
    boundaries = []
    factor = 400. if method == 'penalty' else 4.
    for line, edge_axis in edges:
        rows = np.zeros((4, 6))
        rows[:3, :3] = rotation.T
        rows[3, 3:] = rotation[:, 1-edge_axis]
        boundaries.append(WeakRM5HierBoundary(line, lambda x, y, z: np.zeros(6),
            directions=rows, method=method, safety_factor=factor))
    space.boundary_conditions = boundaries
    vertical = rotation[:, 2]
    load = SurfaceLoad(dof, lambda u, v: np.r_[pressure*vertical, np.zeros(dof-3)])

    # --- Element stiffness, pressure and weak support terms ---
    for cell in space.elements:
        shell.element_routine(cell)
        cell.f_el += load.integrate_contribution(cell, dof)
        contributions = [boundary.integrate_contribution(cell) for boundary in boundaries]
        for matrix, force, _ in contributions:
            cell.K_el += matrix
            cell.f_el += force

    # --- Sparse global assembly and solution ---
    K = MatrixAssembler(dof).assemble(space, 'K_el', use_sparse=True)
    f = VectorAssembler(dof).assemble(space, 'f_el')
    coefficients = solve(K, f, scaling=True).reshape(-1, dof)
    update_cps(space, coefficients.ravel(), dof_per_cp=dof, formulation='RM5_hier')

    # --- Center response and independent displacement L2 norm ---
    def exact_displacement(x, y, z):
        local = rotation.T@np.array([x, y, z])
        return vertical*navier_solution(local[0], local[1], include_shear=True, **reference_parameters)

    displacement = vertical@space.eval_displacement(.5, .5)
    norm = SurfaceL2Quadrature(space, order=degree+3)
    results = dict(n=n_elements, displacement=displacement,
                   reference_displacement=reference_displacement,
                   relative_benchmark_difference=abs(displacement-reference_displacement)/abs(reference_displacement),
                   benchmark_difference=abs(displacement-reference_displacement),
                   l2=norm.error(coefficients[:, :3], exact_displacement),
                   residual=np.linalg.norm(K@coefficients.ravel()-f)/np.linalg.norm(f))
    # --- One result window; deformation and overlays can be toggled ---
    if show:
        plot_solution(space, deformation_scale=deformation_scale)

    return space, results


if __name__ == '__main__':
    _, results = run()
    print(results)
