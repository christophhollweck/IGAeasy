"""Trimmed Cook's membrane: same physical panel, material and load as the fitted setup."""
import numpy as np
from shapely.geometry import LineString, Polygon
from IGAeasy.geometry.get_geometry import polynomial_shell
from IGAeasy.trimming.Brep import BRep
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.elements.membrane2p import Membrane
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.constraints.membrane_weak import WeakMembraneBoundary
from IGAeasy.loads import LineLoad
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.solver.direct_solver import solve
from IGAeasy.postprocessing.postprocessing import update_cps
from IGAeasy.postprocessing.visualization import plot_solution

# --- Plane-stress panel and uniform end shear ---
n_elements = 8
degree = 3
E = 100.
nue = 1/3
thickness = 1.
end_force = 1.
deformation_scale = 20.
# Rounded numerical midpoint benchmark, not an analytical error reference.
# https://dcr-st.com/verification/SB3.html (normalized u_y*E*t/P = 23.91)
reference_displacement = 23.91*end_force/(E*thickness)
# Unit parameter coordinates map to x=48*u, y=60*v.
domain = Polygon([(x/48, y/60) for x, y in
                  [(0., 0.), (48., 44.), (48., 60.), (0., 44.)]])
root_line = LineString([(0., 44/60), (0., 0.)])
end_line = LineString([(1., 44/60), (1., 1.)])


def run(n_elements=n_elements, method='nitsche', show=True):
    # --- Cut the slanted panel boundaries from a rectangular background ---
    geometry = polynomial_shell(n_elements, n_elements, degree, curvature=(0., 0.), centered=False)
    geometry.control_points[:, 0] *= 48
    geometry.control_points[:, 1] *= 60
    space = HierarchicalSpace(*geometry.knot_vectors, degree, geometry.control_points)
    space.BReps = [BRep(1, polygon=domain, outerloop=True)]
    trim(space)

    # --- Material quadrature, support and resultant-normalized end traction ---
    assign_gp_and_properties(space, material_model=LinearElasticMaterial(E, nue, 1.),
                             properties={'t': thickness, 'rho': 1.}, full_quad=True)
    membrane = Membrane()
    load = LineLoad(2, [lambda s: 0., lambda s: end_force/16.], end_line,
                    id=1, at_patch_edge=True)
    boundary = WeakMembraneBoundary(root_line, lambda x, y: np.zeros(2), method=method,
                                    safety_factor=400. if method == 'penalty' else 4.)
    space.boundary_conditions = [boundary]

    # --- Element contributions and sparse assembly ---
    for cell in space.elements:
        if cell.void or not cell.active:
            continue
        membrane.element_routine(cell)
        cell.f_el += load.integrate_contribution(cell, 2)[1]
        matrix, force, _ = boundary.integrate_contribution(cell)
        cell.K_el += matrix
        cell.f_el += force
    K = MatrixAssembler(2).assemble(space, 'K_el', use_sparse=True)
    f = VectorAssembler(2).assemble(space, 'f_el')
    coefficients = solve(K, f, scaling=True)
    update_cps(space, coefficients, dof_per_cp=2, dimensions=2)

    # --- Loaded-edge midpoint, kept distinct from the upper-corner response ---
    displacement = space.eval_displacement(1., 52/60)[1]
    results = dict(n=n_elements, displacement=displacement,
                   upper_corner_displacement=space.eval_displacement(1., 1.)[1],
                   reference_displacement=reference_displacement,
                   normalized_displacement=displacement*E*thickness/end_force,
                   normalized_reference_interval=(23.90, 23.96),
                   benchmark_difference=abs(displacement-reference_displacement),
                   relative_benchmark_difference=abs(displacement-reference_displacement)/abs(reference_displacement),
                   residual=np.linalg.norm(K@coefficients-f)/np.linalg.norm(f))
    if show:
        plot_solution(space, deformation_scale=deformation_scale)
    return space, results


if __name__ == '__main__':
    _, results = run()
    print(results)
