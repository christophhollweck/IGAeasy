"""Trimmed flat cantilever with an exact cylindrical Reissner-Mindlin solution."""
from IGAeasy.trimming.Brep import BRep
import numpy as np
from shapely.geometry import Point, box, LineString
from shapely.affinity import rotate

from IGAeasy.geometry.get_geometry import polynomial_shell
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.elements.RMshell5p import RMShell
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.constraints.shell_weak import WeakRM5Boundary
from IGAeasy.loads import LineLoad
from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.postprocessing.postprocessing import update_cps
from IGAeasy.solver.direct_solver import solve
from IGAeasy.postprocessing.visualization import plot_solution

# --- Geometry, discretization and material ---
n_elements = 8
degree = 3
curvature = (0., 0.)
mixed_terms = (0., 0.)
angle = .6
E = 100.  # Young's modulus
nue = 0.  # No Poisson coupling: cylindrical bending is an exact plate solution.
thickness = .1
deformation_scale = 2.

# A slanted strip cuts the background mesh. Only its left end is clamped.
trim_angle = 8.  # Minimum material fraction stays above 1.5% on 4/8/16 grids.
domain = rotate(box(1/12, 1/3, 11/12, 2/3), trim_angle, origin=(.5, .5))
root_line = rotate(LineString([(1/12, 2/3), (1/12, 1/3)]), trim_angle, origin=(.5, .5))
tip = rotate(Point(11/12, .5), trim_angle, origin=(.5, .5))
end_force = -.001
length = 5/6
width = 1/3
kappa = 5/6
trim_rotation = np.deg2rad(trim_angle)
beam_axis = np.array([np.cos(trim_rotation), np.sin(trim_rotation), 0.])
root_point = np.array([root_line.coords[-1][0]-.5, root_line.coords[-1][1]-.5, 0.])
end_line = rotate(LineString([(11/12, 1/3), (11/12, 2/3)]), trim_angle, origin=(.5, .5))
bending_stiffness = E*width*thickness**3/12
shear_stiffness = kappa*E/(2*(1+nue))*width*thickness
reference_displacement = end_force*(length**3/(3*bending_stiffness)+length/shear_stiffness)


def exact_deflection(x):
    """Cylindrical RM bending plus transverse shear under a total end force."""
    return end_force*(x*x*(3*length-x)/(6*bending_stiffness)+x/shear_stiffness)


def run(n_elements=n_elements, method='nitsche', show=True, rotation_angle=angle):
    if method not in ('nitsche', 'penalty'):
        raise ValueError('Use nitsche or penalty')

    # --- Flat background patch and slanted material region ---
    rotation = rotation_matrix(rotation_angle, (1., 2., 3.))
    geometry = polynomial_shell(n_elements, n_elements, degree, curvature, mixed=mixed_terms)
    space = HierarchicalSpace(*geometry.knot_vectors, degree, geometry.control_points@rotation.T)
    space.BReps = [BRep(1, polygon=domain, outerloop=True)]
    trim(space)

    # --- Full quadrature on the physical material region ---
    assign_gp_and_properties(space, material_model=LinearElasticMaterial(E, nue, 1., kappa=kappa),
                             properties={'t': thickness, 'rho': 1.}, full_quad=True)
    shell = RMShell()
    dof = shell.dof_per_cp
    traction = np.r_[end_force/width*rotation[:, 2], 0., 0.]
    load = LineLoad(dof, [lambda s, value=value: value for value in traction], end_line, id=1)

    # --- Clamp translations and physical director only at the root ---
    boundaries = [WeakRM5Boundary(root_line, lambda x, y, z: np.zeros(6), method=method,
                                  safety_factor=400. if method == 'penalty' else 4.)]
    space.boundary_conditions = boundaries

    # --- Element contributions, then sparse global assembly ---
    cells = [cell for cell in space.elements if cell.active and not cell.void]
    for cell in cells:
        shell.element_routine(cell)
        cell.f_el += load.integrate_contribution(cell, dof)[1]
        contributions = [bc.integrate_contribution(cell) for bc in boundaries]
        for matrix, force, _ in contributions:
            cell.K_el += matrix
            cell.f_el += force
    K = MatrixAssembler(dof).assemble(space, 'K_el', use_sparse=True)
    f = VectorAssembler(dof).assemble(space, 'f_el')

    # --- Solve and evaluate the free-end displacement ---
    coefficients = solve(K, f, scaling=True).reshape(-1, dof)
    fractions = [sum(poly.area for poly in cell.material_polygons)
                 /((cell.u_max-cell.u_min)*(cell.v_max-cell.v_min)) if cell.trimmed else 1.
                 for cell in cells]
    update_cps(space, coefficients.ravel(), dof_per_cp=dof, formulation='RM5')
    displacement = rotation[:, 2]@space.eval_displacement(tip.x, tip.y)
    def exact_displacement(x, y, z):
        local = rotation.T@np.array([x, y, z])
        coordinate = beam_axis@(local-root_point)
        return rotation[:, 2]*exact_deflection(coordinate)

    norm = SurfaceL2Quadrature(space, order=degree+3)
    results = dict(n=n_elements, displacement=displacement,
                   reference_displacement=reference_displacement,
                   absolute_error=abs(displacement-reference_displacement),
                   relative_error=abs(displacement-reference_displacement)/abs(reference_displacement),
                   l2=norm.error(coefficients[:, :3], exact_displacement),
                   residual=np.linalg.norm(K@coefficients.ravel()-f)/np.linalg.norm(f),
                   min_parametric_fraction=min(fractions), n_basis=len(space.basis))

    # --- One result window, including trim edges and support controls ---
    if show:
        plot_solution(space, deformation_scale=deformation_scale)
    return space, results



if __name__ == '__main__':
    _, results = run()
    print(results)
