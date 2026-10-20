"""Manufactured RM5 solution on a rotated saddle with a polygonal circular hole."""
from IGAeasy.trimming.Brep import BRep
import numpy as np
from shapely.geometry import Point, box, LineString
from shapely.geometry.polygon import orient

from IGAeasy.geometry.get_geometry import polynomial_shell
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.elements.RMshell5p import RMShell
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.constraints.shell_weak import WeakRM5Boundary
from IGAeasy.loads import SurfaceLoad
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.postprocessing.postprocessing import update_cps
from IGAeasy.solver.direct_solver import solve
from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature, SurfaceDirectorL2Quadrature
from IGAeasy.manufactured_solutions.rm_paraboloid_reference import ParaboloidRMReference
from IGAeasy.postprocessing.visualization import plot_solution

# --- Geometry, discretization and material ---
n_elements = 8
degree = 3
curvature = (.4, -.2)
mixed_terms = (1.2, -.8)
angle = .6
E = 100.  # Young's modulus
nue = .3  # Poisson's ratio
thickness = .1
deformation_scale = 50.

# Fix the polygon for the whole study: geometry approximation is not mesh refinement.
# This radius keeps the material fractions above 1/9 on the 4/8/16 grids.
domain = box(1/12, 1/12, 11/12, 11/12).difference(Point(.5, .5).buffer(.205, quad_segs=12))


def run(n_elements=n_elements, method='nitsche', show=True, material_domain=domain,
        fitted_rectangle=False, rotation_angle=angle):
    if method not in ('nitsche', 'penalty'):
        raise ValueError('Use nitsche or penalty')

    # --- Analytic physical fields and independently derived equilibrium loads ---
    rotation = rotation_matrix(rotation_angle, (1., 2., 3.))
    reference = ParaboloidRMReference(curvature, E, nue, thickness, .001, rotation,
        centered=True, director_amplitude=.0003, enriched=True, mixed=mixed_terms)

    # --- Either cut a background patch or build the identical rectangle directly ---
    bounds = material_domain.bounds if fitted_rectangle else (0., 0., 1., 1.)
    if fitted_rectangle and not material_domain.equals(box(*bounds)):
        raise ValueError('A fitted comparison requires a rectangle without holes')
    geometry = polynomial_shell(n_elements, n_elements, degree, curvature,
                                mixed=mixed_terms, bounds=bounds)
    space = HierarchicalSpace(*geometry.knot_vectors, degree, geometry.control_points@rotation.T)
    material_domain = orient(material_domain, sign=1.)
    if not fitted_rectangle:
        space.BReps = [BRep(1, polygon=material_domain, outerloop=True)]
        trim(space)

    # --- Full quadrature on the physical material region ---
    assign_gp_and_properties(space, material_model=LinearElasticMaterial(E, nue, 1., kappa=5/6),
                             properties={'t': thickness, 'rho': 1.}, full_quad=True)
    shell = RMShell()
    dof = shell.dof_per_cp
    load = SurfaceLoad(dof, reference.body)

    # --- Prescribe displacement and physical director on outer and inner boundaries ---
    boundaries = [WeakRM5Boundary(LineString(ring.coords), reference.field, method=method,
                                  safety_factor=400. if method == 'penalty' else 4.)
                  for ring in (material_domain.exterior, *material_domain.interiors)]
    space.boundary_conditions = boundaries

    # --- Element contributions, then sparse global assembly ---
    cells = [cell for cell in space.elements if cell.active and not cell.void]
    for cell in cells:
        shell.element_routine(cell)
        cell.f_el += load.integrate_contribution(cell, dof)
        contributions = [bc.integrate_contribution(cell) for bc in boundaries]
        for matrix, force, _ in contributions:
            cell.K_el += matrix
            cell.f_el += force
    K = MatrixAssembler(dof).assemble(space, 'K_el', use_sparse=True)
    f = VectorAssembler(dof).assemble(space, 'f_el')

    # --- Solve and assess errors with independent, higher-order quadrature ---
    coefficients = solve(K, f, scaling=True).reshape(-1, dof)
    norm = SurfaceL2Quadrature(space, order=degree+3)
    director_norm = SurfaceDirectorL2Quadrature(space, 'RM5', order=degree+3)
    fractions = [sum(poly.area for poly in cell.material_polygons)
                 /((cell.u_max-cell.u_min)*(cell.v_max-cell.v_min)) if cell.trimmed else 1.
                 for cell in cells]
    results = dict(n=n_elements, h=(bounds[2]-bounds[0])/n_elements,
                   l2=norm.error(coefficients[:, :3], reference.displacement),
                   director_l2=director_norm.error(coefficients, reference.director),
                   residual=np.linalg.norm(K@coefficients.ravel()-f)/np.linalg.norm(f),
                   min_parametric_fraction=min(fractions), n_basis=len(space.basis))
    update_cps(space, coefficients.ravel(), dof_per_cp=dof, formulation='RM5')

    # --- One result window, including trim edges and support controls ---
    if show:
        plot_solution(space, deformation_scale=deformation_scale)
    return space, results


if __name__ == '__main__':
    _, results = run()
    print(results)
