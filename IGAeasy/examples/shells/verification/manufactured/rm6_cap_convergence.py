"""Independent RM6 equilibrium on a doubly curved graph, including body moments."""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, box

from IGAeasy.geometry.get_geometry import polynomial_shell
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.trimming.Brep import BRep
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.elements.RMshell6p import RMShell
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.constraints.shell_weak import WeakRM6Boundary
from IGAeasy.loads import SurfaceLoad
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.solver.direct_solver import solve
from IGAeasy.postprocessing.postprocessing import update_cps
from IGAeasy.postprocessing.visualization import plot_solution
from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature, SurfaceDirectorL2Quadrature
from IGAeasy.manufactured_solutions.rm6_cap_reference import RM6CapReference


net_sizes = (2, 4, 8, 16)
curvature = (.4, -.2)  # Opposite signs give a saddle; equal signs give a cap.
angle = .6


def solve_level(n, family='THB', rotation_angle=angle, domain=None, degree=3):
    if family not in ('THB', 'LR'):
        raise ValueError('Choose THB or LR')

    # --- Exact geometry, independent six-component force/moment reference ---
    rotation = rotation_matrix(rotation_angle, (1., 2., 3.))
    reference = RM6CapReference(curvature=curvature, rotation=rotation)
    geometry = polynomial_shell(n, n, degree, curvature, centered=False)
    cps = geometry.control_points@rotation.T
    space = (HierarchicalSpace(*geometry.knot_vectors, degree, cps) if family == 'THB'
             else LRSpline(degree, *geometry.knot_vectors, cps))
    material_domain = box(0., 0., 1., 1.) if domain is None else domain
    if domain is not None:
        space.BReps = [BRep(1, polygon=material_domain, outerloop=True)]
        trim(space)

    # --- Full quadrature and nonhomogeneous weak displacement/rotation data ---
    material = LinearElasticMaterial(reference.E, reference.nu, 1.)
    assign_gp_and_properties(space, material, {'t': reference.thickness, 'rho': 1.}, full_quad=True)
    rule = (reference.nodes, reference.weights)
    boundary = WeakRM6Boundary(LineString(material_domain.exterior.coords), reference.field, rule)
    load = SurfaceLoad(6, reference.body)
    shell = RMShell()
    space.boundary_conditions = [boundary]
    for cell in space.elements:
        if cell.void or not getattr(cell, 'active', True):
            continue
        shell.element_routine(cell, rule)
        cell.K_bulk = cell.K_el
        matrix, force, _ = boundary.integrate_contribution(cell)
        cell.K_el = cell.K_el+matrix
        cell.f_el = force+load.integrate_contribution(cell, 6)

    # --- Solution and independent physical displacement/director norms ---
    K = MatrixAssembler(6).assemble(space, 'K_el', use_sparse=True)
    f = VectorAssembler(6).assemble(space, 'f_el')
    coefficients = solve(K, f, scaling=True).reshape(-1, 6)
    norm = SurfaceL2Quadrature(space, order=degree+3)
    director = SurfaceDirectorL2Quadrature(space, 'RM6', order=degree+3)
    drilling, energy = 0., 0.
    for cell in space.elements:
        if cell.void or not getattr(cell, 'active', True):
            continue
        local = coefficients[cell.bfnums].ravel()
        energy += .5*local@cell.K_bulk@local
        for gp in cell.gausspoints:
            drilling += .5*.05*reference.E*(gp.Bdrill@local).item()**2*sum(gp.volume_weights)
    row = dict(n=n, h=1/n, l2=norm.error(coefficients[:, :3], lambda *x: reference.field(*x)[:3]),
               director_l2=director.error(coefficients, reference.director),
               rotation_l2=norm.error(coefficients[:, 3:]),
               drilling_energy=drilling, drilling_fraction=drilling/energy,
               residual=np.linalg.norm(K@coefficients.ravel()-f)/np.linalg.norm(f))
    update_cps(space, coefficients.ravel(), dof_per_cp=6, formulation='RM6')
    return space, row


def main(show=True, sizes=net_sizes):
    rows = []
    for n in sizes:
        print(f'RM6 double curvature: {n} x {n}...', flush=True)
        space, row = solve_level(n)
        rows.append(row)
        print(row, flush=True)
    figure, axis = plt.subplots()
    for key in ('l2', 'director_l2', 'rotation_l2'):
        axis.loglog([r['h'] for r in rows], [r[key] for r in rows], 'o-', label=key)
    axis.set(xlabel='Element size h', ylabel='Physical L2 error')
    axis.grid(True, which='both')
    axis.legend()
    if show:
        plot_solution(space, deformation_scale=50.)
        plt.show()
    else:
        plt.close(figure)
    return rows


if __name__ == '__main__':
    main()
