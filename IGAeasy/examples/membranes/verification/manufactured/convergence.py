from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.elements.membrane2p import Membrane
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.constraints.membrane_weak import WeakMembraneBoundary
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.loads import SurfaceLoad
from IGAeasy.solver.direct_solver import solve
from IGAeasy.postprocessing import postprocessing
from IGAeasy.error_norms.norms_poisson import compute_L2_H1_errors
from IGAeasy.postprocessing.visualization import build_surface_data, plot_surface_results
from shapely.geometry import box, LineString, Point
from shapely.affinity import rotate
from shapely.geometry.polygon import orient
from IGAeasy.trimming.Brep import BRep
import numpy as np
import matplotlib.pyplot as plt

'''Manufactured membrane solution on a disk with a rotated rectangular hole.'''

n = 4
p = 2
n_refinements = 3
methods = ['penalty', 'nitsche']
E = 100.
nue = .3
t = .2
safety_factor = 4.  # Two boundary objects can contribute to the same cut cell.
plot_geometry = True
plot_solution = True
deformation_scale = 0  # display fields on the undeformed geometry
displacement_amplitude = .01
circle_radius = .45
rectangle_width = .30
rectangle_height = .20
rectangle_angle = 30  # degrees
circle_resolution = 64  # segments per quadrant; polygonal circle approximation

outer = orient(Point(.5, .5).buffer(circle_radius, resolution=circle_resolution), sign=1.)
hole = box(.5-rectangle_width/2, .5-rectangle_height/2,
           .5+rectangle_width/2, .5+rectangle_height/2)
hole = orient(rotate(hole, rectangle_angle, origin=(.5, .5)), sign=-1.)
if not outer.contains(hole):
    raise ValueError('The rotated rectangular hole must lie strictly inside the outer circle')
lines = [LineString(outer.exterior.coords), LineString(hole.exterior.coords)]


def exact_displacement(x, y):
    return displacement_amplitude*np.array([np.sin(np.pi*x)*np.sin(np.pi*y), 0.])


def exact_gradient_x(x, y):
    return displacement_amplitude*np.pi*np.array([np.cos(np.pi*x)*np.sin(np.pi*y),
                           np.sin(np.pi*x)*np.cos(np.pi*y)])


# -div(t*sigma(u_exact)), plane stress with engineering shear.
C11 = E/(1-nue**2)
C12 = nue*C11
G = E/(2*(1+nue))
load = SurfaceLoad(2, [
    lambda x, y: displacement_amplitude*t*(C11+G)*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y),
    lambda x, y: -displacement_amplitude*t*(C12+G)*np.pi**2*np.cos(np.pi*x)*np.cos(np.pi*y),
])

def main(show=True, refinements=n_refinements):
    h_list = []
    errors = {method: [] for method in methods}
    geometry_shown = False

    for level in range(refinements+1):
        n_current = n*2**level
        print(f'Preparing shared geometry and bulk matrices: {n_current} x {n_current}...', flush=True)
        h_list.append(1/n_current)
        Bspl = get_geometry.surface(n_current, n_current, 1, 1, 1, 1, p-1, p,
                                   distortion=0, n_waves=0)
        RS = HierarchicalSpace(*Bspl.knot_vectors, p, Bspl.control_points)
        RS.BReps = [BRep(1, polygon=outer, outerloop=True),
                    BRep(2, polygon=hole, outerloop=False)]
        trim(RS)
        assign_gp_and_properties(RS, material_model=LinearElasticMaterial(E, nue, 1.),
                                 properties={'t': t, 'rho': 1.}, full_quad=True)

        bulk_data = {}
        for cell in RS.elements:
            if cell.void or not cell.active:
                continue
            Membrane().element_routine(cell)
            cell.f_el += load.integrate_contribution(cell, 2)
            bulk_data[id(cell)] = (cell.K_el.copy(), cell.f_el.copy())
        for method in methods:
            print(f'{method}: computing {n_current} x {n_current}...', flush=True)
            boundaries = [WeakMembraneBoundary(line, exact_displacement, method=method,
                                              safety_factor=safety_factor) for line in lines]
            stabilization = []
            for cell in RS.elements:
                if cell.void or not cell.active:
                    continue
                bulk_K, bulk_f = bulk_data[id(cell)]
                cell.K_el = bulk_K.copy()
                cell.f_el = bulk_f.copy()
                # Estimate every boundary contribution using the original bulk K.
                contributions = [boundary.integrate_contribution(cell) for boundary in boundaries]
                for addition, force, gamma in contributions:
                    cell.K_el += addition
                    cell.f_el += force
                    if gamma > 0:
                        stabilization.append(gamma)
            K = MatrixAssembler(dof_per_cp=2).assemble(RS, 'K_el', use_sparse=True)
            f = VectorAssembler(dof_per_cp=2).assemble(RS, 'f_el')
            displacement = solve(K, f, scaling='auto')
            postprocessing.update_cps(RS, displacement, dof_per_cp=2, dimensions=2)
            # Independent higher-order material quadrature for both displacement components.
            error_x, _ = compute_L2_H1_errors(RS, lambda x, y: exact_displacement(x, y)[0],
                quadrature_order=p+3, relative=False, field_component=0)
            error_y, _ = compute_L2_H1_errors(RS, lambda x, y: 0.,
                quadrature_order=p+3, relative=False, field_component=1)
            error = np.hypot(error_x, error_y)
            errors[method].append(error)
            print(f'L2 displacement error={error:.6e}, gamma range='
                  f'{min(stabilization):.3e} .. {max(stabilization):.3e}', flush=True)

            if show and plot_geometry and not geometry_shown:
                geometry_mesh, element_boundaries = build_surface_data(RS, deformation_scale=0,
                    samples_per_element=4, fields=('displacement_magnitude',))
                plot_surface_results(geometry_mesh, element_boundaries,
                                     fields=('displacement_magnitude',), deformation_scale=0)
                geometry_shown = True
            if plot_solution and level == n_refinements and method == 'nitsche':
                mesh, element_boundaries = build_surface_data(RS, deformation_scale=deformation_scale,
                    samples_per_element=4, fields=('ux', 'uy'))
                plot_surface_results(mesh, element_boundaries, fields=('ux', 'uy'),
                                     deformation_scale=deformation_scale,
                                     scalar_limits={'ux': (0., displacement_amplitude),
                                                    'uy': (-displacement_amplitude, displacement_amplitude)})

    plt.figure(figsize=(7, 5))
    for method, values in errors.items():
        rates = np.log2(np.array(values[:-1])/np.array(values[1:]))
        print(method, 'L2 convergence rates:', np.round(rates, 3))
        plt.loglog(h_list, values, 'o-', label=method)
    reference_h = np.array(h_list)
    plt.loglog(reference_h, errors['nitsche'][-1]*(reference_h/reference_h[-1])**(p+1),
               'k--', label=f'reference slope h^{p+1}')
    plt.xlabel('Nominal element size h')
    plt.ylabel('Absolute L2 displacement error against manufactured solution')
    plt.grid(True, which='both', alpha=.3)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == '__main__':
    main()
