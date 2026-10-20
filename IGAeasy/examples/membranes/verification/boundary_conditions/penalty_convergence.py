"""Penalty parameter study on a disk with a rotated rectangular hole."""
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.elements.membrane2p import Membrane
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.constraints.membrane_weak import WeakMembraneBoundary
from IGAeasy.constraints.line_helpers import get_integration_points
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.loads import SurfaceLoad
from IGAeasy.solver.direct_solver import solve
from IGAeasy.solver.linear_system import prepare, condition_estimate
from IGAeasy.postprocessing import postprocessing
from IGAeasy.error_norms.norms_poisson import compute_L2_H1_errors
from shapely.geometry import box, LineString, Point
from shapely.affinity import rotate
from shapely.geometry.polygon import orient
from IGAeasy.trimming.Brep import BRep
import numpy as np
import matplotlib.pyplot as plt
import time

n = 4
p = 2
n_refinements = 2
E = 100.
nue = .3
t = .2
displacement_amplitude = .01
safety_factor = 4.
methods = ['penalty 1', 'penalty 100', 'penalty h^-2', 'nitsche']

outer = orient(Point(.5, .5).buffer(.45, quad_segs=64), sign=1.)
hole = orient(rotate(box(.35, .40, .65, .60), 30, origin=(.5, .5)), sign=-1.)
lines = [LineString(outer.exterior.coords), LineString(hole.exterior.coords)]


def exact_displacement(x, y):
    return displacement_amplitude*np.array([np.sin(np.pi*x)*np.sin(np.pi*y), 0.])


C11 = E/(1-nue**2)
C12 = nue*C11
G = E/(2*(1+nue))
load = SurfaceLoad(2, [
    lambda x, y: displacement_amplitude*t*(C11+G)*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y),
    lambda x, y: -displacement_amplitude*t*(C12+G)*np.pi**2*np.cos(np.pi*x)*np.cos(np.pi*y),
])
def main(show=True, *, net_sizes=None, method_names=None, refinements=n_refinements,
         observer=None, observer_only=False):
    selected_methods = methods if method_names is None else list(method_names)
    h_list = []
    errors = {method: [] for method in selected_methods}
    boundary_errors = {method: [] for method in selected_methods}
    condition_estimates = {method: [] for method in selected_methods}

    study_sizes = net_sizes if net_sizes is not None else [n*2**level for level in range(refinements+1)]
    observations = []
    for level, n_current in enumerate(study_sizes):
        setup_start = time.perf_counter()
        h = 1/n_current
        h_list.append(h)
        print(f'Preparing {n_current} x {n_current}...', flush=True)
        Bspl = get_geometry.surface(n_current, n_current, 1, 1, 1, 1, p-1, p,
                                   distortion=0, n_waves=0)
        RS = HierarchicalSpace(*Bspl.knot_vectors, p, Bspl.control_points)
        RS.BReps = [BRep(1, polygon=outer, outerloop=True),
                    BRep(2, polygon=hole, outerloop=False)]
        trim(RS)
        print(f'Geometry and trimming: {time.perf_counter()-setup_start:.2f}s', flush=True)
        integration_start = time.perf_counter()
        assign_gp_and_properties(RS, material_model=LinearElasticMaterial(E, nue, 1.),
                                 properties={'t': t, 'rho': 1.}, full_quad=True)
        cells = [cell for cell in RS.elements if cell.active and not cell.void]
        bulk_data, boundary_data, boundary_samples = {}, {}, []
        for cell_index, cell in enumerate(cells):
            Membrane().element_routine(cell)
            cell.f_el += load.integrate_contribution(cell, 2)
            bulk_data[id(cell)] = (cell.K_el.copy(), cell.f_el.copy())
            if (cell_index+1) % 100 == 0:
                print(f'Bulk setup: {cell_index+1}/{len(cells)} cells, '
                      f'{time.perf_counter()-integration_start:.1f}s', flush=True)
        print(f'Quadrature and bulk matrices: {time.perf_counter()-integration_start:.2f}s', flush=True)
        boundary_start = time.perf_counter()
        for cell_index, cell in enumerate(cells):
            # All bounds refer to the original bulk stiffness, before adding boundary terms.
            contributions = [WeakMembraneBoundary(line, exact_displacement,
                            safety_factor=safety_factor).integrate_both(cell) for line in lines]
            for method in ('penalty', 'nitsche'):
                boundary_data[id(cell), method] = [entry[method] for entry in contributions]
            if (cell_index+1) % 100 == 0:
                print(f'Boundary setup: {cell_index+1}/{len(cells)} cells, '
                      f'{time.perf_counter()-boundary_start:.1f}s', flush=True)
            if observer_only:
                continue
            basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
            samples = []
            # Parametric and physical line measures coincide for this unit-square mapping.
            for line in lines:
                for coordinate, weight in get_integration_points(line, cell):
                    point = line.interpolate(coordinate)
                    values = np.array([b(point.x, point.y, cell=cell) for b in basis])
                    xy = values@np.array([b.cp[:2] for b in basis])
                    samples.append((weight, values, exact_displacement(*xy)))
            boundary_samples.append((basis, samples))

        RS.study_setup_seconds = time.perf_counter()-setup_start
        print(f'Boundary setup finished: {time.perf_counter()-boundary_start:.2f}s', flush=True)

        for method in selected_methods:
            assembly_start = time.perf_counter()
            boundary_method = 'nitsche' if method == 'nitsche' else 'penalty'
            factor = {'penalty 1': 1., 'penalty 100': 100.,
                      'penalty h^-2': h**(-2), 'nitsche': 1.}[method]
            for cell in cells:
                bulk_K, bulk_f = bulk_data[id(cell)]
                cell.K_el, cell.f_el = bulk_K.copy(), bulk_f.copy()
                for addition, force, gamma in boundary_data[id(cell), boundary_method]:
                    cell.K_el += factor*addition
                    cell.f_el += factor*force
            K = MatrixAssembler(dof_per_cp=2).assemble(RS, 'K_el', use_sparse=True)
            f = VectorAssembler(dof_per_cp=2).assemble(RS, 'f_el')
            RS.study_assembly_seconds = time.perf_counter()-assembly_start
            if observer is not None:
                observations.append(observer(RS, K, f, method, n_current, exact_displacement))
            if observer_only:
                continue
            # Retain all nonzero rows even for small cut cells and large penalties.
            reduced, _, _, _, _ = prepare(K, f, tol=0.)
            condition = condition_estimate(reduced)
            condition_estimates[method].append(condition)
            displacement = solve(K, f, scaling='auto')
            postprocessing.update_cps(RS, displacement, dof_per_cp=2, dimensions=2)
            error_x, _ = compute_L2_H1_errors(RS, lambda x, y: exact_displacement(x, y)[0],
                quadrature_order=p+3, relative=False, field_component=0)
            error_y, _ = compute_L2_H1_errors(RS, lambda x, y: 0.,
                quadrature_order=p+3, relative=False, field_component=1)
            error = np.hypot(error_x, error_y)
            errors[method].append(error)
            boundary_squared_error = 0.
            for basis, samples in boundary_samples:
                coefficients = np.array([b.displacements[:2] for b in basis])
                for weight, values, exact in samples:
                    difference = values@coefficients-exact
                    boundary_squared_error += weight*np.dot(difference, difference)
            boundary_error = np.sqrt(boundary_squared_error)
            boundary_errors[method].append(boundary_error)
            print(f'{method}: L2={error:.6e}, boundary L2={boundary_error:.6e}, '
                  f'estimated cond_1={condition:.3e}', flush=True)

    if not observer_only:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for method in selected_methods:
            rates = np.log2(np.array(errors[method][:-1])/np.array(errors[method][1:]))
            print(method, 'L2 convergence rates:', np.round(rates, 3))
            for axis, results in zip(axes, (errors, boundary_errors, condition_estimates)):
                axis.loglog(h_list, results[method], 'o-', label=method)
        reference_h = np.array(h_list)
        axes[0].loglog(reference_h, errors[selected_methods[0]][-1]*(reference_h/reference_h[-1])**(p+1),
                   'k--', label=f'reference slope h^{p+1}')
        for axis, ylabel in zip(axes, ['Absolute L2 displacement error',
                'Boundary L2 displacement error', 'Estimated active-matrix condition (1-norm)']):
            axis.set_xlabel('Nominal element size h')
            axis.set_ylabel(ylabel)
            axis.grid(True, which='both', alpha=.3)
            axis.legend()
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)

    return dict(errors=errors, boundary_errors=boundary_errors,
                condition_estimates=condition_estimates, observations=observations)


if __name__ == '__main__':
    main()
