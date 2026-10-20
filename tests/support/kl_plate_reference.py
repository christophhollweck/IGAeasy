"""Simply supported KL plate: weak deflection and a free normal rotation."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from shapely.geometry import box, LineString, Point
from shapely.geometry.polygon import orient
from IGAeasy.trimming.Brep import BRep
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.elements.KLshell3p import KLShell
from IGAeasy.constraints.kl_plate_weak import WeakKLPlateBoundary
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.loads import SurfaceLoad
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.solver.direct_solver import solve
from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature, SurfaceDirectorL2Quadrature
from IGAeasy.manufactured_solutions.kl_plate_reference import KLPlateReference

n = 2
p = 3
n_refinements = 3
methods = ['nitsche', 'penalty']
family = 'THB'
reference_kind = 'uniform'
normal_rotation = False
angle = .6
penalty_multiplier = 100.
plot_geometry = True


def solve_level(n_current, method='nitsche', family='THB', degree=3,
                reference_kind='uniform', normal_rotation=False, angle=.6,
                domain=None, local_refinement=False):
    if family not in ('THB', 'LR') or degree < 2:
        raise ValueError('Use THB or LR and degree at least two')
    axis = np.array([1., 2., 3.])/np.sqrt(14.)
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    rotation = np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*cross@cross
    reference = KLPlateReference(reference_kind, rotation=rotation)
    geo = get_geometry.surface(n_current, n_current, 1, 1, 1, 1,
                              degree-1, degree, distortion=0, n_waves=0)
    RS = (HierarchicalSpace(*geo.knot_vectors, degree, geo.control_points, 2)
          if family == 'THB' else LRSpline(degree, *geo.knot_vectors, geo.control_points))
    for b in RS.basis:
        b.cp = rotation@np.asarray(b.cp)
    if local_refinement:
        selected = min(RS.basis, key=lambda b: np.linalg.norm(np.asarray(b.cp)-rotation@np.array([.25, .25, 0.])))
        if family == 'THB':
            RS.refine({0: [selected.loc_id]})
        else:
            RS.refine_by_structured_mesh([selected])
            RS.get_connectivity()
    trimmed = domain is not None
    domain = orient(box(0., 0., 1., 1.) if domain is None else domain, sign=1.)
    if trimmed:
        RS.BReps = [BRep(1, polygon=domain, outerloop=True)]
        trim(RS)
    material = LinearElasticMaterial(reference.E, reference.nu, 1.)
    assign_gp_and_properties(RS, material_model=material,
                             properties={'t': reference.thickness, 'rho': 1.}, full_quad=True)
    multiplier = penalty_multiplier if method == 'penalty' else 1.
    boundaries = [WeakKLPlateBoundary(LineString(ring.coords), reference.field,
                   normal_rotation=normal_rotation, method=method,
                   safety_factor=4.*multiplier,
                   normal_moment=None if normal_rotation else reference.normal_moment)
                  for ring in (domain.exterior, *domain.interiors)]
    load = SurfaceLoad(3, reference.body)
    gamma_max = 0.
    for cell in RS.elements:
        if cell.void or not getattr(cell, 'active', True):
            continue
        KLShell().element_routine(cell)
        body = load.integrate_contribution(cell, 3)
        contributions = [boundary.integrate_contribution(cell) for boundary in boundaries]
        for Kbc, fbc, gamma in contributions:
            cell.K_el += Kbc
            cell.f_el += fbc
            gamma_max = max(gamma_max, gamma)
        cell.f_el += body
    K = MatrixAssembler(dof_per_cp=3).assemble(RS, 'K_el', use_sparse=True)
    f = VectorAssembler(dof_per_cp=3).assemble(RS, 'f_el')
    normal = rotation[:, 2]
    projection = sparse.kron(sparse.eye(len(RS.basis)), normal[:, None], format='csr')
    reduced, rhs = projection.T@K@projection, projection.T@f
    w = solve(reduced, rhs, scaling=True)
    coefficients = w[:, None]*normal
    norm = SurfaceL2Quadrature(RS, order=degree+3)
    director_norm = SurfaceDirectorL2Quadrature(RS, 'KL', order=degree+3)
    row = dict(h=1/n_current, l2=norm.error(coefficients, reference.displacement),
               director_l2=director_norm.error(coefficients, lambda x, y, z: reference.field(x, y, z)[3:]),
               gamma_max=gamma_max,
               residual=np.linalg.norm(reduced@w-rhs)/max(np.linalg.norm(rhs), 1e-300))
    center = sum(b(.5, .5)*w[b.id] for b in RS.basis) if domain.covers(Point(.5, .5)) else np.nan
    row.update(center=center, center_error=abs(center-reference.values(.5, .5)[0]))
    for b in RS.basis:
        b.displacements = coefficients[b.id]
    return RS, coefficients, row


if __name__ == '__main__':
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    geometry_shown = False
    comparison = None
    for method in methods:
        rows = []
        for level in range(n_refinements+1):
            current = n*2**level
            print(f'{method}: {current} x {current}...', flush=True)
            RS, _, row = solve_level(current, method, family, p, reference_kind,
                                     normal_rotation=normal_rotation, angle=angle)
            rows.append(row)
            print(f"center={row['center']:.10e}, L2={row['l2']:.3e}, director L2={row['director_l2']:.3e}", flush=True)
            if plot_geometry and not geometry_shown:
                from IGAeasy.postprocessing.visualization import build_surface_data, plot_surface_results
                mesh, edges = build_surface_data(RS, deformation_scale=0.,
                    fields=('displacement_magnitude',), samples_per_element=8)
                plot_surface_results(mesh, edges, deformation_scale=0.)
                geometry_shown = True
        if comparison is None:
            comparison = rows
        for ax, key in zip(axes, ('l2', 'director_l2')):
            ax.loglog([row['h'] for row in rows], [row[key] for row in rows], 'o-', label=method)
    h = np.array([row['h'] for row in comparison])
    for ax, key, order in zip(axes, ('l2', 'director_l2'), (min(p+1, 2*p-2), p)):
        ax.loglog(h, comparison[-1][key]*(h/h[-1])**order, '--', label=f'h^{order} comparison')
    for ax, label in zip(axes, ('Displacement L2 error', 'Director L2 error')):
        ax.set_xlabel('Element size h')
        ax.set_ylabel(label)
        ax.grid(True, which='both')
        ax.legend()
    fig.tight_layout()
    plt.show()
