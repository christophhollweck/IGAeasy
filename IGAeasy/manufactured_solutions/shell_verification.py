"""Reusable manufactured shell problems without plotting or study configuration."""
from IGAeasy.trimming.Brep import BRep
import numpy as np
from shapely.geometry import box, LineString
from shapely.geometry.polygon import orient
from IGAeasy.geometry import get_geometry
from IGAeasy.geometry.transformations import rotation_matrix
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.elements.KLshell3p import KLShell
from IGAeasy.elements.RMshell5p_hier import RMShell
from IGAeasy.elements import RMshell6p, RMshell5p, RMshell5p_v2
from IGAeasy.constraints.kl_clamped_weak import WeakKLClampedBoundary, WeakRM5HierClampedBoundary
from IGAeasy.constraints.kl_mixed_weak import WeakKLMixedBoundary
from IGAeasy.constraints.rm5_hier_weak import WeakRM5HierBoundary
from IGAeasy.constraints.shell_weak import WeakRM6Boundary, WeakRM5Boundary
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.postprocessing.postprocessing import update_cps
from IGAeasy.loads import SurfaceLoad
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.solver.direct_solver import solve
from IGAeasy.error_norms.surface_norms import SurfaceL2Quadrature, SurfaceDirectorL2Quadrature
from IGAeasy.manufactured_solutions.kl_parabolic_reference import (
    ParabolicKLReference, ParabolicRM5HierReference, PolynomialClampedPlateReference, monomial_coefficient)
from IGAeasy.manufactured_solutions.rm_parabolic_reference import ParabolicRMShearReference


def solve_gradient_patch(n_current, family='THB', trimmed=False, angle=0., power=4, local_refinement=False,
                domain=None, formulation='KL', curvature=.2, safety_factor=None,
                stabilization='scalar', degree=None, reference_kind='parabolic',
                boundary_kind='clamped', normal_rotation=False, method='nitsche', hier_directions=None,
                kl_displacement_directions=None):
    degree = 3 if degree is None else degree
    if not isinstance(degree, int) or degree < 2:
        raise ValueError('Gradient shell study requires degree at least two')
    if family not in ('THB', 'LR'):
        raise ValueError('family must be THB or LR')
    if formulation not in ('KL', 'RM5_hier'):
        raise ValueError('formulation must be KL or RM5_hier')
    if boundary_kind not in ('clamped', 'mixed', 'mixed_hier'):
        raise ValueError('boundary_kind must be clamped, mixed or mixed_hier')
    if boundary_kind == 'mixed' and (formulation != 'KL' or reference_kind != 'parabolic'):
        raise ValueError('Mixed reference study currently requires parabolic KL')
    if boundary_kind == 'mixed_hier' and (formulation != 'RM5_hier' or reference_kind != 'parabolic'):
        raise ValueError('mixed_hier requires the hierarchical parabolic reference')
    dof = 3 if formulation == 'KL' else 5
    rotation = rotation_matrix(angle)
    reference_class = ParabolicKLReference if dof == 3 else ParabolicRM5HierReference
    if reference_kind == 'clamped_plate':
        reference_class = PolynomialClampedPlateReference
    elif reference_kind != 'parabolic':
        raise ValueError('reference_kind must be parabolic or clamped_plate')
    reference = reference_class(power=power, rotation=rotation, curvature=curvature)
    geo = get_geometry.surface(n_current, n_current, 1, 1, 1, 1, degree-1, degree, distortion=0, n_waves=0)
    RS = (HierarchicalSpace(*geo.knot_vectors, degree, geo.control_points, 2) if family == 'THB'
          else LRSpline(degree, *geo.knot_vectors, geo.control_points))
    for basis in RS.basis:
        knots = basis.knots_u[1:-1]
        square = monomial_coefficient(knots, 2)
        basis.cp = rotation@np.array([basis.cp[0], basis.cp[1], reference.curvature*square])
        if power <= degree:
            basis.scalar_value = reference.coefficient(knots, basis.knots_v[1:-1])
    if local_refinement:
        selected = min(RS.basis, key=lambda b: np.linalg.norm(np.asarray(b.cp)-rotation@np.array([.25, .25, .0125])))
        if family == 'THB':
            RS.refine({0: [selected.loc_id]})
        else:
            RS.refine_by_structured_mesh([selected])
            RS.get_connectivity()
    if domain is not None:
        trimmed = True
    else:
        domain = box(.13, .07, .91, .87) if trimmed else box(0., 0., 1., 1.)
    domain = orient(domain, sign=1.)
    if trimmed:
        RS.BReps = [BRep(1, polygon=domain, outerloop=True)]
        trim(RS)
    material = LinearElasticMaterial(reference.E, reference.nu, 1.)
    assign_gp_and_properties(RS, material_model=material, properties={'t': reference.thickness}, full_quad=True)
    boundary_class = WeakKLClampedBoundary if dof == 3 else WeakRM5HierClampedBoundary
    if boundary_kind == 'mixed':
        boundaries = [WeakKLMixedBoundary(LineString(ring.coords), reference.field,
            normal_rotation=normal_rotation,
            normal_moment=None if normal_rotation else reference.normal_moment,
            method=method, safety_factor=4. if safety_factor is None else safety_factor,
            displacement_directions=kl_displacement_directions, traction=reference.transverse_traction)
            for ring in (domain.exterior, *domain.interiors)]
    elif boundary_kind == 'mixed_hier':
        boundaries = [WeakRM5HierBoundary(LineString(ring.coords), reference.field,
            components=('ux','uy','uz') if hier_directions is None else ('ux','uy','uz','dx','dy','dz'),
            directions=hier_directions, method=method,
            safety_factor=4. if safety_factor is None else safety_factor,
            natural=lambda x,y,z,mu: np.r_[np.zeros(3), reference.moment_vector(x,y,z,mu)])
            for ring in (domain.exterior, *domain.interiors)]
    else:
        boundaries = [boundary_class(LineString(ring.coords), reference.field, method=method,
                      safety_factor=safety_factor, stabilization=stabilization)
                      for ring in (domain.exterior, *domain.interiors)]
    load = SurfaceLoad(dof, [lambda u, v, i=i: reference.body(u, v)[i] for i in range(dof)])
    gamma_max = 0.
    gamma_u_max, gamma_d_max = 0., 0.
    for cell in RS.elements:
        if cell.void:
            continue
        shell = KLShell() if dof == 3 else RMShell()
        shell.element_routine(cell)
        force = load.integrate_contribution(cell, dof)
        # Each stabilization bound uses the original bulk energy.
        contributions = [boundary.integrate_contribution(cell) for boundary in boundaries]
        for boundary, (kb, fb, gamma) in zip(boundaries, contributions):
            cell.K_el += kb
            cell.f_el += fb
            gamma_max = max(gamma_max, gamma)
            if gamma > 0.:
                gamma_u_max = max(gamma_u_max, boundary.last_trace_penalties[0])
                if boundary_kind == 'mixed':
                    if normal_rotation:
                        gamma_d_max = max(gamma_d_max, boundary.last_trace_penalties[-1])
                elif len(boundary.last_trace_penalties) > 3:
                    gamma_d_max = max(gamma_d_max, boundary.last_trace_penalties[3])
        cell.f_el += force
    K = MatrixAssembler(dof_per_cp=dof).assemble(RS, 'K_el', use_sparse=True)
    f = VectorAssembler(dof_per_cp=dof).assemble(RS, 'f_el')
    if boundary_kind == 'mixed' and kl_displacement_directions is not None:
        from IGAeasy.constraints.linear_constraints import LinearConstraints, solve_linear_constraints
        P = boundaries[0].displacement_projector
        if P is None:
            raise ValueError('This parabolic reference gauge requires constant displacement directions')
        _, _, vectors = np.linalg.svd(P, full_matrices=True)
        gauge = LinearConstraints(len(f))
        # The manufactured field has zero displacement along the free straight
        # generator. Fix its mean coefficient only to remove rigid translation.
        active_basis = [b for b in RS.basis if getattr(b, 'active', True)]
        for direction in vectors[len(P):]:
            gauge.add({3*b.id+j: direction[j]/len(active_basis)
                       for b in active_basis for j in range(3)})
        solution = solve_linear_constraints(K, f, gauge)[0] if len(P) < 3 else solve(K, f, scaling='auto')
    else:
        solution = solve(K, f, scaling='auto')
    norm = SurfaceL2Quadrature(RS, order=degree+3)
    error = norm.error(solution.reshape(-1, dof)[:, :3], lambda x, y, z: reference.field(x, y, z)[:3])
    director_norm = SurfaceDirectorL2Quadrature(RS, formulation, order=degree+3)
    director_error = director_norm.error(solution.reshape(-1, dof), lambda x, y, z: reference.field(x, y, z)[3:])
    update_cps(RS, solution, dof_per_cp=dof, formulation=formulation)
    RS.boundary_conditions = boundaries
    residual = np.linalg.norm(K@solution-f)/max(np.linalg.norm(f), 1e-300)
    row = {'h': 1/n_current, 'l2': error, 'director_l2': director_error, 'gamma_max': gamma_max,
           'gamma_u_max': gamma_u_max, 'gamma_d_max': gamma_d_max, 'residual': residual}
    if dof == 5:
        additional_difference = solution.reshape(-1, dof).copy()
        additional_difference[:, :3] = 0.
        additional_difference[:, 3] -= reference.eta
        row['hierarchical_l2'] = director_norm.error(additional_difference)
    if power <= degree:
        exact = np.zeros_like(solution)
        for basis in RS.basis:
            if getattr(basis, 'active', True):
                exact[dof*basis.id:dof*basis.id+3] = rotation@np.array([0., 0., basis.scalar_value])
                if dof == 5:
                    exact[dof*basis.id+3] = reference.eta
        row['patch_residual'] = np.linalg.norm(K@exact-f)/max(np.linalg.norm(f), 1e-300)
    return RS, solution, row


def solve_rm_patch(n_current, formulation='RM6', method='nitsche', family='THB',
                degree=3, power=4, curvature=.2, angle=.6, trimmed=True,
                local_refinement=False, domain=None, director_amplitude=.002, penalty_multiplier=100.):
    if family not in ('THB', 'LR') or degree < 2:
        raise ValueError('Use THB or LR with degree at least two')
    if method not in ('penalty', 'nitsche', 'penalty_free'):
        raise ValueError('method must be penalty, nitsche or penalty_free')
    rotation = rotation_matrix(angle)
    rule = np.polynomial.legendre.leggauss(3)
    reference = ParabolicRMShearReference(formulation, power=power, curvature=curvature,
                                          rotation=rotation, gauss_t=rule,
                                          director_amplitude=director_amplitude)
    geo = get_geometry.surface(n_current, n_current, 1, 1, 1, 1,
                              degree-1, degree, distortion=0, n_waves=0)
    RS = (HierarchicalSpace(*geo.knot_vectors, degree, geo.control_points, 2)
          if family == 'THB' else LRSpline(degree, *geo.knot_vectors, geo.control_points))
    for b in RS.basis:
        square = monomial_coefficient(b.knots_u[1:-1], 2)
        b.cp = rotation@np.array([b.cp[0], b.cp[1], curvature*square])
        if power <= degree:
            b.scalar_value = reference.amplitude*monomial_coefficient(b.knots_u[1:-1], power)
    if local_refinement:
        selected = min(RS.basis, key=lambda b: np.linalg.norm(np.asarray(b.cp)-rotation@np.array([.25, .25, curvature*.25**2])))
        if family == 'THB':
            RS.refine({0: [selected.loc_id]})
        else:
            RS.refine_by_structured_mesh([selected])
            RS.get_connectivity()
    if domain is None:
        domain = box(.13, .07, .91, .87) if trimmed else box(0., 0., 1., 1.)
    else:
        trimmed = True
    domain = orient(domain, sign=1.)
    if trimmed:
        RS.BReps = [BRep(1, polygon=domain, outerloop=True)]
        trim(RS)
    assign_gp_and_properties(RS, material_model=LinearElasticMaterial(reference.E, reference.nu, 1., kappa=reference.kappa),
                             properties={'t': reference.thickness, 'rho': 1.}, full_quad=True)
    shell = {'RM6': RMshell6p, 'RM5': RMshell5p, 'RM5_v2': RMshell5p_v2}[formulation].RMShell()
    dof = shell.dof_per_cp
    boundaries = [(WeakRM6Boundary(LineString(ring.coords), reference.field, rule, method=method)
                   if dof == 6 else WeakRM5Boundary(LineString(ring.coords), reference.field,
                                                  formulation=formulation, method=method))
                  for ring in (domain.exterior, *domain.interiors)]
    load = SurfaceLoad(dof, reference.body)
    multiplier = penalty_multiplier if method == 'penalty' else 1.
    gamma_max = 0.
    for cell in RS.elements:
        if cell.void or not getattr(cell, 'active', True):
            continue
        if dof == 6:
            shell.element_routine(cell, rule)
        else:
            shell.element_routine(cell)
        force = load.integrate_contribution(cell, dof)
        contributions = [boundary.integrate_contribution(cell) for boundary in boundaries]
        for Kbc, fbc, gamma in contributions:
            cell.K_el += multiplier*Kbc
            cell.f_el += multiplier*fbc
            gamma_max = max(gamma_max, multiplier*gamma)
        cell.f_el += force
    K = MatrixAssembler(dof_per_cp=dof).assemble(RS, 'K_el', use_sparse=True)
    f = VectorAssembler(dof_per_cp=dof).assemble(RS, 'f_el')
    coefficients = solve(K, f, scaling=True).reshape(-1, dof)
    norm = SurfaceL2Quadrature(RS, order=degree+3)
    director_norm = SurfaceDirectorL2Quadrature(RS, formulation, order=degree+3)
    row = dict(h=1/n_current, l2=norm.error(coefficients[:, :3], reference.displacement),
               director_l2=director_norm.error(coefficients, reference.director), gamma_max=gamma_max,
               residual=np.linalg.norm(K@coefficients.ravel()-f)/np.linalg.norm(f))
    if dof == 6:
        row['rotation_l2'] = norm.error(coefficients[:, 3:], lambda x, y, z: reference.field(x, y, z)[3:])
    elif power <= degree:
        exact = np.zeros_like(coefficients)
        for b in RS.basis:
            exact[b.id, :3] = rotation@np.array([0., b.scalar_value, 0.])
            exact[b.id, 4] = b.scalar_value*reference.director_amplitude/reference.amplitude
        row['patch_residual'] = np.linalg.norm(K@exact.ravel()-f)/np.linalg.norm(f)
    update_cps(RS, coefficients.ravel(), dof_per_cp=dof, formulation=formulation)
    RS.boundary_conditions = boundaries
    return RS, coefficients, row
