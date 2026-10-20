"""Separate KL hemisphere discretization, quadrature and reflection effects."""
import numpy as np
from scipy.sparse import coo_matrix
from numpy.polynomial.legendre import leggauss

from IGAeasy.geometry.get_geometry import open_hemisphere
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.numerical_integration.gauss_inplane import gauss_point
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.elements.KLshell3p import KLShell
from IGAeasy.loads import PointLoad
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.solver.direct_solver import solve
from IGAeasy.splines.batch_evaluation import evaluate_derivatives_batch


def sphere_derivatives(u, v):
    """Exact radius-10 sphere in the benchmark's azimuth/polar coordinates."""
    a, b = np.pi/2, np.radians(18.)-np.pi/2
    theta, phi = a*u, np.pi/2+b*v
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    tangents = 10*np.array([[-a*st*sp, a*ct*sp, 0.],
                            [b*ct*cp, b*st*cp, -b*sp]])
    second = 10*np.array([[[-a*a*ct*sp, -a*a*st*sp, 0.],
                            [-a*b*st*cp, a*b*ct*cp, 0.]],
                           [[-a*b*st*cp, a*b*ct*cp, 0.],
                            [-b*b*ct*sp, -b*b*st*sp, -b*b*cp]]])
    return tangents, second


def cartesian_stiffness(cell, exact_sphere=True):
    """Independent Cartesian plane-stress assembly on the sphere or spline fit.

    Displacement bases stay polynomial. Only geometry derivatives are exact;
    no rational-basis support or production element is changed for this check.
    Curvature variation is evaluated through the normal variation directly.
    """
    basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
    cps = np.asarray([b.cp for b in basis])
    data = evaluate_derivatives_batch(basis, [(g.xi, g.eta) for g in cell.gausspoints],
                                      cell, max_order=2)
    size = 3*len(basis)
    matrix = np.zeros((size, size))
    for q, gp in enumerate(cell.gausspoints):
        if exact_sphere:
            tangents, second = sphere_derivatives(gp.xi, gp.eta)
        else:
            tangents = np.array([data[1, 0][q]@cps, data[0, 1][q]@cps])
            mixed = data[1, 1][q]@cps
            second = np.array([[data[2, 0][q]@cps, mixed],
                               [mixed, data[0, 2][q]@cps]])
        cross = np.cross(*tangents)
        jacobian = np.linalg.norm(cross)
        normal = cross/jacobian
        contra = np.linalg.solve(tangents@tangents.T, tangents)
        e1 = tangents[0]/np.linalg.norm(tangents[0])
        frame = np.column_stack((e1, np.cross(normal, e1)))
        Q = contra@frame
        membrane, bending = np.zeros((2, 2, size)), np.zeros((2, 2, size))
        for i in range(len(basis)):
            block = slice(3*i, 3*i+3)
            du, dv = data[1, 0][q, i], data[0, 1][q, i]
            normal_change = -np.outer(du*contra[0]+dv*contra[1], normal)
            hessian = [[data[2, 0][q, i], data[1, 1][q, i]],
                       [data[1, 1][q, i], data[0, 2][q, i]]]
            for a, da in enumerate((du, dv)):
                for b, db in enumerate((du, dv)):
                    membrane[a, b, block] = (da*tangents[b]+db*tangents[a])/2
                    bending[a, b, block] = -normal*hessian[a][b]-second[a, b]@normal_change
        operators = []
        for strain in (membrane, bending):
            cartesian = np.einsum('ai,abk,bj->ijk', Q, strain, Q)
            operators.append(np.vstack((cartesian[0, 0], cartesian[1, 1], 2*cartesian[0, 1])))
        E, nu, t = gp.material_model.E, gp.material_model.nu, gp.t
        C = E/(1-nu*nu)*np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., (1-nu)/2]])
        Bm, Bb = operators
        matrix += (t*Bm.T@C@Bm+t**3/12*Bb.T@C@Bb)*gp.weight_inplane*jacobian
    return matrix


def reflection_space(n_u, n_v):
    """Cartesian even/odd reflection by coefficient elimination on this tensor grid.

    At y=0: uy=0, ux/uz are even. At x=0: ux=0, uy/uz are even.
    Tying two adjacent rows sets the even component's transverse derivative to
    zero. This is a separate strong comparison, not a generic KL boundary API.
    """
    labels = np.arange(3*n_u*n_v).reshape(n_v, n_u, 3)
    labels[:, 1, 0] = labels[:, 0, 0]
    labels[:, 1, 2] = labels[:, 0, 2]
    labels[:, -2, 1] = labels[:, -1, 1]
    labels[:, -2, 2] = labels[:, -1, 2]
    labels[:, 0, 1] = -1
    labels[:, -1, 0] = -1
    labels[labels == labels[0, 0, 2]] = -1  # One vertical translation gauge.
    labels = labels.ravel()
    rows = np.flatnonzero(labels >= 0)
    _, columns = np.unique(labels[rows], return_inverse=True)
    return coo_matrix((np.ones(len(rows)), (rows, columns)),
                      shape=(len(labels), max(columns)+1)).tocsr()


def run(n_elements=16, degree=3, quadrature_order=None, geometry_elements=8,
        exact_sphere=False):
    # --- Refine one fixed cubic fit; degree elevation preserves its geometry ---
    ratio = n_elements/geometry_elements
    if ratio < 1 or not np.isclose(np.log2(ratio), round(np.log2(ratio))):
        raise ValueError('Use a dyadic refinement of geometry_elements')
    if degree < 3:
        raise ValueError('The fixed geometry requires degree >= 3')
    geometry = open_hemisphere(geometry_elements)
    for _ in range(degree-3):
        geometry.elevate_degrees([0, 1])
    for _ in range(round(np.log2(ratio))):
        for direction in (0, 1):
            knots = np.unique(geometry.knot_vectors[direction])
            geometry.insert_knots(direction, (knots[:-1]+knots[1:])/2)
    space = HierarchicalSpace(*geometry.knot_vectors, degree, geometry.control_points)
    n_u, n_v = [len(k)-degree-1 for k in geometry.knot_vectors]
    np.testing.assert_allclose([b.cp for b in sorted(space.basis, key=lambda b: b.id)],
                               geometry.control_points)

    # --- Independent tensor Gauss rule, retaining the production KL element ---
    order = degree+1 if quadrature_order is None else quadrature_order
    nodes, weights = leggauss(order)
    material = LinearElasticMaterial(6.825e7, .3, 1.)
    shell = KLShell()
    for cell in space.elements:
        u = cell.u_min+(nodes+1)*(cell.u_max-cell.u_min)/2
        v = cell.v_min+(nodes+1)*(cell.v_max-cell.v_min)/2
        area_scale = (cell.u_max-cell.u_min)*(cell.v_max-cell.v_min)/4
        cell.gausspoints = []
        for i, x in enumerate(u):
            for j, y in enumerate(v):
                gp = gauss_point(x, y, weights[i]*weights[j]*area_scale)
                gp.material_model, gp.t, gp.rho = material, .04, 1.
                cell.gausspoints.append(gp)
        if exact_sphere:
            cell.K_el = cartesian_stiffness(cell)
            cell.bfnums = np.array(sorted(b.id for b in cell.supported_b_splines))
            cell.f_el = np.zeros(3*len(cell.bfnums))
        else:
            shell.element_routine(cell)

    # --- Unit quarter loads at the two equatorial endpoints ---
    for point, force in (((0., 0.), [1., 0., 0.]), ((1., 0.), [0., -1., 0.])):
        cell = next(c for c in space.elements if c.contains(*point))
        cell.f_el += PointLoad(point, force).compute_contribution(cell, 3)
    K = MatrixAssembler(3).assemble(space, 'K_el', use_sparse=True)
    f = VectorAssembler(3).assemble(space, 'f_el')

    # --- Eliminate Cartesian reflection constraints without Nitsche terms ---
    transform = reflection_space(n_u, n_v)
    reduced, rhs = transform.T@K@transform, transform.T@f
    free_solution = solve(reduced, rhs, scaling=True)
    coefficients = np.asarray(transform@free_solution).reshape(-1, 3)
    return dict(n=n_elements, degree=degree, quadrature_order=order,
                geometry_elements=geometry_elements, exact_sphere=exact_sphere,
                radial_a=coefficients[0, 0],
                radial_b=coefficients[n_u-1, 1],
                residual=np.linalg.norm(reduced@free_solution-rhs)/np.linalg.norm(rhs))


if __name__ == '__main__':
    for settings in (dict(n_elements=16), dict(n_elements=16, quadrature_order=6),
                     dict(n_elements=16, degree=4), dict(n_elements=32),
                     dict(n_elements=32, exact_sphere=True)):
        print('Computing', settings, flush=True)
        print(run(**settings), flush=True)
