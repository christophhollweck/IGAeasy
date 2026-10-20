"""Linear constraints on physical infinitesimal shell director changes."""
import numpy as np
from IGAeasy.differential_geometry.derivatives import get_full_geometric_data
from IGAeasy.trimming.trim_elements import contains_material_point


def constrain_director_at_point(space, constraints, xi, eta, direction,
                                formulation, value=0.):
    """Enforce direction dot delta_director = value at one material point.

    This is a point constraint, not an exact boundary trace constraint. Whole-edge
    clamping needs an appropriate trace discretization or a weak formulation.
    Direction is global Cartesian and is used without normalization.
    """
    layouts = {
        'KL': ('x', 'y', 'z'),
        'RM6': ('x', 'y', 'z', 'rx', 'ry', 'rz'),
        'RM5': ('x', 'y', 'z', 'r_a1', 'r_a2'),
        'RM5_v2': ('x', 'y', 'z', 'r_a1', 'r_a2'),
        'RM5_hier': ('x', 'y', 'z', 'r_a1', 'r_a2'),
    }
    if formulation not in layouts:
        raise ValueError(f'Unknown shell formulation {formulation!r}')
    if getattr(constraints, 'components', None) != layouts[formulation] or constraints.n_control_points != len(space.basis):
        raise ValueError('Constraint layout must match the shell formulation and space')
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or not np.any(direction):
        raise ValueError('Direction must be a finite nonzero three-vector')
    cell = next((cell for cell in space.elements if contains_material_point(cell, xi, eta)), None)
    if cell is None:
        raise ValueError('Director constraint point is outside the material domain')
    cov, contra, _, _, _, _ = get_full_geometric_data(cell, xi, eta)
    normal = cov[3]
    if np.linalg.norm(np.cross(direction, normal)) <= 1e-12*np.linalg.norm(direction):
        raise ValueError('Director direction has no tangential component')
    row = {}
    for basis in cell.supported_b_splines:
        N = basis(xi, eta, cell=cell)
        if formulation in ('KL', 'RM5_hier'):
            grad = basis.grad(xi, eta, cell=cell)
            coefficient = -direction@(grad[0]*contra[1]+grad[1]*contra[2])
            for index, component in enumerate(('x', 'y', 'z')):
                row[constraints.dof(basis.id, component)] = coefficient*normal[index]
        if formulation == 'RM6':
            for index, component in enumerate(('rx', 'ry', 'rz')):
                row[constraints.dof(basis.id, component)] = N*(direction@np.cross(np.eye(3)[index], normal))
        elif formulation != 'KL':
            vectors = contra if formulation == 'RM5' else cov
            for index, component in enumerate(('r_a1', 'r_a2'), start=1):
                row[constraints.dof(basis.id, component)] = N*(direction@vectors[index])
    return constraints.add(row, value)


def constrain_affine_director_boundary(space, constraints, edge, direction, formulation, value=0.):
    """Whole-edge director trace condition on an untrimmed affine patch.

    p+1 samples per boundary span determine its degree-p polynomial trace.
    Returns a new collection containing independent equations, including the
    supplied constraints. Dense rank reduction is intended for small setups.
    """
    from scipy.linalg import qr
    from IGAeasy.constraints.linear_constraints import LinearConstraints
    if edge not in ('umin', 'umax', 'vmin', 'vmax'):
        raise ValueError('Unknown parametric patch edge')
    if any(getattr(cell, 'trimmed', False) or getattr(cell, 'void', False)
           for cell in space.elements if getattr(cell, 'active', True)):
        raise ValueError('Affine whole-edge constraints currently require an untrimmed patch')
    reference = None
    boundary_cells = []
    along_u = edge.startswith('v')
    knots = space.kv if along_u else space.ku
    coordinate = float(knots[0] if edge.endswith('min') else knots[-1])
    # Verify affine geometry over every active polynomial cell, not just at one point.
    for cell in space.elements:
        if not getattr(cell, 'active', True):
            continue
        nodes = np.polynomial.legendre.leggauss(cell.p+1)[0]
        for node_u in nodes:
            for node_v in nodes:
                u = cell.u_min+(node_u+1)/2*(cell.u_max-cell.u_min)
                v = cell.v_min+(node_v+1)/2*(cell.v_max-cell.v_min)
                cov, _, _, _, _, _ = get_full_geometric_data(cell, u, v)
                tangents = np.array([cov[1], cov[2]])
                if reference is None:
                    reference = tangents
                elif not np.allclose(tangents, reference, rtol=1e-11, atol=1e-12):
                    raise ValueError('Whole-edge polynomial constraints require affine geometry')
        boundary = (cell.v_min if edge == 'vmin' else cell.v_max) if along_u else (
            cell.u_min if edge == 'umin' else cell.u_max)
        if boundary == coordinate:
            boundary_cells.append(cell)
    if not boundary_cells:
        raise ValueError('No active elements on the selected patch edge')
    temporary = LinearConstraints.for_control_points(constraints.components, len(space.basis))
    C, d = constraints.matrices()
    for row, rhs in zip(C, d):
        temporary.add(dict(zip(row.indices, row.data)), rhs)
    for cell in boundary_cells:
        start, end = (cell.u_min, cell.u_max) if along_u else (cell.v_min, cell.v_max)
        for node in np.polynomial.legendre.leggauss(cell.p+1)[0]:
            varying = start+(node+1)/2*(end-start)
            u, v = (varying, coordinate) if along_u else (coordinate, varying)
            constrain_director_at_point(space, temporary, u, v, direction, formulation, value)
    matrix, rhs = temporary.matrices()
    dense = matrix.toarray()
    norms = np.linalg.norm(dense, axis=1)
    dense = dense/norms[:, None]
    rhs = rhs/norms
    _, R, pivots = qr(dense.T, mode='economic', pivoting=True)
    diagonal = abs(np.diag(R))
    rank = int(np.count_nonzero(diagonal > max(dense.shape)*np.finfo(float).eps*diagonal.max()))
    chosen = pivots[:rank]
    # Every discarded equation must be consistent with the independent ones.
    candidate = np.linalg.lstsq(dense[chosen], rhs[chosen], rcond=None)[0]
    if not np.allclose(dense@candidate, rhs, rtol=1e-10, atol=1e-12):
        raise ValueError('Inconsistent boundary constraints')
    result = LinearConstraints.for_control_points(constraints.components, len(space.basis))
    for row, rhs_value in zip(dense[chosen], rhs[chosen]):
        result.add({int(i): row[i] for i in np.flatnonzero(row)}, rhs_value)
    return result
