"""Weak physical director constraints on material portions of patch edges."""
import numpy as np
from scipy.sparse import coo_matrix
from shapely.geometry import LineString
from shapely.ops import unary_union
from IGAeasy.constraints.linear_constraints import LinearConstraints
from IGAeasy.constraints.shell_constraints import constrain_director_at_point
from IGAeasy.differential_geometry.derivatives import get_full_geometric_data


def assemble_director_penalty(space, components, edge, formulation, penalty,
                              direction='conormal', value=0., quadrature_order=None,
                              _nitsche_KL=False):
    """Return K_bc,f_bc for penalty/2 * integral (direction.dot(delta_d)-value)^2 ds.

    This is penalty enforcement, not consistent Nitsche enforcement. Directions
    are global three-vectors, or local 'tangent'/'conormal'. value is a constant
    or callable value(xi,eta). Only parametric patch edges are supported.
    """
    if edge not in ('umin', 'umax', 'vmin', 'vmax'):
        raise ValueError('Unknown parametric patch edge')
    if not np.isscalar(penalty) or not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('penalty must be finite and positive')
    if quadrature_order is not None and (isinstance(quadrature_order, bool) or
            not isinstance(quadrature_order, (int, np.integer)) or quadrature_order < 1):
        raise ValueError('quadrature_order must be a positive integer')
    layout = LinearConstraints.for_control_points(components, len(space.basis))
    size = layout.n_dofs
    rows, columns, entries = [], [], []
    force = np.zeros(size)
    along_u = edge.startswith('v')
    knots = space.kv if along_u else space.ku
    coordinate = float(knots[0] if edge.endswith('min') else knots[-1])

    def parts(geometry):
        if geometry.is_empty:
            return []
        if geometry.geom_type == 'LineString':
            return [geometry]
        return [p for g in getattr(geometry, 'geoms', []) for p in parts(g)]

    for cell in space.elements:
        if getattr(cell, 'void', False) or not getattr(cell, 'active', True):
            continue
        boundary = (cell.v_min if edge == 'vmin' else cell.v_max) if along_u else (
            cell.u_min if edge == 'umin' else cell.u_max)
        if boundary != coordinate:
            continue
        start, end = (cell.u_min, cell.u_max) if along_u else (cell.v_min, cell.v_max)
        line = LineString([(start, coordinate), (end, coordinate)] if along_u else
                          [(coordinate, start), (coordinate, end)])
        if getattr(cell, 'trimmed', False):
            polygons = getattr(cell, 'material_polygons', None)
            if polygons is None:
                raise ValueError('Trimmed edge integration requires material polygons')
            segments = parts(line.intersection(unary_union(polygons)))
        else:
            segments = [line]
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order or cell.p+2)
        for segment in segments:
            for node, weight in zip(nodes, weights):
                point = segment.interpolate((node+1)/2, normalized=True)
                u, v = point.x, point.y
                cov, contra, _, _, _, _ = get_full_geometric_data(cell, u, v)
                tangent = cov[1] if along_u else cov[2]
                if isinstance(direction, str):
                    if direction == 'tangent':
                        vector = tangent/np.linalg.norm(tangent)
                    elif direction == 'conormal':
                        vector = contra[2] if along_u else contra[1]
                        vector = vector/np.linalg.norm(vector)*(-1 if edge.endswith('min') else 1)
                    else:
                        raise ValueError('direction must be tangent, conormal or a global vector')
                else:
                    vector = direction
                constraint = LinearConstraints.for_control_points(components, len(space.basis))
                prescribed = value(u, v) if callable(value) else value
                constrain_director_at_point(space, constraint, u, v, vector, formulation, prescribed)
                C, d = constraint.matrices()
                ids, coefficients = C.indices, C.data
                measure = weight*segment.length/2*np.linalg.norm(tangent)
                factor = penalty*measure
                block = factor*np.outer(coefficients, coefficients)
                rhs = factor*coefficients*d[0]
                if _nitsche_KL:
                    # M_nn is conjugate to delta_d.n = -dw/dn in our convention.
                    properties = cell.gausspoints[0]
                    metric = np.array([[cov[1]@cov[1], cov[1]@cov[2]],
                                       [cov[2]@cov[1], cov[2]@cov[2]]])
                    _, Db, _ = properties.material_model.compute_C_voigt_curv_2D(
                        np.linalg.inv(metric), properties.t)
                    normal_components = np.array([vector@cov[1], vector@cov[2]])
                    moment_row = np.array([normal_components[0]**2, normal_components[1]**2,
                                           2*normal_components[0]*normal_components[1]])@Db
                    traction = np.zeros(size)
                    for basis in cell.supported_b_splines:
                        d11, d22 = basis.grad2(u, v, cell=cell)
                        d12 = basis.grad_mix(u, v, cell=cell)
                        coefficient = moment_row@np.array([-d11, -d22, -2*d12])
                        for index, name in enumerate(('x', 'y', 'z')):
                            traction[layout.dof(basis.id, name)] = coefficient*cov[3][index]
                    combined = np.union1d(ids, np.flatnonzero(traction))
                    c = np.zeros(len(combined))
                    c[np.searchsorted(combined, ids)] = coefficients
                    moment = traction[combined]
                    ids = combined
                    block = measure*(penalty*np.outer(c, c)-np.outer(c, moment)-np.outer(moment, c))
                    rhs = measure*(penalty*c-moment)*d[0]
                rows.extend(np.repeat(ids, len(ids)))
                columns.extend(np.tile(ids, len(ids)))
                entries.extend(block.ravel())
                force[ids] += rhs
    return coo_matrix((entries, (rows, columns)), shape=(size, size)).tocsr(), force


def assemble_KL_slope_nitsche(space, edge, stabilization, value=0., quadrature_order=None):
    """Symmetric Nitsche for delta_d.conormal on affine untrimmed KL patches.

    Displacement traces must be prescribed strongly. stabilization is the
    dimensional coefficient (typically beta*D/h), not a dimensionless beta.
    Adequate stabilization is required for coercivity; no automatic bound is used.
    Material and thickness must be constant within each element.
    """
    reference = None
    for cell in space.elements:
        if not getattr(cell, 'active', True):
            continue
        if getattr(cell, 'trimmed', False) or getattr(cell, 'void', False):
            raise ValueError('KL slope Nitsche currently requires an untrimmed affine patch')
        if not cell.gausspoints or not hasattr(cell.gausspoints[0], 'material_model'):
            raise ValueError('Assign material and thickness before boundary assembly')
        nodes = np.polynomial.legendre.leggauss(cell.p+1)[0]
        for x in nodes:
            for y in nodes:
                u = cell.u_min+(x+1)/2*(cell.u_max-cell.u_min)
                v = cell.v_min+(y+1)/2*(cell.v_max-cell.v_min)
                cov, _, _, _, _, _ = get_full_geometric_data(cell, u, v)
                tangents = np.array([cov[1], cov[2]])
                if reference is None:
                    reference = tangents
                elif not np.allclose(tangents, reference, rtol=1e-11, atol=1e-12):
                    raise ValueError('KL slope Nitsche currently requires affine geometry')
    return assemble_director_penalty(space, ('x', 'y', 'z'), edge, 'KL', stabilization,
                                     value=value, quadrature_order=quadrature_order,
                                     _nitsche_KL=True)
