"""Select active spline coefficients with nonzero trace on a patch boundary."""
import numpy as np
from shapely.geometry import LineString


def select_patch_boundary(space, edge):
    """Return coefficient IDs on umin/umax/vmin/vmax material boundary segments.

    These are parametric patch edges, not physical coordinate extrema or interior
    trim curves. Strong trace constraints require an open/clamped patch basis.
    """
    if edge not in ('umin', 'umax', 'vmin', 'vmax'):
        raise ValueError('edge must be umin, umax, vmin or vmax')
    along_u = edge.startswith('v')
    knots = space.kv if along_u else space.ku
    coordinate = float(knots[0] if edge.endswith('min') else knots[-1])
    selected = set()

    def line_parts(geometry):
        if geometry.is_empty:
            return []
        if geometry.geom_type == 'LineString':
            return [geometry]
        return [part for item in getattr(geometry, 'geoms', []) for part in line_parts(item)]

    for cell in space.elements:
        if getattr(cell, 'void', False) or not getattr(cell, 'active', True):
            continue
        boundary = (cell.v_min if edge == 'vmin' else cell.v_max) if along_u else (
            cell.u_min if edge == 'umin' else cell.u_max)
        if boundary != coordinate:
            continue
        start, end = (cell.u_min, cell.u_max) if along_u else (cell.v_min, cell.v_max)
        segment = LineString([(start, coordinate), (end, coordinate)] if along_u else
                             [(coordinate, start), (coordinate, end)])
        if getattr(cell, 'trimmed', False):
            polygons = getattr(cell, 'material_polygons', None)
            if polygons is None:
                raise ValueError('Trimmed boundary selection requires material polygons')
            segments = [part for polygon in polygons for part in line_parts(polygon.intersection(segment))]
        else:
            segments = [segment]
        # More than p samples detect a nonzero polynomial trace on each cell span.
        gauss = np.polynomial.legendre.leggauss(cell.p+1)[0]
        for part in segments:
            if part.length == 0:
                continue
            for node in gauss:
                point = part.interpolate((node+1)/2, normalized=True)
                for basis in cell.supported_b_splines:
                    if abs(basis(point.x, point.y, cell=cell)) > 1e-12:
                        selected.add(basis.id)
    return selected
