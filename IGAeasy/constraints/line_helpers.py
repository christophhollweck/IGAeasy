import numpy as np
from shapely.geometry import LineString, box, Point

def get_boundary_quadrature(line, cell, quadrature_order=None):
    """Return parametric points, weights and unit tangents along a polyline."""
    quadrature = get_integration_points(line, cell, quadrature_order)
    if not quadrature:
        return []
    segments = np.diff(np.asarray(line.coords), axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    segments, lengths = segments[lengths > 0], lengths[lengths > 0]
    ends = np.cumsum(lengths)
    result = []
    for coordinate, weight in quadrature:
        index = min(np.searchsorted(ends, coordinate), len(lengths)-1)
        result.append((line.interpolate(coordinate), weight, segments[index]/lengths[index]))
    return result

def get_iwip(cell, start, end, quadrature_order=None):
    """Map Gauss points and weights onto an arc-length interval."""
    order = cell.p+1 if quadrature_order is None else quadrature_order
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError('quadrature_order must be a positive integer')
    gauss_points, gauss_weights = np.polynomial.legendre.leggauss(order)
    mapped_xi = 0.5 * (end - start) * (gauss_points + 1) + start
    mapped_weights = 0.5 * (end - start) * gauss_weights

    return list(zip(mapped_xi, mapped_weights))


def get_integration_points(line, cell, quadrature_order=None):
    """Integrate material-covered line segments using cumulative arc length s.

    Split at every polyline vertex before clipping, so bends and closed lines
    do not rely on ambiguous projection onto the complete line.
    """
    from shapely.ops import unary_union
    if cell.void:
        return []
    coords = np.asarray(line.coords, dtype=float)
    starts, ends = coords[:-1], coords[1:]
    minimum = np.minimum(starts, ends)
    maximum = np.maximum(starts, ends)
    candidates = ((maximum[:, 0] >= cell.u_min) & (minimum[:, 0] <= cell.u_max) &
                  (maximum[:, 1] >= cell.v_min) & (minimum[:, 1] <= cell.v_max))
    # Interior cells need no polygon operations or neighbor ownership search.
    if not candidates.any():
        return []
    rectangle = box(cell.u_min, cell.v_min, cell.u_max, cell.v_max)
    tolerance = 64*np.finfo(float).eps*max(abs(cell.u_min), abs(cell.u_max),
        abs(cell.v_min), abs(cell.v_max), cell.u_max-cell.u_min, cell.v_max-cell.v_min)
    # Trim boundaries reconstructed by polygon intersections can differ by a few
    # ulps. Buffer only for line membership, keeping the cell bounds unchanged.
    domain = (unary_union(cell.material_polygons).buffer(tolerance).intersection(rectangle)
              if cell.trimmed else rectangle)
    neighbors = []
    if hasattr(cell, '_mesh_elements'):
        for other in cell._mesh_elements:
            if other.void:
                continue
            # Reject separated bounds before constructing Shapely rectangles.
            if (other.u_max < cell.u_min or other.u_min > cell.u_max or
                    other.v_max < cell.v_min or other.v_min > cell.v_max):
                continue
            bounds = box(other.u_min, other.v_min, other.u_max, other.v_max)
            if not bounds.intersects(rectangle):
                continue
            material = unary_union(other.material_polygons).buffer(tolerance).intersection(bounds) if other.trimmed else bounds
            neighbors.append(material)

    def segments(geometry):
        if geometry.is_empty:
            return
        if geometry.geom_type == 'LineString':
            if geometry.length > 0:
                yield geometry
        elif geometry.geom_type in ('MultiLineString', 'GeometryCollection'):
            for part in geometry.geoms:
                yield from segments(part)
        # Isolated points have zero line measure.

    result = []
    offset = 0.0
    for candidate, a, b in zip(candidates, starts, ends):
        # Most segments of a finely sampled trim contour are outside this cell.
        # Reject them before constructing/intersecting Shapely geometries.
        segment_length = np.hypot(b[0]-a[0], b[1]-a[1])
        if not candidate:
            offset += segment_length
            continue
        segment = LineString([a, b])
        if segment.length == 0:
            continue
        intervals = []
        for part in segments(segment.intersection(domain)):
            s0 = segment.project(Point(part.coords[0]))
            s1 = segment.project(Point(part.coords[-1]))
            intervals.append((offset + min(s0, s1), offset + max(s0, s1)))
        for start, end in sorted(intervals):
            cuts = {start, end}
            # Split where the number of material neighbors changes.
            for material in neighbors:
                for part in segments(segment.intersection(material)):
                    for coordinate in (part.coords[0], part.coords[-1]):
                        s = offset + segment.project(Point(coordinate))
                        if start < s < end:
                            cuts.add(s)
            ordered = sorted(cuts)
            for left, right in zip(ordered[:-1], ordered[1:]):
                midpoint = segment.interpolate((left+right)/2-offset)
                owners = sum(material.covers(midpoint) for material in neighbors) if neighbors else 1
                owners = max(1, owners)
                result.extend((s, w/owners) for s, w in get_iwip(cell, left, right, quadrature_order))
        offset += segment.length
    return result
