import numpy as np
from shapely.geometry import LineString, box, Point

def compute_tangent_line(line, s):
    """
    Computes the tangent vector (∂xi/∂s, ∂eta/∂s) at position s ∈ [0, 1] on a LineString.
    The LineString is assumed to be parametrized over [0, 1] with arc length.

    Parameters:
        line (LineString): The input line (must have at least 2 points).
        s (float): The position along the line.

    Returns:
        (∂xi/∂s, ∂eta/∂s): The covariant derivative of (xi, eta) with respect to s.
    """

    coords = list(line.coords)
    total_length = line.length

    accumulated_length = 0.0
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]

        dx = x1 - x0
        dy = y1 - y0
        segment_length = (dx ** 2 + dy ** 2) ** 0.5

        if accumulated_length + segment_length >= s:
            # Tangent vector in segment direction scaled by total length
            direction_x = dx / segment_length
            direction_y = dy / segment_length
            return (direction_x * total_length, direction_y * total_length)

        accumulated_length += segment_length

    # If s == 1.0 exactly, return tangent of last segment
    x0, y0 = coords[-2]
    x1, y1 = coords[-1]
    dx = x1 - x0
    dy = y1 - y0
    segment_length = (dx ** 2 + dy ** 2) ** 0.5
    direction_x = dx / segment_length
    direction_y = dy / segment_length
    return (direction_x * total_length, direction_y * total_length)

def compute_normal(line, s):
    """Outward right-hand unit normal for an oriented boundary polyline."""
    return compute_line_normal_and_tangent(line, line.interpolate(s))[0]


def compute_line_normal_and_tangent(line, pt):
    """Use the exact segment direction, avoiding tiny finite differences.

    The tangent retains the historical reverse-line convention. At vertices
    the incoming segment is selected; the normal is right of line traversal.
    """
    position = line.project(pt)
    offset = 0.0
    coords = np.asarray(line.coords, dtype=float)
    last = None
    for a, b in zip(coords[:-1], coords[1:]):
        delta = b-a
        length = np.linalg.norm(delta)
        if length == 0:
            continue
        tangent = -delta/length
        last = tangent
        if position <= offset+length:
            return np.array([-tangent[1], tangent[0]]), tangent
        offset += length
    if last is None:
        raise ValueError("Cannot compute a normal for a zero-length line.")
    return np.array([-last[1], last[0]]), last

def compute_closest_point(line, pnt):
    """Project integration points on the closest BREP."""

    s = line.project(pnt)  # line parameter (distance along the line)
    closest_point = line.interpolate(s)

    dx = closest_point.x - pnt.x
    dy = closest_point.y - pnt.y
    difference_vector = np.array([dx, dy])

    return closest_point, s, difference_vector