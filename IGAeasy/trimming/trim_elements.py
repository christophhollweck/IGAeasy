from shapely.geometry import box, Polygon, Point, LineString
import triangle
import numpy as np
from collections import defaultdict
from IGAeasy.trimming.surrogate_boundary import ActiveEdge
from IGAeasy.differential_geometry.line import compute_line_normal_and_tangent


def contains_material_point(cell, xi, eta):
    """Parametric material membership, including outer and hole boundaries."""
    if getattr(cell, 'void', False) or not getattr(cell, 'active', True):
        return False
    if not (cell.u_min <= xi <= cell.u_max and cell.v_min <= eta <= cell.v_max):
        return False
    if not getattr(cell, 'trimmed', False):
        return True
    polygons = getattr(cell, 'material_polygons', None)
    if polygons is None:
        raise ValueError('Trimmed element has no material polygons; trim the geometry first')
    point = Point(xi, eta)
    return any(polygon.covers(point) for polygon in polygons)



def trim(RS):

    mark(RS)
    polygonize(RS)
    triangulate(RS)

def trim_SBM(RS, shift_distance=0.0, exclude_boundary=False):

    trim(RS)

    for el in RS.elements:
        if el.trimmed == True:
            if el.material_ratio >= 0.5:
                el.void = False
                el.trimmed = False
            else:
                el.void = True
                el.trimmed = False

    # find active edges
    detect_outer_sides_remove_duplicates(RS, ndigits=16, shift_distance=shift_distance, exclude_boundary=exclude_boundary)


def mark(RS):
    """Marks elements as void or trimmed based on their spatial relation to BReps.

    Rules:
    - If the element is completely outside the outerloop → void.
    - If the element is completely inside an innerloop → void.
    - If the element intersects any BRep → trimmed.
    - Otherwise → active (neither void nor trimmed).
    """
    outerloop_brep = next((B for B in RS.BReps if B.outerloop), None)

    for el in RS.elements:
        element_polygon = box(el.u_min, el.v_min, el.u_max, el.v_max)
        el.void = False
        el.trimmed = False
        el.breps = []

        # Check for intersection with any BRep
        for brep in RS.BReps:
            if brep.polygon.boundary.intersects(element_polygon):
                el.trimmed = True
                el.void = False
                el.breps.append(brep)

        if el.trimmed:
            continue  # already marked as trimmed, no further checks needed

        # Check if element is fully inside an innerloop
        for brep in RS.BReps:
            if not brep.outerloop and brep.polygon.contains(element_polygon):
                el.void = True
                break

        # Check if element is outside the outerloop
        if not el.void and outerloop_brep is not None:
            if not outerloop_brep.polygon.intersects(element_polygon):
                el.void = True

    mark_basis_functions_from_elements(RS)




def polygonize(RS):
    """Computes material and void polygons for each element."""
    for el in RS.elements:
        rect_poly = box(el.u_min, el.v_min, el.u_max, el.v_max)  # Element as a rectangular polygon
        el.area = rect_poly.area
        outerloop = None
        innerloops = []

        # Split Breps into Outerloop and Innerloops
        for brep in el.breps:
            if not brep.polygon.is_valid:
                brep.polygon = brep.polygon.buffer(0)  # repair

            if brep.outerloop:
                outerloop = brep.polygon
            else:
                innerloops.append(brep.polygon)

        # Determine material region
        if outerloop:
            material_domain = rect_poly.intersection(outerloop)
        else:
            material_domain = rect_poly

        # Subtract innerloops
        for innerloop in innerloops:
            material_domain = material_domain.difference(innerloop)

        # Void region = Everything outside the material domain
        void_domain = rect_poly.difference(material_domain)

        # Safeguard against invalid geometries
        material_domain = material_domain.buffer(0)
        void_domain = void_domain.buffer(0)

        # Extract final material and void polygons
        el.material_polygons = extract_polygons(material_domain)

        el.void_polygons = extract_polygons(void_domain)

        el.material_ratio = np.sum([poly.area for poly in el.material_polygons])/el.area


def extract_polygons(geom):
    """Extracts individual polygons from a Shapely geometry."""
    if geom.is_empty:
        return []
    elif geom.geom_type == "Polygon":
        return [geom]
    elif geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    return []



def triangulate(RS):
    for el in RS.elements:
        # # use this to simplify brep polygons to safe integrationpoints
        # el.coarse_material_polygons = [simplify_polygon_douglas_peucker(adaptive_sampling(poly)) for poly in el.material_polygons]
        el.coarse_material_polygons = el.material_polygons
        el.material_triangles = extract_triangles(el.material_polygons)
        el.material_triangles = [item for sublist in el.material_triangles for item in sublist]
        el.coarse_material_triangles = extract_triangles(el.coarse_material_polygons)

        # # use this to simplify brep polygons to safe integrationpoints
        # el.coarse_void_polygons = [simplify_polygon_douglas_peucker(adaptive_sampling(poly)) for poly in el.void_polygons]
        el.coarse_void_polygons = el.void_polygons
        el.void_triangles = extract_triangles(el.void_polygons)
        el.void_triangles = [item for sublist in el.void_triangles for item in sublist]
        el.coarse_void_triangles = extract_triangles(el.coarse_void_polygons)


def simplify_polygon_douglas_peucker(poly, tolerance=0.001):
    """Simplifies a polygon using the Douglas-Peucker algorithm."""
    simplified = LineString(poly.exterior.coords).simplify(tolerance, preserve_topology=True)
    return Polygon(simplified)


def extract_triangles(polygons):
    triangles = []

    for polygon in polygons:
        poly_triangles = triangulate_polygon(polygon)
        triangles.append(poly_triangles)  # Add triangles to the list

    return triangles


def triangulate_polygon(polygon):
    """
    Triangulates a polygon while ensuring that only valid triangles within the polygon are kept.

    Parameters:
    polygon (Polygon): The input polygon to be triangulated.

    Returns:
    list: A list of valid triangle polygons.
    """

    if polygon.is_empty:
        return []
    if not polygon.is_valid:
        raise ValueError("Triangulation requires a valid polygon.")
    points, segments, holes = [], [], []
    for ring in [polygon.exterior, *polygon.interiors]:
        coordinates = list(ring.coords[:-1])
        offset = len(points)
        points.extend(coordinates)
        segments.extend([[offset+i, offset+(i+1) % len(coordinates)]
                         for i in range(len(coordinates))])
    for ring in polygon.interiors:
        # The centroid of a concave hole can lie outside it.
        point = Polygon(ring).representative_point()
        holes.append([point.x, point.y])
    poly_dict = {"vertices": points, "segments": segments}
    if holes:
        poly_dict['holes'] = holes

    # Perform constrained triangulation
    triangulated = triangle.triangulate(poly_dict, 'p')

    # Extract valid triangles
    valid_triangles = []
    for tri in triangulated.get("triangles", []):
        pts = triangulated["vertices"][tri]
        tri_polygon = Polygon(pts)

        # Check if the centroid is inside the original polygon
        centroid = Point(np.mean(pts, axis=0))
        if polygon.contains(centroid):
            valid_triangles.append(tri_polygon)


    return valid_triangles


def adaptive_sampling(poly, threshold=170, max_iter=20):
    """Reduces the number of points adaptively based on the angles between adjacent line segments."""

    points = list(poly.exterior.coords[:-1])  # Remove the last point (closed loop)
    prev_len = len(points)

    for j in range(max_iter):
        temp_points_iter = [points[0]]  # Keep the first point

        for i in range(0, len(points) - 2, 2):  # Iterate through the points except the last one
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[(i + 2) % len(points)]  # If i+2 is out of bounds, take the first point

            angle = calculate_angle(p1, p2, p3)

            if angle < threshold:
                temp_points_iter.append(p2)  # Keep p2 if the angle is below the threshold

            temp_points_iter.append(p3)  # Always add p3

        if temp_points_iter[-1] != points[-1]:
            temp_points_iter.append(points[-1])  # Ensure the last point is retained

        current_len = len(temp_points_iter)
        if prev_len == current_len:
            break  # Stop if no points were removed

        prev_len = current_len
        points = temp_points_iter  # Update the points for the next iteration

    return Polygon(points + [points[0]])  # Close the polygon again


def calculate_angle(p1, p2, p3):
    """Calculates the angle between the line segments (p1->p2) and (p2->p3)."""

    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Vectors from p1 to p2 and from p2 to p3
    v1 = p2 - p1
    v2 = p3 - p2

    # Compute the angle
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Angle in radians

    return 180 - np.degrees(angle)  # Convert to degrees

def mark_basis_functions_from_elements(RS):
    """Mark basis functions as active, trimmed, or void based on their supporting elements."""

    # Reset basis-function flags and counters
    for b in RS.basis:
        b.trimmed = False
        b.void = True

        b.n_active_elements = 0
        b.n_trimmed_elements = 0
        b.n_void_elements = 0

    # Count supporting element types for each basis function
    for el in RS.elements:
        el_is_void = getattr(el, "void", False)
        el_is_trimmed = getattr(el, "trimmed", False)

        for b in el.supported_b_splines:
            if el_is_void:
                b.n_void_elements += 1
                continue

            b.void = False
            b.n_active_elements += 1

            if el_is_trimmed:
                b.trimmed = True
                b.n_trimmed_elements += 1

    # Safety cleanup
    for b in RS.basis:
        if b.void:
            b.trimmed = False


def round_coords(coords, ndigits):
    return tuple(round(c, ndigits) for c in coords)

def canonical_linestring(ls: LineString, ndigits=13):
    coords = [round_coords(pt, ndigits) for pt in ls.coords]
    return tuple(coords) if coords < coords[::-1] else tuple(coords[::-1])

def get_side_segments(el, ndigits=13):
    r = lambda x: round(x, ndigits)
    p0 = (r(el.u_min), r(el.v_min))
    p1 = (r(el.u_max), r(el.v_min))
    p2 = (r(el.u_max), r(el.v_max))
    p3 = (r(el.u_min), r(el.v_max))
    return [
        (0, LineString([p0, p1])),
        (1, LineString([p1, p2])),
        (2, LineString([p2, p3])),
        (3, LineString([p3, p0])),
    ]

def opposing_side(side_idx):
    return {0: 2, 1: 3, 2: 0, 3: 1}[side_idx]

def is_horizontal(line: LineString):
    """Returns True if line is horizontal (dominant x-direction), else False."""
    x0, y0 = line.coords[0]
    x1, y1 = line.coords[-1]
    return abs(x1 - x0) >= abs(y1 - y0)

def shift_line_axis_aligned(line: LineString, x_center, y_center, shift_distance=1e-14):
    """
    Shift horizontal lines vertically toward y_center,
    vertical lines horizontally toward x_center.

    Returns:
        shifted_line: LineString
        is_horizontal: bool
        side: int (0=bottom, 1=right, 2=top, 3=left)
    """
    cx, cy = line.centroid.x, line.centroid.y
    horizontal = is_horizontal(line)

    if horizontal:
        direction = np.sign(y_center - cy)
        shifted = LineString([(x, y + direction * shift_distance) for x, y in line.coords])
        if direction > 0:
            side = 0  # bottom
        else:
            side = 2  # top
    else:
        direction = np.sign(x_center - cx)
        shifted = LineString([(x + direction * shift_distance, y) for x, y in line.coords])
        if direction > 0:
            side = 3  # left
        else:
            side = 1  # right

    return shifted, horizontal, side


def detect_outer_sides_remove_duplicates(RS, ndigits=13, tol=1e-13, shift_distance=0, exclude_boundary=False):
    """
    Compute surrogate boundary (active sides) for all non-void elements,
    activate void elements with exactly three active edges by toggling edge statuses,
    and update their active sides accordingly.
    """

    edge_map = defaultdict(list)  # canonical key -> list of (LineString, element, side_index)

    # 1. Collect sides of non-void elements
    for el in RS.elements:
        if el.void:
            continue
        for side_idx, seg in get_side_segments(el, ndigits):
            key = canonical_linestring(seg, ndigits)
            edge_map[key].append((seg, el, side_idx))

    # 2. Keep only edges that appear once (potential boundary)
    single_edges = {
        key: seg_list[0]
        for key, seg_list in edge_map.items()
        if len(seg_list) == 1
    }

    # 3. Build lookup by (element, side_index)
    element_side_map = {
        (el, side_idx): seg
        for seg, el, side_idx in single_edges.values()
    }

    # 4. Subtract overlapping opposing sides to get cleaned boundaries
    cleaned_element_sides = defaultdict(list)
    for (el, side_idx), line in element_side_map.items():
        opp_idx = opposing_side(side_idx)
        candidates = [
            seg for (e2, i2), seg in element_side_map.items()
            if e2 != el and i2 == opp_idx
        ]
        reduced = line
        for c in candidates:
            if reduced.is_empty:
                break
            reduced = reduced.difference(c)

        if reduced.is_empty or reduced.length < tol:
            continue
        if reduced.geom_type == 'LineString':
            cleaned_element_sides[el].append(reduced)
        elif reduced.geom_type == 'MultiLineString':
            for part in reduced.geoms:
                if part.length >= tol:
                    cleaned_element_sides[el].append(part)

    # 5. Assign surrogate boundary edges to elements
    for el in RS.elements:
        if exclude_boundary:
            tol_b = 1e-13
            if (np.isclose(el.u_min, RS.ku[0], atol=tol_b) or
                np.isclose(el.v_min, RS.kv[0], atol=tol_b) or
                np.isclose(el.u_max, RS.ku[-1], atol=tol_b) or
                np.isclose(el.v_max, RS.kv[-1], atol=tol_b)):
                el.active_sides = []
                continue
        el.active_sides = cleaned_element_sides.get(el, [])

    # # 6. Activate void elements with exactly 3 active edges + BRep check
    # for el in RS.elements:
    #     if not el.void:
    #         continue
    #
    #     side_segments = {idx: seg for idx, seg in get_side_segments(el, ndigits)}
    #     active_indices = []
    #     for idx, seg in side_segments.items():
    #         key = canonical_linestring(seg, ndigits)
    #         if key in single_edges:
    #             active_indices.append(idx)
    #
    #     if len(active_indices) != 3:
    #         continue
    #
    #     # --- additional check with BRep intersections ---
    #     if len(el.breps) != 1:
    #         continue
    #     brep = el.breps[0]
    #     element_polygon = box(el.u_min, el.v_min, el.u_max, el.v_max)
    #     intersections = brep.polygon.exterior.intersection(element_polygon)
    #
    #     # collect intersection points
    #     points = []
    #     if intersections.is_empty:
    #         continue
    #     elif intersections.geom_type == 'Point':
    #         points = [intersections]
    #     elif intersections.geom_type == 'MultiPoint':
    #         points = list(intersections.geoms)
    #     elif intersections.geom_type == 'LineString':
    #         coords = intersections.coords
    #         points = [Point(coords[0]), Point(coords[-1])]
    #     elif intersections.geom_type == 'MultiLineString':
    #         for seg in intersections.geoms:
    #             coords = seg.coords
    #             points.append(Point(coords[0]))
    #             points.append(Point(coords[-1]))
    #     else:
    #         continue
    #
    #     if len(points) != 2:
    #         continue
    #
    #     # Determine which sides are cut
    #     cut_info = []
    #     for idx, seg in side_segments.items():
    #         for p in points:
    #             if seg.distance(p) < 1e-13:  # Point-on-line tolerance
    #                 cut_info.append((idx, p))
    #                 break
    #
    #     if len(cut_info) != 2:
    #         continue
    #
    #
    #     side_indices = sorted(i for i, _ in cut_info)
    #     if not (side_indices == [0, 2] or side_indices == [1, 3]):
    #         continue
    #
    #     # Angle check: activate if at least one angle < 10 degrees
    #     max_angle = np.deg2rad(10)
    #     activate = False  # start with False
    #     for side_idx, pnt in cut_info:
    #         n_line, t_line = compute_line_normal_and_tangent(brep.polygon.exterior, pnt)
    #         edge_n = np.array([0.0, 1.0]) if side_idx in (0, 2) else np.array([1.0, 0.0])
    #         angle = np.arccos(abs(np.dot(t_line, edge_n)))
    #         if angle < max_angle:  # only one small angle needed
    #             activate = True
    #             break
    #
    #     if not activate:
    #         continue
    #
    #     # # --- activate element ---
    #     # print(f"Activating void element with 3 active edges: "
    #     #       f"u_min={el.u_min}, u_max={el.u_max}, v_min={el.v_min}, v_max={el.v_max}")
    #
    #     el.void = False
    #     el.trimmed = False
    #     inactive_idx = ({0, 1, 2, 3} - set(active_indices)).pop()
    #     inactive_seg = side_segments[inactive_idx]
    #
    #     # Deactivate 3 active edges in neighbors and own active_sides
    #     for idx in active_indices:
    #         seg = side_segments[idx]
    #         key = canonical_linestring(seg, ndigits)
    #         neighbors = edge_map.get(key, [])
    #         for seg2, neighbor_el, neighbor_side_idx in neighbors:
    #             if neighbor_el is not el:
    #                 neighbor_el.active_sides = [
    #                     s for s in neighbor_el.active_sides if not s.equals(seg2)
    #                 ]
    #         el.active_sides = [
    #             s for s in el.active_sides if not s.geometry.equals(seg)
    #         ]
    #
    #     # Activate previously inactive edge
    #     x_center = np.mean([el.u_min, el.u_max])
    #     y_center = np.mean([el.v_min, el.v_max])
    #     shifted, horizontal, side = shift_line_axis_aligned(
    #         inactive_seg, x_center, y_center, shift_distance
    #     )
    #     el.active_sides.append(ActiveEdge(shifted, horizontal, side, el.p + 1, el))

    # 7. Small displacement toward element center
    for el in RS.elements:
        x_center = np.mean([el.u_min, el.u_max])
        y_center = np.mean([el.v_min, el.v_max])
        shifted_lines = []
        for line in el.active_sides:
            if isinstance(line, ActiveEdge):
                shifted_lines.append(line)
                continue
            shifted, horizontal, side = shift_line_axis_aligned(line, x_center, y_center, shift_distance)
            shifted_lines.append(ActiveEdge(shifted, horizontal, side, el.p + 1, el))
        el.active_sides = shifted_lines

    # 8. Deactivate elements with exactly 4 active edges
    for el in RS.elements:
        if el.void:
            continue
        if len(el.active_sides) == 4:
            el.void = True
            el.active_sides = []
            # print(f"Deactivating element with 4 active edges: "
            #       f"u_min={el.u_min}, u_max={el.u_max}, v_min={el.v_min}, v_max={el.v_max}")


