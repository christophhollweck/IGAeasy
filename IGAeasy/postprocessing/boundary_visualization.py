"""Map physical boundary curves and shell support symbols onto spline surfaces."""
import numpy as np
import pyvista as pv
from shapely.geometry import box, Point
from shapely.ops import unary_union


def _line_parts(geometry):
    if geometry.geom_type in ('LineString', 'LinearRing'):
        if geometry.length > 0:
            yield geometry
    else:
        for part in getattr(geometry, 'geoms', []):
            yield from _line_parts(part)


def _mapped_lines(space, line, deformation_scale, samples):
    """Clip each curve to active material before evaluating its cell-sided basis."""
    mapped = []
    seen = set()
    for cell in space.elements:
        if cell.void or not getattr(cell, 'active', True):
            continue
        material = (unary_union(cell.material_polygons) if cell.trimmed else
                    box(cell.u_min, cell.v_min, cell.u_max, cell.v_max))
        basis = list(cell.supported_b_splines)
        cps = np.asarray([b.cp for b in basis])
        displacements = np.asarray([b.displacements for b in basis])
        # Polygon clipping may move a trim edge by roundoff. Include its trace,
        # but keep samples inside the cell where its basis is evaluated.
        cell_box = box(cell.u_min, cell.v_min, cell.u_max, cell.v_max)
        tolerance = 1e-12*max(cell.u_max-cell.u_min, cell.v_max-cell.v_min)
        clipped = line.intersection(cell_box).intersection(material.buffer(tolerance))
        for part in _line_parts(clipped):
            positions, normals = [], []
            # Retain polyline corners rather than interpolating across them.
            coordinates = np.asarray(part.coords)
            midpoint = Point((coordinates[0]+coordinates[1])/2)
            distance = line.project(midpoint)
            step = min(part.length*.01, line.length*1e-6)
            source_tangent = (np.asarray(line.interpolate(distance+step).coords)[0]
                              -np.asarray(line.interpolate(distance-step).coords)[0])
            # Intersection output may reverse an edge; restore the original curve orientation.
            if np.dot(coordinates[1]-coordinates[0], source_tangent) < 0:
                coordinates = coordinates[::-1]
            key = tuple(map(tuple, np.round(coordinates, 12)))
            if key in seen:
                continue
            seen.add(key)
            for start, end in zip(coordinates[:-1], coordinates[1:]):
                for u, v in np.linspace(start, end, samples):
                    N = np.array([b(u, v, cell=cell) for b in basis])
                    gradients = np.array([b.grad(u, v, cell=cell) for b in basis])
                    tangents = gradients.T@(cps+deformation_scale*displacements)
                    normal = np.cross(*tangents)
                    normals.append(normal/np.linalg.norm(normal))
                    positions.append(N@(cps+deformation_scale*displacements))
            if positions:
                mapped.append((line.project(Point(coordinates[0])),
                               (np.asarray(positions), np.asarray(normals))))
    return [curve for _, curve in sorted(mapped, key=lambda entry: entry[0])]


def _polylines(curves):
    points, cells = [], []
    for curve in curves:
        offset = len(points)
        points.extend(curve)
        cells.extend([len(curve), *range(offset, offset+len(curve))])
    return pv.PolyData(np.asarray(points), lines=np.asarray(cells)) if points else pv.PolyData()


def _support_kind(boundary):
    from IGAeasy.constraints.membrane_weak import WeakMembraneBoundary
    from IGAeasy.constraints.shell_weak import WeakRM5Boundary, WeakRM6Boundary
    from IGAeasy.constraints.kl_clamped_weak import WeakKLClampedBoundary, WeakRM5HierClampedBoundary
    from IGAeasy.constraints.kl_mixed_weak import WeakKLMixedBoundary
    from IGAeasy.constraints.kl_symmetry_weak import WeakKLSymmetryBoundary
    from IGAeasy.constraints.rm5_hier_weak import WeakRM5HierBoundary
    if isinstance(boundary, WeakMembraneBoundary) or getattr(boundary, 'support_kind', None) == 'Translation support':
        return 'Translation support', boundary.projector
    if getattr(boundary, 'normal_only', False):
        return 'Normal support', ()
    if isinstance(boundary, WeakKLSymmetryBoundary):
        return 'Reflection symmetry', ()
    if isinstance(boundary, WeakKLMixedBoundary):
        if boundary.normal_rotation:
            return 'Mixed / director constraint', ()
        return 'Translation support', boundary.displacement_projector
    if isinstance(boundary, (WeakRM5Boundary, WeakRM6Boundary, WeakRM5HierBoundary)):
        if callable(getattr(boundary, 'directions', None)):
            return 'Mixed / director constraint', ()
        projector = boundary.projector
        if np.allclose(projector.T@projector, np.eye(6)):
            return 'Clamped', ()
        if np.allclose(projector[:, 3:], 0.):
            return 'Translation support', projector[:, :3]
        return 'Mixed / director constraint', ()
    if type(boundary) in (WeakKLClampedBoundary, WeakRM5HierClampedBoundary):
        return 'Clamped', ()
    raise ValueError('Support visualization currently accepts RM5/RM6 and fully clamped KL boundaries')


def build_boundary_layers(space, boundary_conditions=(), deformation_scale=1.,
                          samples_per_element=5):
    """Return named actor groups for plot_surface_results(overlays=...).

    Conditions are the actual weak-boundary objects used by the solver.
    Colors identify constrained traces, not prescribed values or reactions.
    """
    if samples_per_element < 2:
        raise ValueError('Use at least two curve samples')
    layers = {}
    cells = [c for c in space.elements if getattr(c, 'active', True) and not c.void]
    if any(c.trimmed for c in cells):
        material = unary_union([polygon for c in cells for polygon in
            (c.material_polygons if c.trimmed else [box(c.u_min, c.v_min, c.u_max, c.v_max)])])
        mapped = _mapped_lines(space, material.boundary, deformation_scale, samples_per_element)
        layers['Trim boundaries'] = [(_polylines([p for p, _ in mapped]), 'darkorange', 'Material boundary')]

    supports = []
    for boundary in boundary_conditions:
        kind, _ = _support_kind(boundary)
        mapped = _mapped_lines(space, boundary.line, deformation_scale, samples_per_element)
        if not mapped:
            continue
        curves = [points for points, _ in mapped]
        color = {'Clamped': 'royalblue', 'Translation support': 'seagreen',
                 'Normal support': 'seagreen', 'Reflection symmetry': 'purple',
                 'Mixed / director constraint': 'purple'}[kind]
        supports.append((_polylines(curves), color, f'{kind} ({boundary.method})'))
    if supports:
        layers['Supports'] = supports
    return layers
