"""Material-aware visualization of spline surfaces and scalar result fields."""
import numpy as np
import pyvista as pv
from shapely.geometry import box, LineString
from shapely.ops import unary_union
from IGAeasy.trimming.trim_elements import extract_triangles
from IGAeasy.postprocessing.boundary_visualization import _mapped_lines, _polylines
from IGAeasy.postprocessing.result_controls import scalar_range


def plot_solution(space, fields=('displacement_magnitude',), deformation_scale=1.,
                  samples_per_element=4, projected_fields=None, boundary_conditions=None,
                  result_selection=True, **plot_options):
    """Show one result window with a checkbox for undeformed/deformed geometry."""
    from IGAeasy.postprocessing.boundary_visualization import build_boundary_layers
    conditions = (getattr(space, 'boundary_conditions', ()) if boundary_conditions is None
                  else boundary_conditions)
    result_fields = None
    geometry_fields = fields
    selectable = ('ux', 'uy', 'uz', 'displacement_magnitude', 'stress_von_mises',
                  'stress_11', 'stress_22', 'stress_12', 'strain_11', 'strain_22', 'strain_12')
    if (len(fields) == 1 and fields[0] in selectable
            and hasattr(space, 'shell_formulation') and projected_fields is None):
        from IGAeasy.postprocessing.shell_fields import ShellResultFields
        result_fields = ShellResultFields(space)
        geometry_fields = ('ux', 'uy', 'uz', 'displacement_magnitude')

    def geometry(scale):
        mesh, edges = build_surface_data(space, scale, samples_per_element, geometry_fields, projected_fields)
        overlays = build_boundary_layers(space, conditions, scale, samples_per_element)
        return mesh, edges, overlays

    mesh, edges, overlays = geometry(deformation_scale)
    original = geometry(0.)
    if result_fields is not None and fields[0] in result_fields.fields:
        values = result_fields.sample(mesh, fields[0])
        mesh[fields[0]] = original[0][fields[0]] = values
    return plot_surface_results(mesh, edges, fields, deformation_scale,
                                overlays=overlays, undeformed=original,
                                result_fields=result_fields if result_selection else None, **plot_options)


def build_surface_data(space, deformation_scale=1, samples_per_element=4,
                       fields=('uz', 'stress_von_mises'), projected_fields=None):
    if samples_per_element < 2:
        raise ValueError('samples_per_element must be at least two')
    supported = ('ux', 'uy', 'uz', 'displacement_magnitude', 'scalar_value', 'stress_von_mises')
    if not fields or any(field not in supported for field in fields):
        raise ValueError(f'fields must be selected from {supported}')
    projected_fields = {} if projected_fields is None else projected_fields
    points, faces, parametric_edges = [], [], []
    coordinates, cell_indices = [], []
    data = {field: [] for field in fields}
    seen_segments = set()
    for cell_index, cell in enumerate(space.elements):
        if getattr(cell, 'void', False) or not getattr(cell, 'active', True):
            continue
        basis = list(cell.supported_b_splines)
        polygons = cell.material_polygons if cell.trimmed else [box(cell.u_min, cell.v_min, cell.u_max, cell.v_max)]

        def sample(u, v):
            values = np.array([b(u, v, cell=cell) for b in basis])
            geometry = values @ np.array([b.cp for b in basis])
            displacement = values @ np.array([b.displacements for b in basis])
            return geometry+deformation_scale*displacement, displacement, values

        us = np.linspace(cell.u_min, cell.u_max, samples_per_element)
        vs = np.linspace(cell.v_min, cell.v_max, samples_per_element)
        for polygon in polygons:
            for ring in [polygon.exterior, *polygon.interiors]:
                coords = np.asarray(ring.coords)
                for start, end in zip(coords[:-1], coords[1:]):
                    segment_key = tuple(sorted((tuple(start), tuple(end))))
                    if segment_key in seen_segments:
                        continue
                    seen_segments.add(segment_key)
                    parametric_edges.append(LineString([start, end]))
            for u0, u1 in zip(us[:-1], us[1:]):
                for v0, v1 in zip(vs[:-1], vs[1:]):
                    intersection = polygon.intersection(box(u0, v0, u1, v1))
                    parts = ([intersection] if intersection.geom_type == 'Polygon' else
                             [p for p in getattr(intersection, 'geoms', []) if p.geom_type == 'Polygon'])
                    for part in parts:
                        if part.is_empty or part.area == 0:
                            continue
                        for triangle in extract_triangles([part])[0]:
                            offset = len(points)
                            for u, v in list(triangle.exterior.coords)[:3]:
                                position, displacement, values = sample(u, v)
                                points.append(position)
                                coordinates.append((u, v))
                                cell_indices.append(cell_index)
                                for field in fields:
                                    if field in projected_fields:
                                        value = values @ np.array([projected_fields[field][b.id] for b in basis])
                                    elif field in ('ux', 'uy', 'uz'):
                                        value = displacement[('ux', 'uy', 'uz').index(field)]
                                    elif field == 'displacement_magnitude':
                                        value = np.linalg.norm(displacement)
                                    elif field == 'scalar_value':
                                        value = values @ np.array([b.scalar_value for b in basis])
                                    else:
                                        if not hasattr(cell, 'vM_interp'):
                                            raise ValueError('Interpolate stresses or supply projected_fields first')
                                        value = cell.vM_interp([[u, v]]).item()
                                    data[field].append(value)
                            faces.append([3, offset, offset+1, offset+2])
    if not points:
        raise ValueError('No active material surface to plot')
    mesh = pv.PolyData(np.asarray(points), np.asarray(faces).ravel())
    mesh.field_data['parametric_coordinates'] = np.asarray(coordinates)
    mesh.field_data['element_indices'] = np.asarray(cell_indices)
    for field, values in data.items():
        mesh.point_data[field] = np.asarray(values)
    # Split hanging-node overlaps before mapping, so each geometric edge occurs once.
    edge_network = unary_union(parametric_edges)
    mapped = _mapped_lines(space, edge_network, deformation_scale, samples_per_element)
    boundaries = _polylines([curve for curve, _ in mapped])
    return mesh, boundaries


def plot_surface_results(mesh, boundaries=None, fields=None, deformation_scale=1,
                         show_element_boundaries=True, show=True, off_screen=False,
                         scalar_limits=None, overlays=None, show_controls=True,
                         show_legend=True, show_supports=False,
                         show_surface=True, color_by_field=True, undeformed=None, result_fields=None):
    """Plot scalar results with optional boundary overlays and shared visibility controls."""
    fields = tuple(mesh.point_data.keys()) if fields is None else tuple(fields)
    if not fields:
        raise ValueError('Select at least one scalar field')
    plotter = pv.Plotter(shape=(1, len(fields)), off_screen=off_screen)
    groups = {} if overlays is None else dict(overlays)
    if boundaries is not None and boundaries.n_points:
        groups['Element edges'] = [(boundaries, 'black', 'Element edges')]
    actors = {name: [] for name in groups}
    surface_actors, scalar_bars = [], []
    display = {'Color field': color_by_field, 'Surface': show_surface}
    geometry_actors = []
    visibility = {name: (show_element_boundaries if name == 'Element edges' else
                         show_supports if name == 'Supports' else True) for name in groups}

    for column, field in enumerate(fields):
        values = np.asarray(mesh.point_data[field])
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f'{field} must be a finite scalar point field')
        low, high = scalar_range(values, (scalar_limits or {}).get(field))
        plotter.subplot(0, column)
        # Separate copies prevent one subplot from changing another's active scalars.
        surface = plotter.add_mesh(mesh, scalars=field, cmap='coolwarm', clim=(low, high),
                         copy_mesh=True, show_edges=False, interpolate_before_map=True,
                         scalar_bar_args={'title': field}, name='result_surface')
        surface.GetProperty().SetColor(.85, .85, .85)
        surface_actors.append(surface)
        if undeformed is not None:
            geometry_actors.append((surface, mesh, undeformed[0]))
        scalar_bars.append(plotter.scalar_bars[field])
        for name, entries in groups.items():
            for i, (data, color, label) in enumerate(entries):
                if not data.n_points:
                    continue
                actor = plotter.add_mesh(data, color=color,
                    line_width={'Element edges': 1, 'Trim boundaries': 2, 'Supports': 4}.get(name, 2),
                    render_lines_as_tubes=False, name=f'overlay:{name}:{i}', copy_mesh=True)
                if undeformed is not None:
                    reference = (undeformed[1] if name == 'Element edges' else undeformed[2][name][i][0])
                    geometry_actors.append((actor, data, reference))
                actor.SetVisibility(visibility[name])
                # Keep coplanar lines just ahead of the filled surface in the depth buffer.
                actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
                actor.mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0., -2.)
                actors[name].append(actor)
        if result_fields is None:
            plotter.add_text(f'{field}\nScale when deformed: {deformation_scale}x', font_size=10)
        plotter.add_axes()
        plotter.view_isometric()
    plotter.link_views()

    def update_surface():
        for surface, bar in zip(surface_actors, scalar_bars):
            surface.SetVisibility(display['Surface'])
            surface.GetMapper().SetScalarVisibility(display['Color field'])
            bar.SetVisibility(display['Surface'] and display['Color field'])

    update_surface()

    # One control panel changes every linked view without rebuilding plot data.
    plotter.subplot(0, len(fields)-1)
    if show_controls:
        controls = []
        for name in display:
            def toggle_display(checked, name=name):
                display[name] = checked
                update_surface()
                plotter.render()

            controls.append((name, display[name], toggle_display))

        for name, group in actors.items():
            if group:
                def toggle(checked, group=group):
                    for actor in group:
                        actor.SetVisibility(checked)
                    plotter.render()

                controls.append((name, visibility[name], toggle))

        if undeformed is not None:
            def toggle_deformation(checked):
                for actor, deformed, original in geometry_actors:
                    actor.mapper.dataset.copy_from(deformed if checked else original)
                plotter.render()

            label = f'Deformation ({deformation_scale:g}x)' if result_fields is not None else 'Deformation'
            controls.append((label, True, toggle_deformation))

        width, height = plotter.window_size
        for row, (name, visible, callback) in enumerate(controls):
            y = height-45-32*row
            plotter.add_checkbox_button_widget(callback, value=visible,
                position=(width/len(fields)-205, y), size=20,
                color_on='royalblue', color_off='lightgray')
            plotter.add_text(name, position=(width/len(fields)-175, y+2), font_size=8)

    if show_legend and groups:
        entries = {(label, color) for group in groups.values() for data, color, label in group if data.n_points}
        if entries:
            height = min(.028*len(entries), .16)
            width = min(.45, 240/(plotter.window_size[0]/len(fields)))
            legend = plotter.add_legend([[label, color] for label, color in sorted(entries)],
                face='line', size=(width, height))
            legend.SetPosition(.02, (.57 if result_fields is not None else .80)-height)
    if result_fields is not None:
        from IGAeasy.postprocessing.result_controls import add_result_controls
        add_result_controls(plotter, mesh, undeformed[0], result_fields, fields[0], scalar_limits)
    if show:
        plotter.show()
    return plotter
