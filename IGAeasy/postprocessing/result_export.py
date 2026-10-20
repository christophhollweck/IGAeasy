"""Export computed integration-point results without spatial averaging."""
from pathlib import Path
import numpy as np
import pyvista as pv


def export_shell_results(space, filename, fields=('stress_von_mises',),
                         layer='Middle', samples_per_element=4):
    """Export reference surfaces with displacement vectors and separate result blocks."""
    from IGAeasy.postprocessing.visualization import build_surface_data
    from IGAeasy.postprocessing.shell_fields import ShellResultFields

    path = Path(filename)
    if path.suffix.lower() != '.vtm':
        raise ValueError('Shell surface results require a .vtm filename')
    results = ShellResultFields(space)
    fields = tuple(fields)
    if not fields or any(field not in results.fields for field in fields):
        raise ValueError(f'Select section fields from {results.fields}')
    if layer not in results.layers:
        raise ValueError(f'Select a layer from {tuple(results.layers)}')

    mesh, _ = build_surface_data(space, deformation_scale=0.,
        samples_per_element=samples_per_element, fields=('ux', 'uy', 'uz'))
    mesh['displacement'] = np.column_stack([mesh[name] for name in ('ux', 'uy', 'uz')])
    mesh.field_data['section_layer'] = [layer]
    blocks = pv.MultiBlock()
    for representation in results.representations:
        surface = mesh.copy()
        for field in fields:
            surface[field] = results.sample(mesh, field, layer, representation)
        blocks[representation] = surface
    blocks.save(path)
    return blocks


def export_integration_point_results(space, filename, quantities=('N', 'M', 'vM')):
    """Write a VTP point cloud; N/M/Q components use each point's local frame."""
    supported = {'N': 3, 'M': 3, 'Q': 2, 'vM': 1}
    quantities = tuple(quantities)
    if not quantities or any(key not in supported for key in quantities):
        raise ValueError('Select one or more of N, M, Q, vM')
    path = Path(filename)
    if path.suffix.lower() != '.vtp':
        raise ValueError('Integration-point results require a .vtp filename')
    points, frames1, frames2 = [], [], []
    data = {key: [] for key in quantities}
    for cell in space.elements:
        if getattr(cell, 'void', False) or not getattr(cell, 'active', True):
            continue
        for gp in cell.gausspoints:
            point = sum(np.asarray(b.cp)*b(gp.xi, gp.eta, cell=cell)
                        for b in cell.supported_b_splines)
            points.append(point)
            e1, e2, _ = gp.local_basis[0]
            frames1.append(e1)
            frames2.append(e2)
            for key in quantities:
                value = getattr(gp, 'sigma_vM' if key == 'vM' else key, None)
                if value is None:
                    raise ValueError(f'Missing {key}; compute results or omit this quantity')
                array = np.asarray(value, dtype=float)
                if array.size != supported[key] or not np.all(np.isfinite(array)):
                    raise ValueError(f'Invalid {key} integration-point result')
                data[key].append(array.reshape(-1) if key != 'vM' else array.item())
    if not points:
        raise ValueError('No active integration points to export')
    cloud = pv.PolyData(np.asarray(points))
    cloud.point_data['local_e1'] = np.asarray(frames1)
    cloud.point_data['local_e2'] = np.asarray(frames2)
    for key, values in data.items():
        cloud.point_data['stress_von_mises' if key == 'vM' else key] = np.asarray(values)
    cloud.save(path)
    return cloud
