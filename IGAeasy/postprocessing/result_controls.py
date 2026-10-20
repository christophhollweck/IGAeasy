"""Compact result selectors for the existing VTK viewer, without a Qt dependency."""
import numpy as np


def scalar_range(values, limits=None):
    """Keep nearly constant fields from turning roundoff into apparent variation."""
    if limits is not None:
        low, high = limits
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError('Scalar limits must be finite and strictly increasing')
        return low, high
    low, high = float(values.min()), float(values.max())
    if high-low <= 1e-10*max(abs(low), abs(high), 1e-12):
        width = max(abs(low)*1e-6, 1e-12)
        low, high = low-width, high+width
    return low, high


def add_result_controls(plotter, mesh, original, results, initial_field, scalar_limits=None):
    labels = {'displacement_magnitude': '|u|', 'ux': 'u_x', 'uy': 'u_y', 'uz': 'u_z',
              'stress_von_mises': 'Section von Mises', 'stress_11': 'Section stress 11 (local)',
              'stress_22': 'Section stress 22 (local)', 'stress_12': 'Section stress 12 (local)',
              'strain_11': 'Section strain 11 (local)', 'strain_22': 'Section strain 22 (local)',
              'strain_12': 'Section strain 12 (tensor shear)'}
    state = {'Field': initial_field, 'Section display': 'Cell mean', 'Section layer': 'Middle'}
    surface = plotter.renderer.actors['result_surface']
    bar = next(iter(plotter.scalar_bars.values()))
    sampled = {}

    def update():
        field = state['Field']
        key = field, state['Section layer'], state['Section display']
        if field in results.fields and key not in sampled:
            sampled[key] = results.sample(mesh, *key)
        values = sampled[key] if field in results.fields else np.asarray(mesh[field])
        if not np.isfinite(values).all():
            raise ValueError('Cannot display nonfinite result values')
        # Both geometry states carry the same physical scalar field.
        for data in (mesh, original, surface.mapper.dataset):
            data[field] = values
            data.set_active_scalars(field)
        surface.mapper.array_name = field
        surface.mapper.scalar_range = scalar_range(values, (scalar_limits or {}).get(field))
        bar.SetLookupTable(surface.mapper.lookup_table)
        bar.SetTitle(labels[field])
        plotter.render()

    # One field is active across all three rows; representations form a separate group.
    groups = {'Field': {}, 'Section display': {}}
    plotter.result_controls = groups
    height = plotter.window_size[1]

    def row(name, choices, captions, title, y):
        plotter.add_text(title, position=(15, y+27), font_size=8)
        for index, (choice, caption) in enumerate(zip(choices, captions)):
            def select(checked, choice=choice):
                if choice not in groups[name]:
                    return  # VTK calls the callback while creating the widget.
                if checked:
                    state[name] = choice
                for key, button in groups[name].items():
                    button.GetRepresentation().SetState(int(key == state[name]))
                update()

            x = 15+index*145
            groups[name][choice] = plotter.add_checkbox_button_widget(
                select, value=choice == state[name], position=(x, y), size=20)
            plotter.add_text(caption, position=(x+25, y+2), font_size=8)

    row('Field', list(labels)[:4], ['|u|', 'u_x', 'u_y', 'u_z'],
        'Displacement (global)', height-85)
    row('Field', list(labels)[4:8], ['von Mises', '11', '22', '12'],
        'Stress (local, middle surface)', height-150)
    row('Field', list(labels)[8:], ['11', '22', '12 (tensor)'],
        'Strain (local, middle surface)', height-215)
    row('Section display', list(results.representations), ['Cell mean', 'Element', 'Global L2'],
        'Stress / strain representation', height-280)
