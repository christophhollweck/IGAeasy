"""Local section fields reconstructed from shell force and moment resultants."""
import numpy as np
from scipy.interpolate import RBFInterpolator

from IGAeasy.postprocessing.projection import project_scalar_results
from IGAeasy.postprocessing.shell_results import reconstruct_shell_stress, von_mises


class ShellResultFields:
    """Prepare GP results once; cache only the scalar representations requested."""
    fields = ('stress_von_mises', 'stress_11', 'stress_22', 'stress_12',
              'strain_11', 'strain_22', 'strain_12')
    layers = {'Bottom': -.5, 'Middle': 0., 'Top': .5}
    representations = ('Cell mean', 'Element interpolation', 'Global L2')

    def __init__(self, space):
        self.space = space
        self.ready = False
        self.cache = {}

    def _prepare(self):
        from IGAeasy.postprocessing import postprocessing
        formulation = self.space.shell_formulation
        routines = {'RM6': postprocessing.update_stress_strain_RM6dof,
                    'RM5': postprocessing.update_stress_strain_RM5dof,
                    'RM5_v2': postprocessing.update_stress_strain_RM5dof,
                    'RM5_hier': postprocessing.update_stress_strain_RM5dof,
                    'KL': postprocessing.update_stress_strain_KL}
        dof = {'RM6': 6, 'RM5': 5, 'RM5_v2': 5, 'RM5_hier': 5, 'KL': 3}[formulation]
        if any(np.asarray(getattr(b, 'dof_values', [])).shape != (dof,) for b in self.space.basis):
            raise ValueError('Shell results require complete per-control-point DOFs')
        routines[formulation](self.space)
        self.ready = True

    @staticmethod
    def section_value(gp, field, layer):
        stress = reconstruct_shell_stress(gp.N, gp.M, gp.Q, gp.t,
                                          ShellResultFields.layers[layer]*gp.t)
        if field == 'stress_von_mises':
            return von_mises(stress)
        index = {'11': 0, '22': 1, '12': 3}[field.rsplit('_', 1)[1]]
        if field.startswith('stress_'):
            return stress[index]
        # In-plane section strain from isotropic plane-stress compliance.
        # epsilon_12 is tensor shear, not engineering gamma_12.
        E, nu = gp.material_model.E, gp.material_model.nu
        strain = ((stress[0]-nu*stress[1])/E, (stress[1]-nu*stress[0])/E,
                  (1+nu)*stress[3]/E)
        return strain[{0: 0, 1: 1, 3: 2}[index]]

    def _representation(self, field, layer, representation):
        key = field, layer, representation
        if key in self.cache:
            return self.cache[key]
        if not self.ready:
            self._prepare()
        values = {}
        cells = {}
        for index, cell in enumerate(self.space.elements):
            if cell.void or not getattr(cell, 'active', True):
                continue
            samples = np.array([self.section_value(gp, field, layer) for gp in cell.gausspoints])
            values.update((id(gp), value) for gp, value in zip(cell.gausspoints, samples))
            if representation == 'Cell mean':
                weights = [gp.weight_inplane*np.linalg.norm(np.cross(*gp.surface_tangents))
                           for gp in cell.gausspoints]
                cells[index] = np.average(samples, weights=weights)
            elif representation == 'Element interpolation':
                points = np.array([(gp.xi, gp.eta) for gp in cell.gausspoints])
                points, unique = np.unique(points, axis=0, return_index=True)
                cells[index] = RBFInterpolator(points, samples[unique], kernel='linear', degree=0)
        result = (project_scalar_results(self.space, lambda gp: values[id(gp)])
                  if representation == 'Global L2' else cells)
        self.cache[key] = result
        return result

    def sample(self, mesh, field, layer='Middle', representation='Cell mean'):
        if field not in self.fields or layer not in self.layers or representation not in self.representations:
            raise ValueError('Unsupported shell field, layer or representation')
        data = self._representation(field, layer, representation)
        coordinates = mesh.field_data['parametric_coordinates']
        indices = mesh.field_data['element_indices'].ravel()
        result = np.empty(len(coordinates))
        for index in np.unique(indices):
            mask = indices == index
            if representation == 'Cell mean':
                result[mask] = data[index]
            elif representation == 'Element interpolation':
                result[mask] = data[index](coordinates[mask])
            else:
                cell = self.space.elements[index]
                basis = list(cell.supported_b_splines)
                result[mask] = [sum(b(u, v, cell=cell)*data[b.id] for b in basis)
                                for u, v in coordinates[mask]]
        return result
