from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.geometry import get_geometry
from IGAeasy.elements import RMshell6p
from IGAeasy.material import LinearElasticMaterial
from IGAeasy.numerical_integration import assign_gp_and_properties
from IGAeasy.numerical_integration.costum_rule import gauss_lobatto
from IGAeasy.trimming import trim_elements
from IGAeasy.constraints import dirichlet
from IGAeasy.constraints.prescribed_dofs import PrescribedDofs
from IGAeasy.constraints.boundary_selection import select_patch_boundary
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.loads import SurfaceLoad
from IGAeasy.solver import direct_solver
from IGAeasy.postprocessing import postprocessing
import numpy as np
from IGAeasy.postprocessing.visualization import plot_solution

'''Scordelis-Lo roof: RM6 solution and pointwise displacement/stress plots.'''

n_elx = 8  # counts for a quarter; full roof has twice as many
n_ely = 8
p = 2
full_quad = True
deformation_scale = 10
samples_per_element = 5
show_element_boundaries = True
show_plots = True

t = .25
E = 4.32e8
nue = 0.
load = -90.

def main(show=True, n_elx=n_elx, n_ely=n_ely):
    Bspl = get_geometry.scordelis_lo_roof(n_elx, n_ely, p, full=True)
    ku, kv = Bspl.knot_vectors
    RS = HierarchicalSpace(ku, kv, p, Bspl.control_points, 1, 'h')
    shell = RMshell6p.RMShell()
    dof = shell.dof_per_cp
    material = LinearElasticMaterial(E, nue, 1, kappa=5/6)
    loads = SurfaceLoad(dof, [lambda u, v: 0., lambda u, v: 0., lambda u, v: load]
                        + [lambda u, v: 0.]*3)
    trim_elements.trim(RS)
    assign_gp_and_properties(RS, material_model=material,
                             properties={'t': t, 'E': E, 'rho': 1}, full_quad=full_quad)

    # Rigid end diaphragms: ux=uz=0; longitudinal translation remains free.
    conditions = PrescribedDofs(('x', 'y', 'z', 'rx', 'ry', 'rz'), len(RS.basis))
    for edge in ('vmin', 'vmax'):
        ids = select_patch_boundary(RS, edge)
        conditions.add(ids, ('x', 'z'))
    # Remove the one remaining rigid longitudinal translation.
    corner = select_patch_boundary(RS, 'umin') & select_patch_boundary(RS, 'vmin')
    conditions.add([min(corner)], 'y')
    fixed, prescribed_values = conditions.arrays()

    for cell in RS.elements:
        if not cell.active:
            continue
        shell.element_routine(cell, gauss_lobatto(3))
        cell.f_el += loads.integrate_contribution(cell, dof)

    K = MatrixAssembler(dof_per_cp=dof).assemble(RS, 'K_el', use_sparse=True)
    f = VectorAssembler(dof_per_cp=dof).assemble(RS, 'f_el')
    Kred, fred, free = dirichlet.block_partitioning_dbc(
        K=K, f=f, prescribed_DOFs=fixed, u_c=prescribed_values)
    displacement = np.zeros(len(f))
    displacement[fixed] = prescribed_values
    displacement[free] = direct_solver.solve(Kred, fred)
    postprocessing.update_cps(RS, displacement, dof_per_cp=dof, dimensions=3, formulation='RM6')


    print('RM6 free-edge midpoint uz:', RS.eval_displacement(1., .5)[2], flush=True)
    if show and show_plots:
        plot_solution(RS, fields=('stress_von_mises',),
                             deformation_scale=deformation_scale, samples_per_element=samples_per_element,
                             show_element_boundaries=show_element_boundaries)


if __name__ == '__main__':
    main()
