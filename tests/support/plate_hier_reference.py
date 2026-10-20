import numpy as np
import numpy.polynomial.legendre as gq

from IGAeasy.elements import RMshell5p_hier
from IGAeasy.trimming import trim_elements
from IGAeasy.numerical_integration import map_gp_to_element
from IGAeasy.material import linear_elastic
# from IGAeasy.constraints import apply_boundary_conditions
from IGAeasy.constraints import dirichlet, helpers
from IGAeasy.constraints.boundary_selection import select_patch_boundary
from IGAeasy.constraints.linear_constraints import LinearConstraints, solve_linear_constraints
from IGAeasy.constraints.shell_constraints import constrain_affine_director_boundary
from IGAeasy.postprocessing import postprocessing
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.solver import direct_solver, iterative_solver
from IGAeasy.loads import SurfaceLoad, PointLoad


def solve(RS, use_sparse=False, compute_stress=True, verbose=True, boundary='clamped', full_quad=False):
    """
    Solves a plate problem using Reissner-Mindlin shell formulation.
    """

    # --- Material & geometry properties ---
    t = 0.1               # Plate thickness
    E = 10920             # Young's modulus
    nue = 0.3             # Poisson's ratio
    rho = 1               # Density

    # --- Loads & integration ---
    surfaceloads = [SurfaceLoad(
        num_local_dofs=5,
        load_functions=[
            lambda u, v: -0.0,  # x-component
            lambda u, v: 0.0,  # y-component
            lambda u, v: -1.0,  # z-component (gravity)
            lambda u, v: 0.0,  # rotation r1
            lambda u, v: 0.0,  # rotation r2
        ]
    )]

    pointloads = [
        PointLoad((0.3, 0.5), [0, 0, 0, 0, 0]),
        PointLoad((0.5, 0.5), [0, 0, 0, 0, 0])
    ]

    # Initialize material model
    material_model = linear_elastic.LinearElasticMaterial(E, nue, rho, kappa=5 / 6)

    # Trimming
    trim_elements.trim(RS)

    # --- Loads & integration ---
    map_gp_to_element.assign_gp_and_properties(
        RS,
        material_model=material_model,
        properties={"t": t, "E": E, "rho": rho},
        full_quad=full_quad
    )

    # Initialize shell element
    shell_element = RMshell5p_hier.RMShell()
    dof_per_cp = shell_element.dof_per_cp

    # --- Boundary conditions ---
    fixed_DOFs = determine_fixed_dofs(RS, boundary) if boundary != 'clamped' else None

    # --- Element-wise computations of K_el and f_el ---
    for element in RS.elements:
        if element.active:
            shell_element.element_routine(element)
            for surfaceload in surfaceloads:
                element.f_el += surfaceload.integrate_contribution(element, dof_per_cp)
            for pointload in pointloads:
                element.f_el += pointload.compute_contribution(element, dof_per_cp)

    # --- Global system assembly ---
    matrix_assembler = MatrixAssembler(dof_per_cp=dof_per_cp)
    vector_assembler = VectorAssembler(dof_per_cp=dof_per_cp)
    K = matrix_assembler.assemble(RS, 'K_el', use_sparse=use_sparse)
    f = vector_assembler.assemble(RS, 'f_el')

    displacements = np.zeros(len(f))

    # --- Apply Dirichlet boundary conditions ---
    if boundary == 'clamped':
        conditions = LinearConstraints.for_control_points(
            ('x', 'y', 'z', 'r_a1', 'r_a2'), len(RS.basis))
        ids = set().union(*(select_patch_boundary(RS, edge)
                           for edge in ('umin', 'umax', 'vmin', 'vmax')))
        conditions.prescribe(ids, ('x', 'y', 'z'))
        for edge in ('umin', 'umax', 'vmin', 'vmax'):
            for direction in ([1., 0., 0.], [0., 1., 0.]):
                conditions = constrain_affine_director_boundary(
                    RS, conditions, edge, direction, 'RM5_hier')
        displacements, _ = solve_linear_constraints(K, f, conditions)
    else:
        Kred, fred, free_dofs = dirichlet.block_partitioning_dbc(K=K, f=f, prescribed_DOFs=fixed_DOFs)
        displacements[free_dofs] = direct_solver.solve(Kred, fred)
    # print(Kred, fred, free_disp)

    # --- Update results ---
    postprocessing.update_cps(RS, displacements, dof_per_cp=dof_per_cp, dimensions=3)
    if compute_stress:
        postprocessing.update_stress_strain_RM5dof(RS)

    # --- Evaluate displacement at center point (0.5, 0.5) ---
    uz_ref = RS.eval_displacement(0.5, 0.5)[2]
    if verbose:
        print('z-displacement at center (0.5, 0.5):', uz_ref)

    return RS


def determine_fixed_dofs(RS, boundary='clamped'):
    fixed_DOFs = set()
    if boundary == 'clamped':
        raise ValueError('Clamping RM5_hier requires coupled director constraints; use solve()')
    elif boundary == 'simply_supported':
        # Hard simple support on a flat, axis-aligned plate.
        x_dofs = ['x', 'y', 'z', 'r_a2']
        y_dofs = ['x', 'y', 'z', 'r_a1']
    else:
        raise ValueError("boundary must be 'clamped' or 'simply_supported'")
    add_fixed_dofs(RS, fixed_DOFs, xmin=True, dofs=x_dofs)
    add_fixed_dofs(RS, fixed_DOFs, xmax=True, dofs=x_dofs)
    add_fixed_dofs(RS, fixed_DOFs, ymin=True, dofs=y_dofs)
    add_fixed_dofs(RS, fixed_DOFs, ymax=True, dofs=y_dofs)
    return fixed_DOFs

def add_fixed_dofs(RS, fixed_DOFs, dofs, **kwargs):
    """
    Helper function to identify fixed DOFs on an edge.
    """
    cps_on_edges = helpers.find_control_point_ids_on_edges(RS, **kwargs)
    fixed_DOFs.update(
        RMshell5p_hier.get_dofs_from_cps(cps_on_edges, dofs)
    )
