import numpy as np
import numpy.polynomial.legendre as gq

from IGAeasy.elements import KLshell3p
from IGAeasy.trimming import trim_elements
from IGAeasy.numerical_integration import map_gp_to_element
from IGAeasy.material import linear_elastic
# from IGAeasy.constraints import apply_boundary_conditions
from IGAeasy.constraints import dirichlet, helpers
from IGAeasy.postprocessing import postprocessing
from IGAeasy.assembler import MatrixAssembler, VectorAssembler
from IGAeasy.solver import direct_solver, iterative_solver
from IGAeasy.loads import SurfaceLoad, PointLoad


def solve(RS, use_sparse=False, compute_stress=False, verbose=True,
          boundary='simply_supported', full_quad=False):
    """
    Solves a simply supported plate using Kirchhoff-Love shell formulation.
    """

    # --- Material & geometry properties ---
    t = 0.1               # Plate thickness
    E = 10920             # Young's modulus
    nue = 0.3             # Poisson's ratio
    rho = 1               # Density



    # --- Loads & integration ---
    surfaceloads = [SurfaceLoad(
        num_local_dofs=3,
        load_functions=[
            lambda u, v: 0.0,  # x-component
            lambda u, v: 0.0,  # y-component
            lambda u, v: -1.0,  # z-component
        ]
    )]

    pointloads = [
        PointLoad((0.3, 0.5), [0, 0, 0]),
        PointLoad((0.5, 0.5), [0, 0, 0])
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
    shell_element = KLshell3p.KLShell()
    dof_per_cp = shell_element.dof_per_cp

    # --- Boundary conditions ---
    if boundary != 'simply_supported':
        raise ValueError('This KL example implements simply_supported boundaries only')
    fixed_DOFs = determine_fixed_dofs(RS)

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
    Kred, fred, free_dofs = dirichlet.block_partitioning_dbc(K=K, f=f, prescribed_DOFs=fixed_DOFs)

    # --- Solve the linear system ---
    free_disp = direct_solver.solve(Kred, fred)
    #displacements = iterative_solver.solve(Kred, fred, preconditioner='Jacobi')

    displacements[free_dofs] = free_disp
    # print(Kred, fred, free_disp)

    # --- Update results ---
    postprocessing.update_cps(RS, displacements, dof_per_cp=dof_per_cp, dimensions=3)
    if compute_stress:
        postprocessing.update_stress_strain_KL(RS)

    # --- Evaluate displacement at center point (0.5, 0.5) ---
    uz_ref = RS.eval_displacement(0.5, 0.5)[2]
    if verbose:
        print('z-displacement at center (0.5, 0.5):', uz_ref)

    return RS


def determine_fixed_dofs(RS):
    """
    Determines the fixed degrees of freedom (DOFs) based on boundary conditions.
    """
    fixed_DOFs = set()

    # # ymin:
    # add_fixed_dofs(RS, fixed_DOFs, ymin=True, dofs=['x', 'y', 'z'])
    #
    # # ymax:
    # add_fixed_dofs(RS, fixed_DOFs, ymax=True, dofs=['x', 'y', 'z'])

    # xmin:
    add_fixed_dofs(RS, fixed_DOFs, xmin=True, dofs=['x', 'y', 'z'])

    # xmax:
    add_fixed_dofs(RS, fixed_DOFs, ymin=True, dofs=['x', 'y', 'z'])
    add_fixed_dofs(RS, fixed_DOFs, xmax=True, dofs=['x', 'y', 'z'])
    add_fixed_dofs(RS, fixed_DOFs, ymax=True, dofs=['x', 'y', 'z'])

    return fixed_DOFs


def add_fixed_dofs(RS, fixed_DOFs, dofs, **kwargs):
    """
    Helper function to identify fixed DOFs on an edge.
    """
    cps_on_edges = helpers.find_control_point_ids_on_edges(RS, **kwargs)
    fixed_DOFs.update(
        KLshell3p.get_dofs_from_cps(cps_on_edges, dofs)
    )
