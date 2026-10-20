import numpy as np
from IGAeasy.splines.batch_evaluation import evaluate_derivatives_batch
from IGAeasy.differential_geometry import derivatives
from IGAeasy.differential_geometry.surface import resultant_transformations

"""
RM5 shell with covariant director coefficients in the physical basis a^1, a^2.
"""

class RMShell:
    def __init__(self):
        self.dof_per_cp = 5

    def element_routine(self, cell, kinematics=None):
        """
        Element-level routine to compute stiffness, mass, and force contributions.

        :param cell: Cell object with geometry, material, and basis data.
        """
        kinematics = compute_B_matrix if kinematics is None else kinematics
        dof = self.dof_per_cp
        bfs = sorted(cell.supported_b_splines, key=lambda b: b.id)
        bfnums = np.array([b.id for b in bfs])
        n_bfs = len(bfs)
        ndof = dof * n_bfs

        K_el = np.zeros((ndof, ndof))
        M_el = np.zeros((ndof, ndof))
        f_el = np.zeros(ndof)

        points = [(gp.xi, gp.eta) for gp in cell.gausspoints]
        values = evaluate_derivatives_batch(bfs, points, cell, max_order=2)
        cps = np.asarray([b.cp for b in bfs])

        for q, gp in enumerate(cell.gausspoints):
            xi, eta = gp.xi, gp.eta
            w = gp.weight_inplane
            t, rho = gp.t, gp.rho

            a_cov, a_con, del_a_cov, A_cov, B_cov, basis = derivatives.get_full_geometric_data(
                cell, xi, eta, geometry_derivatives=(
                    (values[1, 0][q] @ cps, values[0, 1][q] @ cps),
                    (values[2, 0][q] @ cps, values[0, 2][q] @ cps, values[1, 1][q] @ cps)))
            a1, a2, a3 = a_cov[1], a_cov[2], a_cov[3]
            a1_con, a2_con = a_con[1], a_con[2]
            a3_1, a3_2 = del_a_cov[3][1], del_a_cov[3][2]
            a1_1, a2_2 = del_a_cov[1][1], del_a_cov[2][2]
            a1_2, a2_1 = del_a_cov[1][2], del_a_cov[2][1]
            e1, e2, e3 = basis[1], basis[2], basis[3]

            A_con = np.linalg.inv(A_cov)
            J_surf = np.linalg.norm(np.cross(a1, a2))

            gp.N_evals = values[0, 0][q].copy()
            gp.surface_tangents = (a1, a2)
            gp.local_basis = [[e1, e2, e3]]

            Cm, Cb, Ds = gp.material_model.compute_C_tensors_curv_2D(A_con, t)
            Dm = gp.material_model.tensor4_to_voigt2D(Cm)
            Db = gp.material_model.tensor4_to_voigt2D(Cb)


            Bm, Bb, Bs = np.zeros((3, ndof)), np.zeros((3, ndof)), np.zeros((2, ndof))

            for i, bf in enumerate(bfs):
                Ni = values[0, 0][q, i]
                Ni_1, Ni_2 = values[1, 0][q, i], values[0, 1][q, i]

                Bm_i, Bb_i, Bs_i = kinematics(
                    Ni, Ni_1, Ni_2,
                    a1, a2, a3,
                    a1_con, a2_con,
                    a1_1, a2_2, a1_2, a2_1,
                    a3_1, a3_2
                )

                idx = slice(i * dof, (i + 1) * dof)
                Bm[:, idx] = Bm_i
                Bb[:, idx] = Bb_i
                Bs[:, idx] = Bs_i


            gp.Bmatrices = [(Bm, Bb, Bs)]
            gp.D = (Dm, Db, Ds)
            gp.rotationmatrices = [resultant_transformations(a1, a2, e1, e2)]
            dA = w * J_surf

            K_el += (Bm.T @ Dm @ Bm +
                     Bb.T @ Db @ Bb +
                     Bs.T @ Ds @ Bs) * dA

        # Mass assembly is not implemented for this formulation.
        cell.K_el = K_el
        cell.M_el = M_el
        cell.f_el = f_el
        cell.bfnums = bfnums

def compute_B_matrix(Ni, Ni_1, Ni_2,
                     a1, a2, a3,
                     a1_con, a2_con,
                     a1_1, a2_2, a1_2, a2_1,
                     a3_1, a3_2):
    """
    Compute membrane, bending and shear B-matrices for Reissner–Mindlin shell.

    :param Ni: Basis function value at GP.
    :param Ni_1: ∂Ni/∂ξ.
    :param Ni_2: ∂Ni/∂η.
    :param a1, a2, a3: Covariant base vectors and surface normal.
    :param a1_con, a2_con: Contravariant base vectors.
    :param a1_1, a2_2, a1_2, a2_1: Derivatives of a1 and a2.
    :param a3_1, a3_2: Derivatives of normal vector a3.
    :return: Bm, Bb, Bs (3x5, 3x5, 2x5).
    """

    # Membrane B-matrix
    Bm = np.zeros((3, 5))
    Bm[0, 0:3] = Ni_1 * a1
    Bm[1, 0:3] = Ni_2 * a2
    Bm[2, 0:3] = Ni_2 * a1 + Ni_1 * a2


    # Bending B-matrix
    Bb = np.zeros((3, 5))
    Bb[0, 0:3] = Ni_1 * a3_1
    Bb[0, 3] = Ni_1 - np.dot(a1_1, a1_con) * Ni
    Bb[0, 4] = -np.dot(a1_1, a2_con) * Ni

    Bb[1, 0:3] = Ni_2 * a3_2
    Bb[1, 3] = -np.dot(a2_2, a1_con) * Ni
    Bb[1, 4] = Ni_2 - np.dot(a2_2, a2_con) * Ni

    Bb[2, 0:3] = Ni_1 * a3_2 + Ni_2 * a3_1
    Bb[2, 3] = Ni_2 - (np.dot(a1_2, a1_con) + np.dot(a2_1, a1_con)) * Ni
    Bb[2, 4] = Ni_1 - (np.dot(a1_2, a2_con) + np.dot(a2_1, a2_con)) * Ni

    # Shear B-matrix
    Bs = np.zeros((2, 5))
    Bs[0, 0:3] = Ni_1 * a3
    Bs[0, 3] = Ni
    Bs[1, 0:3] = Ni_2 * a3
    Bs[1, 4] = Ni

    return Bm, Bb, Bs


def get_dofs_from_cps(cps, loc_dofs, dof_per_cp=5):
    from IGAeasy.constraints.prescribed_dofs import control_point_dofs
    return control_point_dofs(cps, loc_dofs, ('x', 'y', 'z', 'r_a1', 'r_a2'), dof_per_cp)
