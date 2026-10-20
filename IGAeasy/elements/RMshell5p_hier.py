import numpy as np
from IGAeasy.differential_geometry import derivatives
from IGAeasy.differential_geometry.surface import resultant_transformations
from IGAeasy.splines.batch_evaluation import evaluate_derivatives_batch

"B-Matrix; stresses, strains and material in curvilinear basis"

class RMShell:
    def __init__(self):
        self.dof_per_cp = 5

    def element_routine(self, cell):
        """
        Perform element-level stiffness and force computations.

        :param cell: Cell element containing geometry, material, and basis info.
        :return: Updates cell.K_el, cell.f_el, cell.M_el, and cell.bfnums.
        """
        dof = self.dof_per_cp
        bfs = sorted(cell.supported_b_splines, key=lambda b: b.id)
        bfnums = np.sort([b.id for b in bfs])

        n_bfs = len(bfs)
        n_dof = dof * n_bfs
        K_el = np.zeros((n_dof, n_dof))
        M_el = np.zeros((n_dof, n_dof))
        f_el = np.zeros(n_dof)

        data = evaluate_derivatives_batch(bfs, [(gp.xi, gp.eta) for gp in cell.gausspoints], cell, max_order=2)
        cps = np.asarray([b.cp for b in bfs])
        for q, gp in enumerate(cell.gausspoints):
            xi, eta = gp.xi, gp.eta
            w = gp.weight_inplane
            t = gp.t
            rho = gp.rho

            tangents = np.array([data[1, 0][q]@cps, data[0, 1][q]@cps])
            second = np.array([data[2, 0][q]@cps, data[0, 2][q]@cps, data[1, 1][q]@cps])
            a_cov, a_con, del_a_cov, A_cov, B_cov, basis = derivatives.get_full_geometric_data(
                cell, xi, eta, geometry_derivatives=(tangents, second))
            a1, a2, a3 = a_cov[1], a_cov[2], a_cov[3]
            a1_con, a2_con = a_con[1], a_con[2]
            a3_1, a3_2 = del_a_cov[3][1], del_a_cov[3][2]
            a1_1, a2_2, a1_2, a2_1 = del_a_cov[1][1], del_a_cov[2][2], del_a_cov[1][2], del_a_cov[2][1]
            e1, e2, e3 = basis[1], basis[2], basis[3]

            A_con = np.linalg.inv(A_cov)
            J_surf = np.linalg.norm(np.cross(a1, a2))

            gp.surface_tangents = (a1, a2)
            gp.local_basis = [[e1, e2, e3]]

            Dm, Db, Ds = gp.material_model.compute_C_voigt_curv_2D(A_con, t)

            # Cm, Cb, Ds = gp.material_model.compute_C_tensors_curv_2D(A_con, t)
            # Dm = gp.material_model.tensor4_to_voigt2D(Cm)
            # Db = gp.material_model.tensor4_to_voigt2D(Cb)

            Bm_full = np.zeros((3, dof * n_bfs))
            Bb_full = np.zeros((3, dof * n_bfs))
            Bs_full = np.zeros((2, dof * n_bfs))
            N_evals = np.zeros(n_bfs)

            for i in range(n_bfs):
                Ni = data[0, 0][q, i]
                N_evals[i] = Ni
                Ni_1, Ni_2 = data[1, 0][q, i], data[0, 1][q, i]
                Ni_11, Ni_22 = data[2, 0][q, i], data[0, 2][q, i]
                Ni_12 = data[1, 1][q, i]

                Bm_i = np.zeros((3, 5))
                Bm_i[0, 0:3] = Ni_1 * a1
                Bm_i[1, 0:3] = Ni_2 * a2
                Bm_i[2, 0:3] = Ni_2 * a1 + Ni_1 * a2

                Bb_i = np.zeros((3, 5))
                Bb_i[0, 0:3] = -a3 * Ni_11 + (a3 * Ni_1) * np.dot(a1_1, a1_con) + (a3 * Ni_2) * np.dot(a1_1, a2_con)
                Bb_i[1, 0:3] = -a3 * Ni_22 + (a3 * Ni_1) * np.dot(a2_2, a1_con) + (a3 * Ni_2) * np.dot(a2_2, a2_con)
                Bb_i[2, 0:3] = - 2 * a3 * Ni_12 + 2 * (a3 * Ni_1) * np.dot(a1_2, a1_con) + 2 * (a3 * Ni_2) * np.dot(a1_2,
                                                                                                                  a2_con)
                # hierarchical components
                Bb_i[0, 3] = (Ni_1 * np.dot(a1, a1) + Ni * np.dot(a1_1, a1))
                Bb_i[0, 4] = (Ni_1 * np.dot(a2, a1) + Ni * np.dot(a2_1, a1))

                Bb_i[1, 3] = (Ni_2 * np.dot(a1, a2) + Ni * np.dot(a1_2, a2))
                Bb_i[1, 4] = (Ni_2 * np.dot(a2, a2) + Ni * np.dot(a2_2, a2))

                Bb_i[2, 3] = ((Ni_1 * np.dot(a1, a2) + Ni * np.dot(a1_1, a2)) +
                              (Ni_2 * np.dot(a1, a1) + Ni * np.dot(a1_2, a1)))

                Bb_i[2, 4] = ((Ni_1 * np.dot(a2, a2) + Ni * np.dot(a2_1, a2)) +
                              (Ni_2 * np.dot(a2, a1) + Ni * np.dot(a2_2, a1)))

                # Shear B-matrix
                Bs_i = np.zeros((2, 5))
                Bs_i[0, 3] = Ni * np.dot(a1, a1)
                Bs_i[0, 4] = Ni * np.dot(a2, a1)
                Bs_i[1, 3] = Ni * np.dot(a1, a2)
                Bs_i[1, 4] = Ni * np.dot(a2, a2)

                Bm_full[:, i * dof:(i + 1) * dof] = Bm_i
                Bb_full[:, i * dof:(i + 1) * dof] = Bb_i
                Bs_full[:, i * dof:(i + 1) * dof] = Bs_i

            gp.Bmatrices = [(Bm_full, Bb_full, Bs_full)]
            gp.rotationmatrices = [resultant_transformations(a1, a2, e1, e2)]
            gp.D = (Dm, Db, Ds)
            gp.N_evals = N_evals

            K_el += (Bm_full.T @ Dm @ Bm_full +
                     Bb_full.T @ Db @ Bb_full +
                     Bs_full.T @ Ds @ Bs_full) * w * J_surf

        cell.K_el = K_el
        cell.f_el = f_el
        cell.bfnums = bfnums


def get_dofs_from_cps(cps, loc_dofs, dof_per_cp=5):
    from IGAeasy.constraints.prescribed_dofs import control_point_dofs
    return control_point_dofs(cps, loc_dofs, ('x', 'y', 'z', 'r_a1', 'r_a2'), dof_per_cp)


