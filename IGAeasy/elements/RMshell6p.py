import numpy as np
from IGAeasy.splines.batch_evaluation import evaluate_derivatives_batch
from IGAeasy.differential_geometry import derivatives
from IGAeasy.loads import SurfaceLoad

class RMShell():

    def __init__(self):
        self.dof_per_cp = 6

    def element_routine(self, cell, gauss_t):
        """
        Perform the element-level calculations for the stiffness matrix and force vector.

        :param cell: The current cell element.
        :param gauss_t: Gauss quadrature points and weights for integration.
        :return: The element stiffness matrix (K_el), element force vector (f_el), and the list of function ids (bfnums).
        """

        n_el_cps = len(cell.supported_b_splines)
        dof_per_cp = self.dof_per_cp
        # sort bf ids according to their ids
        bfnums = np.sort([b.id for b in cell.supported_b_splines])
        bfs_loc = list(cell.supported_b_splines)
        bfs_loc.sort(key=lambda bspline: bspline.id)

        ngp_t = len(gauss_t[0])
        gp_t = gauss_t[0]
        gw_t = gauss_t[1]

        K_el = np.zeros((dof_per_cp * n_el_cps, dof_per_cp * n_el_cps))
        M_el = np.zeros((dof_per_cp * n_el_cps, dof_per_cp * n_el_cps))
        f_el = np.zeros(dof_per_cp * n_el_cps)
        # cps
        cpsx = [b.cp[0] for b in bfs_loc]
        cpsy = [b.cp[1] for b in bfs_loc]
        cpsz = [b.cp[2] for b in bfs_loc]

        # inplane loop
        points = [(gp.xi, gp.eta) for gp in cell.gausspoints]
        values = evaluate_derivatives_batch(bfs_loc, points, cell, max_order=2)
        cps = np.asarray([b.cp for b in bfs_loc])

        for q, iwip in enumerate(cell.gausspoints):
            iwip.zeta = []
            iwip.weight_outerplane = []
            iwip.Bmatrices = []
            iwip.local_basis = []
            iwip.rotationmatrices = []
            iwip.volume_weights = []
            gpx = iwip.xi
            gpy = iwip.eta
            gw = iwip.weight_inplane
            # material law defined in a local orthonormal basis tangential to the shell
            Cmat_loc = iwip.material_model.get_C_voigt_cart_3D()
            t = iwip.t
            rho = iwip.rho

            # bfs arranged in a Vector
            N_vec = values[0, 0][q]
            iwip.N_evals = N_vec.copy()
            # bfs derivatives arranged in a Matrix
            dNdxi_mat = np.vstack((values[1, 0][q], values[0, 1][q]))

            # bfs arranged in a Matrix
            N_mat = np.zeros((dof_per_cp, dof_per_cp * n_el_cps))
            I = np.identity(dof_per_cp)  # [dof_per_cp, dof_per_cp]
            for i, bf_val in enumerate(N_vec):
                N_mat[:, i * dof_per_cp:(i + 1) * dof_per_cp] = I * bf_val

            # get the covariante basis vectors a1, a2, normal n and derivatives of the normal dn/dxi and dn/deta
            # dn/dxi and dn/deta is perpendicular to n -> dotproduct == 0
            a1, a2, n, a1_xi, a2_eta, a1_eta, a2_xi, dn_dxi, dn_deta = derivatives.get_local_basis_and_variations(
                cell, gpx, gpy, geometry_derivatives=(
                    (values[1, 0][q] @ cps, values[0, 1][q] @ cps),
                    (values[2, 0][q] @ cps, values[0, 2][q] @ cps, values[1, 1][q] @ cps)))
            n1, n2, n3 = n

            # a_cov, a_con, del_a_cov, A_cov, B_cov, basis = derivatives.get_full_geometric_data(cell, gpx, gpy)
            # a1, a2, n = a_cov[1], a_cov[2], a_cov[3]
            # a1_con, a2_con = a_con[1], a_con[2]
            # dn_dxi, dn_deta = del_a_cov[3][1], del_a_cov[3][2]
            # a1_1, a2_2, a1_2, a2_1 = del_a_cov[1][1], del_a_cov[2][2], del_a_cov[1][2], del_a_cov[2][1]
            # e1, e2, e3 = basis[1], basis[2], basis[3]
            #
            # n1, n2, n3 = n

            iwip.surface_tangents = (a1, a2)
            surface_basis = np.column_stack((a1, a2))
            surface_gradients = surface_basis @ np.linalg.solve(
                surface_basis.T @ surface_basis, dNdxi_mat)
            Bdrill = self.compute_B_drill_matrix(
                n_el_cps, dof_per_cp, surface_gradients[0], surface_gradients[1],
                N_vec, dNdz=surface_gradients[2], normal=n)
            iwip.Bdrill = Bdrill

            # loop for thickness direction
            for k in range(ngp_t):
                zeta = gp_t[k]
                iwip.zeta.append(zeta)
                iwip.weight_outerplane.append(gw_t[k])

                # Calculation of the Jacobian matrix and determinants
                J, det_J2, det_J3, Jinv = self.compute_jacobian(dNdxi_mat, cpsx, cpsy, cpsz, dn_dxi, dn_deta, n, t,
                                                                zeta)
                iwip.volume_weights.append(gw * gw_t[k] * det_J3)

                # Calculation of the basis function and normal derivatives
                dNdx, dNdy, dNdz, dn1dx, dn2dx, dn3dx, dn1dy, dn2dy, dn3dy, dn1dz, dn2dz, dn3dz = \
                    self.compute_basis_functions_and_normals(dNdxi_mat, Jinv, dn_dxi, dn_deta, n_el_cps)

                # Calculation of the local basis
                dxdxi, dydxi, dzdxi = J[0, 0], J[0, 1], J[0, 2]
                dzetadx, dzetady, dzetadz = Jinv[0, 2], Jinv[1, 2], Jinv[2, 2]
                t1, t2, t3 = self.compute_local_basis(dxdxi, dydxi, dzdxi, n)

                iwip.local_basis.append([t1, t2, t3])
                # Rotation T from t1, t2, t3-> x, y, z
                T = self.get_rotationsmatrix(t1, t2, t3)
                iwip.rotationmatrices.append(T)

                # Rotate the material law into the global system x,y,z, where also B is definied
                Cmat_glob = T @ Cmat_loc @ np.transpose(T)

                # B containes the derivatives w.r.t. global coordinates x,y,z -> stress and straina also in x,y,z
                B = self.compute_B_matrix(n_el_cps, dof_per_cp, dNdx, dNdy, dNdz, N_vec, dn1dx, dn1dy, dn1dz,
                                          dn2dx, dn2dy, dn2dz, dn3dx, dn3dy, dn3dz, dzetadx, dzetady, dzetadz,
                                          n1, n2, n3, zeta, t)

                iwip.Bmatrices.append(B)
                K_el += (np.transpose(B) @ Cmat_glob @ B) * gw * gw_t[k] * det_J3

                K_el += (0.05 * iwip.material_model.E * (Bdrill.T @ Bdrill) * gw * gw_t[k] * det_J3)
                M_el += rho * (np.transpose(N_mat) @ N_mat) * gw * gw_t[k] * det_J3


        cell.M_el = M_el
        cell.K_el = K_el
        cell.f_el = f_el
        cell.bfnums = bfnums



    def get_rotationsmatrix(self, t1, t2, t3):
        # Rotationmatrix from local System (t1,t2,t3) -> (x,y,z)
        T = np.zeros((6, 6))

        T[0, 0] = t1[0] * t1[0]
        T[0, 1] = t2[0] * t2[0]
        T[0, 2] = t3[0] * t3[0]
        T[0, 3] = 2 * t1[0] * t2[0]
        T[0, 4] = 2 * t2[0] * t3[0]
        T[0, 5] = 2 * t1[0] * t3[0]

        T[1, 0] = t1[1] * t1[1]
        T[1, 1] = t2[1] * t2[1]
        T[1, 2] = t3[1] * t3[1]
        T[1, 3] = 2 * t1[1] * t2[1]
        T[1, 4] = 2 * t2[1] * t3[1]
        T[1, 5] = 2 * t1[1] * t3[1]

        T[2, 0] = t1[2] * t1[2]
        T[2, 1] = t2[2] * t2[2]
        T[2, 2] = t3[2] * t3[2]
        T[2, 3] = 2 * t1[2] * t2[2]
        T[2, 4] = 2 * t2[2] * t3[2]
        T[2, 5] = 2 * t1[2] * t3[2]

        T[3, 0] = t1[0] * t1[1]
        T[3, 1] = t2[0] * t2[1]
        T[3, 2] = t3[0] * t3[1]
        T[3, 3] = t1[0] * t2[1] + t2[0] * t1[1]
        T[3, 4] = t2[0] * t3[1] + t3[0] * t2[1]
        T[3, 5] = t1[0] * t3[1] + t3[0] * t1[1]

        T[4, 0] = t1[1] * t1[2]
        T[4, 1] = t2[1] * t2[2]
        T[4, 2] = t3[1] * t3[2]
        T[4, 3] = t1[1] * t2[2] + t2[1] * t1[2]
        T[4, 4] = t2[1] * t3[2] + t3[1] * t2[2]
        T[4, 5] = t1[1] * t3[2] + t3[1] * t1[2]

        T[5, 0] = t1[0] * t1[2]
        T[5, 1] = t2[0] * t2[2]
        T[5, 2] = t3[0] * t3[2]
        T[5, 3] = t1[0] * t2[2] + t2[0] * t1[2]
        T[5, 4] = t2[0] * t3[2] + t3[0] * t2[2]
        T[5, 5] = t1[0] * t3[2] + t3[0] * t1[2]

        return T

    import numpy as np

    def compute_B_matrix(self, n_el_cps, dof_per_cp, dNdx, dNdy, dNdz, N_vec, dn1dx, dn1dy, dn1dz,
                         dn2dx, dn2dy, dn2dz, dn3dx, dn3dy, dn3dz, dzetadx, dzetady, dzetadz,
                         n1, n2, n3, zeta, t):
        """
        Compute the B-matrix for a Reissner-Mindlin shell element.

        Parameters:
            n_el_cps (int): Number of control points per element.
            dof_per_cp (int): Degrees of freedom per control point.
            dNdx, dNdy, dNdz (array): Derivatives of shape functions.
            N_vec (array): Shape function vector.
            dn1dx, dn1dy, dn1dz, dn2dx, dn2dy, dn2dz, dn3dx, dn3dy, dn3dz (array):
                Derivatives of director components.
            dzetadx, dzetady, dzetadz (array): Derivatives of thickness coordinate function.
            n1, n2, n3 (array): Director components.
            zeta (float): Thickness coordinate.
            t (float): Thickness of the shell.

        Returns:
            np.ndarray: B-matrix of size (6, 6 * n_el_cps).
        """
        # B-Matrix
        B = np.zeros((6, 6 * n_el_cps))

        # Bv
        B[0, 0::dof_per_cp] = dNdx
        B[1, 1::dof_per_cp] = dNdy
        B[2, 2::dof_per_cp] = dNdz
        B[3, 0::dof_per_cp] = dNdy
        B[3, 1::dof_per_cp] = dNdx
        B[4, 1::dof_per_cp] = dNdz
        B[4, 2::dof_per_cp] = dNdy
        B[5, 0::dof_per_cp] = dNdz
        B[5, 2::dof_per_cp] = dNdx

        # B_omega
        B[0, 4::dof_per_cp] = (N_vec * (dn3dx * zeta + dzetadx * n3) + dNdx * n3 * zeta) * t / 2
        B[0, 5::dof_per_cp] = (-N_vec * (dn2dx * zeta + dzetadx * n2) - dNdx * n2 * zeta) * t / 2

        B[1, 3::dof_per_cp] = (-N_vec * (dn3dy * zeta + dzetady * n3) - dNdy * n3 * zeta) * t / 2
        B[1, 5::dof_per_cp] = (N_vec * (dn1dy * zeta + dzetady * n1) + dNdy * n1 * zeta) * t / 2

        B[2, 3::dof_per_cp] = (N_vec * (dn2dz * zeta + dzetadz * n2) + dNdz * n2 * zeta) * t / 2
        B[2, 4::dof_per_cp] = (-N_vec * (dn1dz * zeta + dzetadz * n1) - dNdz * n1 * zeta) * t / 2

        B[3, 3::dof_per_cp] = (-N_vec * (dn3dx * zeta + dzetadx * n3) - dNdx * n3 * zeta) * t / 2
        B[3, 4::dof_per_cp] = (N_vec * (dn3dy * zeta + dzetady * n3) + dNdy * n3 * zeta) * t / 2
        B[3, 5::dof_per_cp] = (N_vec * (
                dn1dx * zeta - dn2dy * zeta + dzetadx * n1 - dzetady * n2) + dNdx * n1 * zeta - dNdy * n2 * zeta) * t / 2

        B[4, 3::dof_per_cp] = (N_vec * (
                dn2dy * zeta - dn3dz * zeta + dzetady * n2 - dzetadz * n3) + dNdy * n2 * zeta - dNdz * n3 * zeta) * t / 2
        B[4, 4::dof_per_cp] = (-N_vec * (dn1dy * zeta + dzetady * n1) - dNdy * n1 * zeta) * t / 2
        B[4, 5::dof_per_cp] = (N_vec * (dn1dz * zeta + dzetadz * n1) + dNdz * n1 * zeta) * t / 2

        B[5, 3::dof_per_cp] = (N_vec * (dn2dx * zeta + dzetadx * n2) + dNdx * n2 * zeta) * t / 2
        B[5, 4::dof_per_cp] = (-N_vec * (
                dn1dx * zeta - dn3dz * zeta + dzetadx * n1 - dzetadz * n3) - dNdx * n1 * zeta + dNdz * n3 * zeta) * t / 2
        B[5, 5::dof_per_cp] = (-N_vec * (dn2dz * zeta + dzetadz * n2) - dNdz * n2 * zeta) * t / 2

        return B

    def compute_B_drill_matrix(self, n_el_cps, dof_per_cp, dNdx, dNdy, N_vec,
                              dNdz=None, normal=None):
        """
        Compute the drilling B-matrix for a Reissner-Mindlin shell element.

        Parameters:
            n_el_cps (int): Number of control points per element.
            dof_per_cp (int): Degrees of freedom per control point.
            dNdx, dNdy (array): Derivatives of shape functions.
            N_vec (array): Shape function vector.

        Returns:
            np.ndarray: B_drill matrix of size (1, dof_per_cp * n_el_cps).
        """
        # Normal component of surface spin minus the normal rotation DOF.
        normal = np.array([0., 0., 1.]) if normal is None else np.asarray(normal)
        dNdz = np.zeros_like(dNdx) if dNdz is None else dNdz
        gradients = np.column_stack((dNdx, dNdy, dNdz))
        spin = .5*np.cross(normal, gradients)
        Bdrill = np.zeros((1, dof_per_cp * n_el_cps))
        for component in range(3):
            Bdrill[0, component::dof_per_cp] = spin[:, component]
            Bdrill[0, (3+component)::dof_per_cp] = -N_vec*normal[component]

        return Bdrill

    def compute_jacobian(self, dNdxi_mat, cpsx, cpsy, cpsz, dn_dxi, dn_deta, n, t, zeta):
        """
        Compute the 3D Jacobian matrix for a shell element at a given point in parametric space.

        Parameters
        ----------
        dNdxi_mat : ndarray of shape (2, n_bf)
            Derivatives of shape functions w.r.t xi and eta.
        cpsx, cpsy, cpsz : ndarray of shape (n_bf,)
            Control point coordinates in x, y, z.
        dn_dxi, dn_deta : array_like of shape (3,)
            Derivatives of the shell normal vector w.r.t. xi and eta.
        n : array_like of shape (3,)
            Shell director (normal vector).
        t : float
            Thickness.
        zeta : float
            Thickness coordinate in [-1, 1].

        Returns
        -------
        J : ndarray of shape (3, 3)
            Jacobian matrix.
        det_J2 : float
            Surface Jacobian (area scaling).
        det_J3 : float
            Full Jacobian determinant (volume scaling).
        Jinv : ndarray of shape (3, 3)
            Inverse of Jacobian.
        """
        dxdxi = np.dot(dNdxi_mat[0, :], cpsx) + t / 2 * zeta * dn_dxi[0]
        dydxi = np.dot(dNdxi_mat[0, :], cpsy) + t / 2 * zeta * dn_dxi[1]
        dzdxi = np.dot(dNdxi_mat[0, :], cpsz) + t / 2 * zeta * dn_dxi[2]
        dxdeta = np.dot(dNdxi_mat[1, :], cpsx) + t / 2 * zeta * dn_deta[0]
        dydeta = np.dot(dNdxi_mat[1, :], cpsy) + t / 2 * zeta * dn_deta[1]
        dzdeta = np.dot(dNdxi_mat[1, :], cpsz) + t / 2 * zeta * dn_deta[2]
        dxdzeta = t / 2 * n[0]
        dydzeta = t / 2 * n[1]
        dzdzeta = t / 2 * n[2]

        J = np.array([[dxdxi, dydxi, dzdxi], [dxdeta, dydeta, dzdeta], [dxdzeta, dydzeta, dzdzeta]])

        g1 = np.array([dxdxi, dydxi, dzdxi])
        g2 = np.array([dxdeta, dydeta, dzdeta])
        cross_product = np.cross(g1, g2)
        det_J2 = np.linalg.norm(cross_product)

        det_J3 = np.linalg.det(J)
        Jinv = np.linalg.inv(J)

        return J, det_J2, det_J3, Jinv

    def compute_basis_functions_and_normals(self, dNdxi_mat, Jinv, dn_dxi, dn_deta, n_el_cps):
        # Basisfunktionen
        dNdxi_3mat = np.zeros((3, n_el_cps))
        dNdxi_3mat[:2, :] = dNdxi_mat
        dNdx_mat = Jinv @ dNdxi_3mat

        dn_dXi = np.vstack([dn_dxi, dn_deta, np.zeros(3)])
        dndX = Jinv @ dn_dXi

        # Vectorized derivatives
        dndx, dndy, dndz = dndX
        dn1dx, dn2dx, dn3dx = dndx
        dn1dy, dn2dy, dn3dy = dndy
        dn1dz, dn2dz, dn3dz = dndz

        dNdx = dNdx_mat[0, :]
        dNdy = dNdx_mat[1, :]
        dNdz = dNdx_mat[2, :]

        return dNdx, dNdy, dNdz, dn1dx, dn2dx, dn3dx, dn1dy, dn2dy, dn3dy, dn1dz, dn2dz, dn3dz

    def compute_local_basis(self, dxdxi, dydxi, dzdxi, n):
        t1 = np.array([dxdxi, dydxi, dzdxi])
        t1 = t1 / np.linalg.norm(t1)
        t3 = n
        t2 = np.cross(t3, t1)
        return t1, t2, t3


def get_dofs_from_cps(cps, loc_dofs, dof_per_cp=6):
    from IGAeasy.constraints.prescribed_dofs import control_point_dofs
    return control_point_dofs(cps, loc_dofs, ('x', 'y', 'z', 'rx', 'ry', 'rz'), dof_per_cp)




