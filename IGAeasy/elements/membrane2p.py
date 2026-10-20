from scipy.linalg import eigh
import numpy as np
from IGAeasy.loads import SurfaceLoad


class FE_bf():
    def __init__(self):
        self.funs = [self.N1, self.N2, self.N3, self.N4]
        self.derivs = [self.dN1, self.dN2, self.dN3, self.dN4]


    def N1(self, xi, eta):
        return 1/4 * (1 - xi) * (1 - eta)
    def N2(self, xi, eta):
        return 1 / 4 * (1 + xi) * (1 - eta)
    def N3(self, xi, eta):
        return 1 / 4 * (1 - xi) * (1 + eta)
    def N4(self, xi, eta):
        return 1 / 4 * (1 + xi) * (1 + eta)


    def dN1(self, xi, eta):
        return [-1/4*(1 - eta), -1/4*(1 - xi)]
    def dN2(self, xi, eta):
        return [1/4*(1 - eta), -1/4*(1 + xi)]
    def dN3(self, xi, eta):
        return [-1/4*(1 + eta), 1/4*(1 - xi)]
    def dN4(self, xi, eta):
        return [1/4*(1 + eta), 1/4*(1 + xi)]

def build_K_M_fem(HS, t, E, rho, nue):
    # find element area in physical space
    area_minel = 10e-6
    num_minel = -1
    P_minel = []
    for i, el in enumerate(HS.elements):
        AABB = [[el.u_min, el.u_max], [el.v_min, el.v_max]]
        P = []
        for v in AABB[1]:
            for u in AABB[0]:
                P.append(HS(u, v)[0:2])
        dx = P[1][0] - P[0][0]
        dy = P[2][1] - P[0][1]
        area = abs(dx * dy)
        if area > area_minel:
            area_minel = area
            num_minel = i
            P_minel = P

    FE_bfs = FE_bf()
    Cmat = E / (1 - nue ** 2) * np.array([[1, nue, 0], [nue, 1, 0], [0, 0, 0.5 * (1 - nue)]])
    n_el_cps = 4
    bfnums = [0,1,2,3]
    funs = FE_bfs.funs
    derivs = FE_bfs.derivs
    M_el = np.zeros((2 * n_el_cps, 2 * n_el_cps))
    K_el = np.zeros((2 * n_el_cps, 2 * n_el_cps))
    cpsx = [p[0] for p in P_minel]
    cpsy = [p[1] for p in P_minel]
    iwip_el = [[-1/np.sqrt(3), -1/np.sqrt(3), 1],[1/np.sqrt(3), -1/np.sqrt(3), 1],[-1/np.sqrt(3), 1/np.sqrt(3), 1],[1/np.sqrt(3), 1/np.sqrt(3), 1]]
    for iwip in iwip_el:
        gpx = iwip[0]
        gpy = iwip[1]
        gw = iwip[2]
        B = np.zeros((3, 2 * n_el_cps))
        N_mat = np.zeros((2, 2 * n_el_cps))
        dNdxi_mat = np.zeros((2, n_el_cps))
        # bfnum = global bf indices, loc_bfnum = local bf indices starting from 0
        for bfnum, loc_bfnum in zip(bfnums, range(n_el_cps)):
            dNdxi_mat[:, loc_bfnum] = derivs[bfnum](gpx, gpy)
            N_mat[:, 2 * loc_bfnum: 2 * loc_bfnum + 2] = np.identity(2) * \
                                                         funs[bfnum](gpx, gpy)

        dxdxi = np.dot(dNdxi_mat[0, :], cpsx)
        dydxi = np.dot(dNdxi_mat[0, :], cpsy)
        dxdeta = np.dot(dNdxi_mat[1, :], cpsx)
        dydeta = np.dot(dNdxi_mat[1, :], cpsy)

        J = np.array([[dxdxi, dydxi], [dxdeta, dydeta]])
        # print(J)

        dNdx_mat = np.zeros((2, n_el_cps))
        for loc_bfnum in range(len(bfnums)):
            dNdx_mat[:, loc_bfnum] = np.linalg.solve(J, dNdxi_mat[:, loc_bfnum])

        B[0, ::2] = dNdx_mat[0, :]
        B[1, 1::2] = dNdx_mat[1, :]
        B[2, ::2] = dNdx_mat[1, :]
        B[2, 1::2] = dNdx_mat[0, :]

        # print(dNdx_mat, B)

        det_J = np.linalg.det(J)
        K_el = K_el + t * (np.transpose(B) @ Cmat @ B * gw * det_J)
        M_el = M_el + rho * t * (np.transpose(N_mat) @ N_mat * gw * det_J)


    M_el = np.diag(np.sum(M_el, axis=1))
    eigvalues, _ = eigh(K_el, M_el)
    omega = np.sqrt(np.abs(eigvalues))
    # print([om / (2 * np.pi) for om in omega])
    # print(len(omega))

    return np.max(omega)


class Membrane:
    def __init__(self):
        self.dof_per_cp = 2

    def element_routine(self, cell):

        """
        Perform the element-level calculations for the stiffness and mass matrix.

        :param cell: The current cell element.
        :param gauss_t: Gauss quadrature points and weights for integration.
        :return: The element stiffness matrix (K_el) and mass matrix (M_el), and the list of function ids (bfnums).
        """

        n_el_cps = len(cell.supported_b_splines)
        dof_per_cp = self.dof_per_cp
        # sort bf ids according to their ids
        bfnums = np.sort([b.id for b in cell.supported_b_splines])
        bfs_loc = list(cell.supported_b_splines)
        bfs_loc.sort(key=lambda bspline: bspline.id)
        n_bfs = len(bfs_loc)

        K_el = np.zeros((dof_per_cp * n_el_cps, dof_per_cp * n_el_cps))
        f_el = np.zeros(dof_per_cp * n_el_cps)
        M_el = np.zeros((dof_per_cp * n_el_cps, dof_per_cp * n_el_cps))

        # cps
        cpsx = [b.cp[0] for b in bfs_loc]
        cpsy = [b.cp[1] for b in bfs_loc]

        from IGAeasy.splines.batch_evaluation import evaluate_basis_batch
        batch_values, batch_gradients = evaluate_basis_batch(
            bfs_loc, [(gp.xi, gp.eta) for gp in cell.gausspoints], cell)
        for gp_index, iwip in enumerate(cell.gausspoints):
            gpx = iwip.xi
            gpy = iwip.eta
            gw = iwip.weight_inplane
            Cmat = iwip.material_model.get_C_voigt_cart_2D()
            t = iwip.t
            rho = iwip.rho

            B = np.zeros((3, 2 * n_el_cps))
            N_mat = np.zeros((2, 2 * n_el_cps))
            dNdxi_mat = np.zeros((2, n_el_cps))
            N_evals = np.zeros(n_bfs)

            # bfnum = global bf indices, loc_bfnum = local bf indices starting from 0
            for bf_loc, loc_bfnum in zip(bfs_loc, range(n_el_cps)):
                dNdxi_mat[:, loc_bfnum] = batch_gradients[gp_index, loc_bfnum]
                N_i = batch_values[gp_index, loc_bfnum]
                N_evals[loc_bfnum] = N_i
                N_mat[:, 2 * loc_bfnum: 2 * loc_bfnum + 2] = np.identity(2) * N_i


            dxdxi = np.dot(dNdxi_mat[0, :], cpsx)
            dydxi = np.dot(dNdxi_mat[0, :], cpsy)
            dxdeta = np.dot(dNdxi_mat[1, :], cpsx)
            dydeta = np.dot(dNdxi_mat[1, :], cpsy)

            J = np.array([[dxdxi, dydxi], [dxdeta, dydeta]])
            g1 = np.array([dxdxi, dydxi, 0])
            g2 = np.array([dxdeta, dydeta, 0])

            cross = np.cross(g1, g2)
            norm_cross = np.linalg.norm(cross)
            e3 = cross / norm_cross
            e1 = g1 - np.dot(g1, e3) * e3  # project a1 onto tangent plane
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(e3, e1)

            iwip.surface_tangents = (g1, g2)
            iwip.local_basis = [[e1, e2, e3]]

            dNdx_mat = np.linalg.solve(J, dNdxi_mat)

            B[0, ::2] = dNdx_mat[0, :]
            B[1, 1::2] = dNdx_mat[1, :]
            B[2, ::2] = dNdx_mat[1, :]
            B[2, 1::2] = dNdx_mat[0, :]

            det_J = np.linalg.det(J)

            K_el = K_el + t * (np.transpose(B) @ Cmat @ B * gw * det_J)
            M_el = M_el + rho * t * (np.transpose(N_mat) @ N_mat * gw * det_J)

            iwip.Bmatrices = [B]
            iwip.N_evals = N_evals

        cell.K_el = K_el
        cell.M_el = M_el
        cell.f_el = f_el
        cell.bfnums = bfnums


    def get_max_omega_el(self, RS, lumping=True):
        self.omega_el_dic = {}
        omegas_max = []
        for el in RS.elements:
            el.omega_el_max = 0.0
            if lumping and len(el.gausspoints) > 0:
                M_el = el.M_el
                K_el = el.K_el
                M_el = np.diag(np.sum(M_el, axis=1))
                eigvalues, _ = eigh(K_el, M_el)
                max_ev = np.max(np.sqrt(np.abs(eigvalues)))
                el.omega_el_max = max_ev
                self.omega_el_dic[el.id] = max_ev
                omegas_max.append(max_ev)
        return np.max(omegas_max)

    def plot_max_omega_el(self, RS, uniform_parameters, lumping=True):
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        import numpy as np

        self.get_max_omega_el(RS, lumping=lumping)

        t, E, rho, nue = uniform_parameters
        # fe_omega = build_K_M_fem(RS, t, E, rho, nue)
        fe_omega = 1
        max_omega = max(self.omega_el_dic.values())
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.axis('off')

        cells = RS.elements

        # Normalize the omega values to the range [0, 1]
        # norm = Normalize(vmin=0.5, vmax=4)
        norm = Normalize(vmin=20, vmax=70)
        cmap = plt.get_cmap('Purples')

        for cell in cells:
            if len(cell.gausspoints) > 0:
                omega = self.omega_el_dic[cell.id]
                omegas_ratio = omega / fe_omega
                omega_norm = norm(omega / fe_omega)
                #print(omega_norm)
                AABB = [[cell.u_min, cell.u_max], [cell.v_min, cell.v_max]]
                cellAABB = np.array(AABB)
                x = cellAABB[0, [0, 1, 1, 0, 0]]
                y = cellAABB[1, [0, 0, 1, 1, 0]]

                # Map the normalized omega to a color
                color = cmap(omega_norm)
                ax.fill(x, y, color=color)
                ax.plot(x, y, color='black', linewidth=0.5)

                plt.text(np.mean(cellAABB[0]), np.mean(cellAABB[1]), round(omegas_ratio, 2),
                         color='black', ha='center', va='center', fontsize=10)

        # print('Max El Omega:', np.max(omegas_max))
        for Brep in RS.BReps:
            x, y = Brep.polygon.exterior.xy
            ax.plot(x, y, linewidth=2, color='blue')

        # Create the colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        # cbar.set_label('Maximale Eigenfrequenz')

        #import tikzplotlib
        #plt.show()
