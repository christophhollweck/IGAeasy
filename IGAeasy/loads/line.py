import numpy as np
from shapely.geometry import LineString, box, Point
from IGAeasy.constraints.helpers import compute_taylor_expansion_contribution, compute_gradient_taylor_expansion_contribution
from IGAeasy.constraints.line_helpers import get_integration_points
from IGAeasy.differential_geometry.line import *
from IGAeasy.differential_geometry.surface import *
from IGAeasy.numerical_integration import gauss_line

class LineLoad:
    def __init__(self, num_local_dofs, load_functions, line, id:int, at_patch_edge=False):
        """
        Initializes the LineLoad class with load functions, a line, and a polygon for intersection.

        :param num_local_dofs: Number of degrees of freedom per control point.
        :param load_functions: List of functions representing the line load (one for each degree of freedom).
               The loadfunction with parameter s ranges from 0 to 1.
        :param line: A LineString object representing the load line.
        """
        assert len(load_functions) == num_local_dofs, "Number of load functions must match num_local_dofs."
        self.num_local_dofs = num_local_dofs
        self.load_functions = load_functions
        self.line = line
        self.id = id
        self.at_patch_edge = at_patch_edge


    def evaluate_load(self, s):
        """
        Evaluates the load functions at a given parameter s along the line.

        :param s: The parameter along the line.
        :return: The load values for each degree of freedom at parameter s.
        """
        return np.array([f(s) for f in self.load_functions])

    # def evaluate_load(self, u, v):
    #     """
    #     Evaluates the load functions at a given parameter s along the line.
    #
    #     :param s: The parameter along the line.
    #     :return: The load values for each degree of freedom at parameter s.
    #     """
    #     return np.array([f(u, v) for f in self.load_functions])


    def integrate_contribution(self, cell, dof_per_cp, use_sbm=False):
        """
        Computes the line load contribution to the local force vector f_el.
        Supports standard and SBM-based integration.

        :param cell: The finite element cell.
        :param dof_per_cp: Number of degrees of freedom per control point.
        :param use_sbm: If True, uses SBM integration; otherwise standard.
        :return: Tuple (K_el, f_el)
        """

        supported_bfs = list(cell.supported_b_splines)
        supported_bfs.sort(key=lambda bf: bf.id)
        n_cps = len(supported_bfs)
        f_el = np.zeros(n_cps * dof_per_cp)
        K_el = np.zeros((n_cps * dof_per_cp, n_cps * dof_per_cp))
        if use_sbm:
            conductivity = getattr(cell.gausspoints[0], 'k', 1.0) if cell.gausspoints else 1.0
            cpsx = [b.cp[0] for b in supported_bfs]
            cpsy = [b.cp[1] for b in supported_bfs]

            for active_edge in cell.active_sides:
                for gp_sbm in active_edge.neumann_data:
                    if self.id == gp_sbm.closest_line.id:

                        sidenum = active_edge.side

                        u = gp_sbm.u
                        v = gp_sbm.v
                        s = gp_sbm.s
                        gw = gp_sbm.weight
                        n = gp_sbm.n
                        t = gp_sbm.t
                        line = gp_sbm.closest_line.line

                        closest_point, s_line_closest_point, difference_vector = compute_closest_point(line, Point(u, v))

                        gp_sbm.closest_point = closest_point
                        gp_sbm.s_line_closest_point = s_line_closest_point
                        gp_sbm.difference_vector = difference_vector
                        # # # FIXME (fixed on 23/06/25):
                        # n_line_closest_point, t_line_closest_point = compute_line_normal_and_tangent(line, closest_point)
                        n_line_closest_point, t_line_closest_point = compute_line_normal_and_tangent(line, Point(u, v))


                        # ##############
                        # # When projection on the brep lands on a kink/edge of the brep, the normal might deviate a bit from the difference vector
                        # # this is corrected here. This currently only works, when the flux is defined here and t_N is not taken from the input
                        # norm = np.linalg.norm(difference_vector)
                        # if norm > 1e-14:
                        #     temp = n_line_closest_point
                        #     n_line_closest_point = difference_vector / norm
                        #     if np.dot(n_line_closest_point, temp) < 0:
                        #         n_line_closest_point *= -1
                        # #############

                        gp_sbm.n_line_closest_point = n_line_closest_point
                        gp_sbm.t_line_closest_point = t_line_closest_point

                        # # # FIXME:
                        # n_line_closest_point = n
                        # difference_vector = np.array([0, 0])
                        # xx = u + difference_vector[0]
                        # yy = v + difference_vector[1]
                        # # gradu = [yy * np.sin(yy) * (np.cosh(xx) + xx * np.sinh(xx)),
                        # #          xx * np.cosh(xx) * (np.sin(yy) + yy * np.cos(yy))]
                        # gradu = [1, 1]
                        # t_N = [np.dot(gradu, n_line_closest_point)]

                        t_N = self.evaluate_load(s_line_closest_point)

                        N = np.array([bf(u, v, cell=cell) for bf in supported_bfs])
                        grad_H_sum = compute_gradient_taylor_expansion_contribution(u, v, cell, difference_vector)
                        grad_H_sum_dot_n_brep = grad_H_sum.T @ n_line_closest_point

                        dN_dxi = np.array([bf.grad(u, v, cell=cell) for bf in supported_bfs]).T
                        J = np.dot(dN_dxi, np.array([cpsx, cpsy]).T)

                        # Check if J is (approximately) identity
                        if not np.allclose(J, np.eye(J.shape[0])):
                            print("J is not identity:\n", J, (u,v))


                        dN_dx = np.linalg.solve(J, dN_dxi)

                        g1, g2 = compute_jacobian_surface(cell, u, v)
                        l1, l2 = t
                        detJ = np.linalg.norm(g1 * l1 + g2 * l2)
                        gwl = detJ * gw

                        n_n_tilde = np.dot(n_line_closest_point, n)
                        # print(cell.id, u, v, gwl, t_N)

                        for i in range(n_cps):
                            for j in range(n_cps):
                                for a in range(dof_per_cp):
                                    K_el[dof_per_cp * i + a, dof_per_cp * j + a] += (
                                        - conductivity * gwl * (dN_dx[0, j] * n[0] + dN_dx[1, j] * n[1]) * N[i]
                                        + conductivity * gwl * grad_H_sum_dot_n_brep[j] * n_n_tilde * N[i]
                                    )

                        for i in range(n_cps):
                            for a in range(dof_per_cp):
                                f_el[dof_per_cp * i + a] += gwl * t_N[a] * n_n_tilde * N[i]

        else:
            # cpsx = [b.cp[0] for b in supported_bfs]
            # cpsy = [b.cp[1] for b in supported_bfs]

            iwip = get_integration_points(self.line, cell)
            gps = []
            for s, gw in iwip:
                pt = self.line.interpolate(s)
                u, v = pt.x, pt.y

                u_len = cell.u_max - cell.u_min
                v_len = cell.v_max - cell.v_min
                on_u_edge = abs(u - cell.u_min) < 1e-12 * u_len or abs(u - cell.u_max) < 1e-12 * u_len
                on_v_edge = abs(v - cell.v_min) < 1e-12 * v_len or abs(v - cell.v_max) < 1e-12 * v_len
                if (on_u_edge or on_v_edge) and not self.at_patch_edge and not hasattr(cell, '_mesh_elements'):
                    gw *= 0.5

                load_values = self.evaluate_load(s)
                N = np.array([bf(u, v, cell=cell) for bf in supported_bfs])

                g1, g2 = compute_jacobian_surface(cell, u, v)
                n_param, t_param = compute_line_normal_and_tangent(self.line, pt)
                l1, l2 = t_param
                detJ = np.linalg.norm(g1 * l1 + g2 * l2)
                gwl = detJ * gw

                gp = gauss_line.gauss_point(u, v, s, gw, self.line)

                gp.n = n_param
                gp.t = t_param
                gps.append(gp)
                # print(cell.id,u,v,gwl,load_values)

                for i in range(n_cps):
                    for a in range(dof_per_cp):
                        f_el[dof_per_cp * i + a] += N[i] * load_values[a] * gwl

            if not hasattr(cell, "gps_line"):
                cell.gps_line = gps
            else:
                cell.gps_line.extend(gps)

        return K_el, f_el
