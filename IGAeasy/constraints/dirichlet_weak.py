import numpy as np
from shapely.geometry import LineString, box, Point
from IGAeasy.constraints.helpers import compute_taylor_expansion_contribution
from IGAeasy.constraints.line_helpers import get_integration_points
from IGAeasy.differential_geometry.line import *
from IGAeasy.differential_geometry.surface import *
from IGAeasy.numerical_integration import gauss_line


class WeakLineConstraint:
    def __init__(self, num_local_dofs, prescribed_field_functions, penalty_factors, line, id:int):
        """
        Initializes the LineLoad class with load functions, a line, and a polygon for intersection.

        :param num_local_dofs: Number of degrees of freedom per control point.
        :param prescribed_field_functions: List of functions representing the prescribed field (one for each degree of freedom).
               The function with parameter s ranges from 0 to 1.
        :param penalty_factors: ...add description
        :param line: A LineString object representing the load line.
        """
        assert len(prescribed_field_functions) == num_local_dofs, "Number of load functions must match num_local_dofs."
        self.num_local_dofs = num_local_dofs
        self.prescribed_field_functions = prescribed_field_functions
        self.penalty_factors = penalty_factors
        self.line = line
        self.id = id


    def evaluate_prescribed_field_functions(self, s):
        """
        Evaluates the load functions at a given parameter s along the line.

        :param s: The parameter along the line.
        :return: The load values for each degree of freedom at parameter s.
        """
        return np.array([f(s) for f in self.prescribed_field_functions])


    def integrate_contribution(self, cell, dof_per_cp, use_nitsche=True, use_sbm=False):
        """
        Computes the weak Dirichlet contribution (either Nitsche or Penalty) to the local stiffness matrix and force vector.

        :param cell: The finite element cell.
        :param dof_per_cp: Number of degrees of freedom per control point.
        :param use_nitsche: If True, use Nitsche's method; if False, use penalty method.
        :param use_sbm: If True, use SBM-based integration; otherwise standard.
        :return: Tuple (K_el, f_el)
        """

        supported_bfs = list(cell.supported_b_splines)
        supported_bfs.sort(key=lambda bf: bf.id)
        n_cps = len(supported_bfs)

        f_el = np.zeros(n_cps * dof_per_cp)
        K_el = np.zeros((n_cps * dof_per_cp, n_cps * dof_per_cp))

        cpsx = [b.cp[0] for b in supported_bfs]
        cpsy = [b.cp[1] for b in supported_bfs]

        p = cell.p
        if cell.trimmed:
            h = np.sqrt(np.sum([poly.area for poly in cell.material_polygons]))
        else:
            # h = min([cell.u_max - cell.u_min, cell.v_max - cell.v_min])
            h = np.sqrt((cell.u_max - cell.u_min) * (cell.v_max - cell.v_min))

        if use_sbm:
            conductivity = getattr(cell.gausspoints[0], 'k', 1.0) if cell.gausspoints else 1.0
            for active_edge in cell.active_sides:
                for gp_sbm in active_edge.dirichlet_data:
                    if self.id != gp_sbm.closest_line.id:
                        continue

                    u = gp_sbm.u
                    v = gp_sbm.v
                    s = gp_sbm.s
                    gw = gp_sbm.weight
                    n = gp_sbm.n
                    t = gp_sbm.t

                    sidenum = active_edge.side

                    closest_point, s_line_closest_point, difference_vector = compute_closest_point(gp_sbm.closest_line.line, Point(u,v))
                    u_D = self.evaluate_prescribed_field_functions(s_line_closest_point)

                    gp_sbm.closest_point = closest_point
                    gp_sbm.s_line_closest_point = s_line_closest_point
                    gp_sbm.difference_vector = difference_vector
                    n_line_closest_point, t_line_closest_point = compute_line_normal_and_tangent(gp_sbm.closest_line.line, closest_point)
                    gp_sbm.n_line_closest_point = n_line_closest_point
                    gp_sbm.t_line_closest_point = t_line_closest_point

                    # # Fake body-fitted
                    # xx = u
                    # yy = v
                    # difference_vector = np.array([0, 0])
                    # dxx = difference_vector[0]
                    # dyy = difference_vector[1]
                    # # u_D = [np.cosh(xx + dxx) * np.sin(yy + dyy) * (xx+dxx) * (yy+dyy)]
                    # u_D = [(xx + dxx) + (yy + dyy) + 1]

                    g1, g2 = compute_jacobian_surface(cell, u, v)
                    l1, l2 = t
                    detJ = np.linalg.norm(g1*l1 + g2*l2)

                    N = np.array([bf(u, v, cell=cell) for bf in supported_bfs])  # shape: (n_cps,)
                    H_sum = compute_taylor_expansion_contribution(u, v, cell, difference_vector)  # shape: (n_cps,)

                    dN_dxi = np.array([bf.grad(u, v, cell=cell) for bf in supported_bfs]).T  # shape: (2, n_cps)
                    J = np.dot(dN_dxi, np.array([cpsx, cpsy]).T)

                    # Check if J is (approximately) identity
                    if not np.allclose(J, np.eye(J.shape[0])):
                    # if np.allclose((1.0, 0.106943), (u, v), atol=1e-4):
                        print("J is not identity:\n", J, (u,v))
                        # print(supported_bfs)

                    dN_dx = np.linalg.solve(J, dN_dxi)

                    penalty_near_zero = self.penalty_factors and all(
                        np.isclose(pfac, 0.0, atol=1e-8) for pfac in self.penalty_factors)
                    theta = -1 if penalty_near_zero else 1
                    if penalty_near_zero and not use_nitsche:
                        print("Warning: penalty factors are near zero but Nitsche is disabled.")

                    gwl = gw*detJ

                    for i in range(n_cps):
                        for j in range(n_cps):
                            for a in range(dof_per_cp):
                                K_el[dof_per_cp * i + a, dof_per_cp * j + a] += (
                                        + conductivity * p ** 2 / h * self.penalty_factors[a] * N[i] * H_sum[j] * gwl
                                        + (- conductivity * gwl * np.dot(dN_dx[:, j], n) * N[i]
                                           - conductivity * gwl * theta * np.dot(dN_dx[:, i], n) * H_sum[j]
                                           ) * use_nitsche
                                )

                    for i in range(n_cps):
                        for a in range(dof_per_cp):
                            f_el[dof_per_cp * i + a] += (
                                    + conductivity * p ** 2 / h * self.penalty_factors[a] * N[i] * u_D[a] * gwl
                                    - use_nitsche * conductivity * gwl * theta * np.dot(dN_dx[:, i], n) * u_D[a]
                            )

        else:
            # TODO: check and validate for IBRA
            conductivity = getattr(cell.gausspoints[0], 'k', 1.0) if cell.gausspoints else 1.0
            iwip = get_integration_points(self.line, cell)
            gps = []
            for xi, gw in iwip:
                pt = self.line.interpolate(xi)
                u, v = pt.x, pt.y
                u_D = self.evaluate_prescribed_field_functions(xi)
                N = np.array([bf(u, v, cell=cell) for bf in supported_bfs])

                g1, g2 = compute_jacobian_surface(cell, u, v)
                n_param, t_param = compute_line_normal_and_tangent(self.line, pt)
                l1, l2 = t_param
                detJ = np.linalg.norm(g1 * l1 + g2 * l2)
                gwl = detJ * gw

                gp = gauss_line.gauss_point(u, v, xi, gw, self.line)

                # if use_nitsche:
                dN_dxi = np.array([bf.grad(u, v, cell=cell) for bf in supported_bfs]).T
                J = np.dot(dN_dxi, np.array([cpsx, cpsy]).T)
                dN_dx = np.linalg.solve(J, dN_dxi)

                n_param, _ = compute_line_normal_and_tangent(self.line, pt)
                # J stores parameter derivatives in rows: J^-1 maps a
                # parameter-space normal to the physical conormal.
                n_phys = np.linalg.solve(J, n_param)
                n_phys /= np.linalg.norm(n_phys)

                penalty_near_zero = self.penalty_factors and all(
                    np.isclose(pfac, 0.0, atol=1e-8) for pfac in self.penalty_factors)
                theta = -1 if penalty_near_zero else 1
                if penalty_near_zero and not use_nitsche:
                    print("Warning: penalty factors are near zero but Nitsche is disabled.")

                gp.n = n_param
                gp.t = t_param

                gps.append(gp)


                for i in range(n_cps):
                    for j in range(n_cps):
                        for a in range(dof_per_cp):
                            K_el[dof_per_cp * i + a, dof_per_cp * j + a] += (
                                    + conductivity * p ** 2 / h * self.penalty_factors[a] * N[i] * N[j] * gwl
                                    + use_nitsche * (
                                            -conductivity * gwl * np.dot(dN_dx[:, j], n_phys) * N[i]
                                            -conductivity * gwl * theta * np.dot(dN_dx[:, i], n_phys) * N[j]
                                    )
                            )
                    for a in range(dof_per_cp):
                        f_el[dof_per_cp * i + a] += (
                                + conductivity * p ** 2 / h * self.penalty_factors[a] * N[i] * u_D[a] * gwl
                                + use_nitsche * (
                                        -conductivity * gwl * theta * np.dot(dN_dx[:, i], n_phys) * u_D[a]
                                )
                        )

            if not hasattr(cell, "gps_line"):
                cell.gps_line = gps
            else:
                cell.gps_line.extend(gps)

        return K_el, f_el
