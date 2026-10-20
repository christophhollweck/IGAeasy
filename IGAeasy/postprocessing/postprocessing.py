from IGAeasy.postprocessing.shell_results import (interpolate_shell_results, evaluate_shell_result, reconstruct_shell_stress, von_mises, invalidate_shell_interpolators)
import numpy as np

def update_stress_strain_membrane2D(RS):
    """Plane-stress membrane results in the local orthonormal frame."""
    for el in RS.elements:
        invalidate_shell_interpolators(el)
        if getattr(el, 'void', False) or not getattr(el, 'active', True):
            continue
        bfs = sorted(el.supported_b_splines, key=lambda b: b.id)
        values = np.concatenate([np.asarray(b.dof_values) for b in bfs])
        for gp in el.gausspoints:
            strain = gp.Bmatrices[0] @ values
            e1, e2, _ = gp.local_basis[0]
            rotation = np.array([e1[:2], e2[:2]])
            # B uses engineering shear; rotate the strain tensor, then repack.
            tensor = np.array([[strain[0], strain[2]/2], [strain[2]/2, strain[1]]])
            local = rotation @ tensor @ rotation.T
            local_strain = np.array([local[0, 0], local[1, 1], 2*local[0, 1]])
            stress = gp.material_model.get_C_voigt_cart_2D() @ local_strain
            gp.strain_global = [strain]
            gp.strain_local = [local_strain]
            gp.stress_local = [stress]
            gp.N = gp.t * stress
            gp.sigma_vM = von_mises([stress[0], stress[1], 0., stress[2], 0., 0.])


def interpolate_stress_membrane2D(RS, type=None):
    return interpolate_shell_results(RS, type, available=('N', 'vM'))


def eval_stress_membrane2D(RS, xi, eta, type=None):
    return evaluate_shell_result(RS, xi, eta, type, available=('N', 'vM'))

def update_stress_strain_RM6dof(RS):
    # Postprocessing
    for el in RS.elements:
        invalidate_shell_interpolators(el)
        if getattr(el, 'void', False) or not getattr(el, 'active', True):
            continue
        bfs_el = list(el.supported_b_splines)
        bfs_el.sort(key=lambda bspline: bspline.id)
        u_el = np.array([bf.dof_values for bf in bfs_el]).flatten()
        for gauss_xy in el.gausspoints:
            t = gauss_xy.t
            N = np.zeros(3)  # Normal forces: Nx, Ny, Nz
            M = np.zeros(3)  # Moments: Mx, My, Mz
            Q = np.zeros(2)
            gauss_xy.strain_global = []
            gauss_xy.strain_local = []
            gauss_xy.stress_local = []
            gauss_xy.sigma_vM = []
            for i in range(len(gauss_xy.weight_outerplane)):
                zeta = gauss_xy.zeta[i]
                w = gauss_xy.weight_outerplane[i]
                B = gauss_xy.Bmatrices[i]
                C_loc = gauss_xy.material_model.get_C_voigt_cart_3D()
                R = gauss_xy.rotationmatrices[i]

                eps_glob = B @ u_el
                gauss_xy.strain_global.append(eps_glob)

                eps_loc = R.T @ eps_glob
                gauss_xy.strain_local.append(eps_loc)

                sigma_loc = C_loc @ eps_loc
                gauss_xy.stress_local.append(sigma_loc)

                # Calculation of shear forces and moments in local orthonormal system
                # σ in Voigt notation: [σ_xx, σ_yy, σ_zz, τ_xy, τ_yz, τ_xz]
                sigma_xx = sigma_loc[0]
                sigma_yy = sigma_loc[1]
                sigma_zz = sigma_loc[2]
                tau_xy = sigma_loc[3]
                tau_yz = sigma_loc[4]
                tau_xz = sigma_loc[5]

                gauss_xy.sigma_vM.append(von_mises(sigma_loc))

                # Calculation of normal and shear forces (N, Q) and moments (M)
                A = w * t / 2
                z = zeta * t / 2

                N[0] += sigma_xx * A  # Nx
                N[1] += sigma_yy * A  # Ny
                N[2] += tau_xy * A

                # Calculate moments
                M += np.array([sigma_xx, sigma_yy, tau_xy]) * z * A

                # Calculate shear force (assumed calculation)
                Q += np.array([tau_xz, tau_yz]) * A

            gauss_xy.sigma_vM = gauss_xy.sigma_vM[np.argmax(np.abs(gauss_xy.sigma_vM))]
            gauss_xy.N = N
            gauss_xy.M = M
            gauss_xy.Q = Q


def interpolate_stress_RM6dof(RS, type=None):
    return interpolate_shell_results(RS, type, available=('N', 'M', 'Q', 'vM'))


def eval_stress_RM6dof(RS, xi, eta, type=None):
    return evaluate_shell_result(RS, xi, eta, type, available=('N', 'M', 'Q', 'vM'))


def update_stress_strain_KL(RS):
    # Postprocessing
    for el in RS.elements:
        invalidate_shell_interpolators(el)
        if getattr(el, 'void', False) or not getattr(el, 'active', True):
            continue
        bfs_el = list(el.supported_b_splines)
        bfs_el.sort(key=lambda bspline: bspline.id)
        u_el = np.array([bf.dof_values for bf in bfs_el]).flatten()
        for gauss_xy in el.gausspoints:
            t = gauss_xy.t
            N = np.zeros(3)
            M = np.zeros(3)

            Bm, Bb = gauss_xy.Bmatrices[0]
            Dm, Db = gauss_xy.D
            # not necessarily a pure rotation
            R = gauss_xy.rotationmatrices[0]

            # covariant components in contravariant basis
            eps_m_cov = Bm @ u_el
            eps_b_cov = Bb @ u_el

            # contravariant components in covariant basis
            n_con = Dm @ eps_m_cov
            m_con = Db @ eps_b_cov

            # N and M are in Voigt notation (11, 22, 12) in a local curvilinear basis
            # R maps from contravariant basis to orthonormal basis

            N = R @ n_con  # Membrane forces: [N_11, N_22, N_12]
            M = R @ m_con  # Bending moments: [M_11, M_22, M_12]

            gauss_xy.sigma_vM = max(von_mises(reconstruct_shell_stress(N, M, None, t, z))
                                      for z in (-t/2, 0., t/2))
            gauss_xy.N = N
            gauss_xy.M = M
            gauss_xy.Q = None  # KL has no constitutive transverse-shear field.


def interpolate_stress_KL(RS, type=None):
    return interpolate_shell_results(RS, type, available=('N', 'M', 'vM'))


def eval_stress_KL(RS, xi, eta, type=None):
    return evaluate_shell_result(RS, xi, eta, type, available=('N', 'M', 'vM'))


def update_stress_strain_RM5dof(RS):
    # Postprocessing
    for el in RS.elements:
        invalidate_shell_interpolators(el)
        if getattr(el, 'void', False) or not getattr(el, 'active', True):
            continue
        bfs_el = list(el.supported_b_splines)
        bfs_el.sort(key=lambda bspline: bspline.id)
        u_el = np.array([bf.dof_values for bf in bfs_el]).flatten()
        for gauss_xy in el.gausspoints:
            t = gauss_xy.t
            N = np.zeros(3)
            M = np.zeros(3)
            Q = np.zeros(2)

            Bm, Bb, Bs = gauss_xy.Bmatrices[0]
            Dm, Db, Ds = gauss_xy.D
            Rmb, Rs = gauss_xy.rotationmatrices[0]

            # covariant components in contravariant basis
            eps_m_cov = Bm @ u_el
            eps_b_cov = Bb @ u_el
            eps_s_cov = Bs @ u_el

            # contravariant components in covariant basis
            n_con = Dm @ eps_m_cov
            m_con = Db @ eps_b_cov
            s_con = Ds @ eps_s_cov

            # N and M are in Voigt notation (11, 22, 12) in an orthonormal basis
            # Q in Voigt notation (13, 23) in an orthonormal basis
            # R maps from contravariant basis to orthonormal basis

            N = Rmb @ n_con  # Membrane forces: [N_11, N_22, N_12]
            M = Rmb @ m_con  # Bending moments: [M_11, M_22, M_12]
            Q = Rs @ s_con   # Shear forces: [Q_13, Q_23]

            gauss_xy.sigma_vM = max(von_mises(reconstruct_shell_stress(N, M, Q, t, z))
                                      for z in (-t/2, 0., t/2))
            gauss_xy.N = N
            gauss_xy.M = M
            gauss_xy.Q = Q

def interpolate_stress_RM5dof(RS, type=None):
    return interpolate_shell_results(RS, type, available=('N', 'M', 'Q', 'vM'))


def eval_stress_RM5dof(RS, xi, eta, type=None):
    return evaluate_shell_result(RS, xi, eta, type, available=('N', 'M', 'Q', 'vM'))




def update_cps(RS, dis, dof_per_cp=6, dimensions=3, formulation=None):
    if formulation is not None:
        counts = {'RM6': 6, 'RM5': 5, 'RM5_v2': 5, 'RM5_hier': 5, 'KL': 3}
        if formulation not in counts or counts[formulation] != dof_per_cp:
            raise ValueError('Shell formulation and DOF count do not match')
        if np.asarray(dis).shape != (len(RS.basis)*dof_per_cp,):
            raise ValueError('Supply the complete shell coefficient vector')
        RS.shell_formulation = formulation
    for b in RS.basis:
        i = b.id
        b.displacements[:dimensions] = np.array(dis[dof_per_cp * i:dof_per_cp * i + dimensions])
        b.dof_values = np.array(dis[dof_per_cp * i:dof_per_cp * i + dof_per_cp])

def update_scalar_field(RS, values, dof_per_cp=1):
    """
    Updates scalar field values (e.g. temperature) at control points.

    :param RS: The function space (e.g. containing B-spline basis functions)
    :param values: Global solution vector with scalar values
    :param dof_per_cp: Degrees of freedom per control point (default: 1)
    """
    for b in RS.basis:
        i = b.id
        b.scalar_value = values[dof_per_cp * i]
        b.dof_values = values[dof_per_cp * i : dof_per_cp * i + dof_per_cp]

def lift_gps_to_cps(RS):
    """Quadrature-weighted L2 projection of von Mises values to spline coefficients."""
    from IGAeasy.postprocessing.projection import project_scalar_results
    return project_scalar_results(RS)
