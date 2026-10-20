import numpy as np


class LinearElasticMaterial:
    """Isotropic plane-stress elasticity and thickness-integrated shell laws.

    Voigt strains use engineering shear; stresses use tensor shear components.
    kappa corrects transverse shell shear only, never membrane shear.
    """
    def __init__(self, E, nu, rho, kappa=5/6):
        """
        Initializes a linear elastic material.

        Args:
            E (float): Young's modulus [Pa].
            nu (float): Poisson's ratio.
            rho (float): Density [kg/m^3].
            kappa (float): Shear correction factor.
        """
        self.E = E
        self.nu = nu
        self.rho = rho
        self.kappa = kappa


    # --- 2D constitutive relations ---
    def get_C_voigt_cart_2D(self):
        """
        Constructs the 2D stiffness matrix in Voigt notation (3x3).
        Assumes plane stress.

        C = E/(1-nu^2) * [[1,    nu,                    0],
                          [nu,   1,                     0],
                          [0,    0,         (1-nu)/2]]

        Strain order: (epsilon_xx, epsilon_yy, gamma_xy).
        Stress order: (sigma_xx, sigma_yy, tau_xy).
        """
        nu = self.nu
        factor = self.E / (1 - nu ** 2)
        C_voigt = factor * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1 - nu) / 2]
        ])
        return C_voigt

    # --- 3D constitutive relations ---
    def get_C_voigt_cart_3D(self):
        """
        Constructs the 3D stiffness matrix in Voigt notation (6x6) for isotropic elasticity with
        imposed plane stress condition.

        Order: (xx, yy, zz, xy, yz, xz), with engineering shear strains.
        The zz row/column is zero: this is a shell law, not solid 3D elasticity.
        """
        E = self.E
        nu = self.nu
        kappa = self.kappa
        C_voigt = E / (1 - nu ** 2) * np.array(
            [[1, nu, 0, 0, 0, 0],
             [nu, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0.5 * (1 - nu), 0, 0],
             [0, 0, 0, 0, kappa / 2 * (1 - nu), 0],
             [0, 0, 0, 0, 0, kappa / 2 * (1 - nu)]])

        return C_voigt

    def compute_C_voigt_curv_2D(self, A_contra, t):
        """Return membrane, bending and transverse shear matrices.

        A_contra is the inverse surface metric. Membrane/bending strains are
        (e11, e22, 2*e12), and resultants are (n11, n22, n12).
        The returned matrices include thickness factors t, t**3/12 and t.
        """

        A11 = A_contra[0, 0]
        A22 = A_contra[1, 1]
        A12 = A_contra[0, 1]
        A21 = A_contra[1, 0]
        nu = self.nu
        E = self.E
        kappa = self.kappa


        C = E / (1 - nu ** 2) * np.array([
            [A11 ** 2, nu * A11 * A22 + (1 - nu) * A12 ** 2, A11 * A12],
            [nu * A11 * A22 + (1 - nu) * A12 ** 2, A22 ** 2, A22 * A12],
            [A11 * A12, A22 * A12, 0.5 * ((1 - nu) * A11 * A22 + (1 + nu) * A12 ** 2)]
        ])

        Dm = t * C
        Db = (t ** 3) / 12 * C
        G = E / (2 * (1 + nu))
        Ds = kappa * G * t * np.array([[A11, A12], [A21, A22]])

        return Dm, Db, Ds


    def compute_C_voigt_curv_2D_directional(self, A_contra, metric_direction, t):
        """Metric-direction derivatives of membrane, bending and shear laws.

        Material parameters and thickness are held constant. Both arguments
        are contravariant (inverse) metrics, not covariant metrics.
        """
        a, b, c = A_contra[0, 0], A_contra[1, 1], A_contra[0, 1]
        da, db, dc = metric_direction[0, 0], metric_direction[1, 1], metric_direction[0, 1]
        nu = self.nu
        mixed = nu*(da*b+a*db)+2*(1-nu)*c*dc
        C = self.E/(1-nu**2)*np.array([
            [2*a*da, mixed, da*c+a*dc],
            [mixed, 2*b*db, db*c+b*dc],
            [da*c+a*dc, db*c+b*dc, .5*(1-nu)*(da*b+a*db)+(1+nu)*c*dc]])
        return t*C, t**3/12*C, self.kappa*self.E*t/(2*(1+nu))*np.asarray(metric_direction)

    def compute_C_tensors_curv_2D(self, A_contra, t):
        """
        Computes thickness-integrated isotropic plane-stress shell tensors.

        Parameters:
        - A_contra: contravariant metric tensor (2x2 NumPy array)
        - t: shell thickness

        Returns:
        - D0: membrane stiffness (4th-order tensor, shape 2x2x2x2)
        - D2: bending stiffness (4th-order tensor, shape 2x2x2x2)
        - D0_shear: shear stiffness (2x2 tensor)
        """

        D0 = np.zeros((2, 2, 2, 2))
        D2 = np.zeros((2, 2, 2, 2))
        D0_shear = np.zeros((2, 2))

        A = A_contra
        E = self.E
        nu = self.nu
        alpha_s = self.kappa
        E_t = E * t
        E_t3 = E * t ** 3

        # Constitutive coefficients
        C1 = (nu * E_t) / (1 - nu ** 2)
        C2 = E_t / (2 * (1 + nu))
        C3 = (nu * E_t3) / (12 * (1 - nu ** 2))
        C4 = E_t3 / (24 * (1 + nu))

        for a in range(2):
            for b in range(2):
                for g in range(2):
                    for d in range(2):
                        D0[a, b, g, d] = (
                                C1 * A[a, b] * A[g, d]
                                + C2 * (A[a, g] * A[b, d] + A[a, d] * A[b, g])
                        )
                        D2[a, b, g, d] = (
                                C3 * A[a, b] * A[g, d]
                                + C4 * (A[a, g] * A[b, d] + A[a, d] * A[b, g])
                        )

        for a in range(2):
            for b in range(2):
                D0_shear[a, b] = alpha_s * E / (2 * (1 + nu)) * t * A[a, b]

        return D0, D2, D0_shear

    def tensor4_to_voigt2D(self, C):
        """
        Map 2x2x2x2 stiffness tensor to 3x3 Voigt matrix in 2D (plane stress or strain).
        Symmetries assumed.
        Maps covariant engineering strains (e11, e22, 2*e12) to contravariant
        stress/resultant components (s11, s22, s12).
        """
        voigt = np.zeros((3, 3))

        index_map = {
            (0, 0): 0,
            (1, 1): 1,
            (0, 1): 2,
            (1, 0): 2
        }

        for α in range(2):
            for β in range(2):
                for γ in range(2):
                    for δ in range(2):
                        i = index_map[(α, β)]
                        j = index_map[(γ, δ)]
                        coeff = 1.0
                        if (α != β):
                            coeff *= 0.5
                        if (γ != δ):
                            coeff *= 0.5
                        voigt[i, j] += coeff * C[α, β, γ, δ]

        return voigt
