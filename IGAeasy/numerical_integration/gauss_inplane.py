
class gauss_point:

    def __init__(self, xi: float, eta: float, weight_inplane: float) -> None:
        """
        Initialize an inplane Integrationpoint location via xi and eta.
        The actual Integrationpoints are defined later with a thickness coordinate zeta

        :param xi: coordinate in xi direction
        :param eta: coordinate in eta direction
        :param weight_inplane: weight for inplane coordinates (xi, eta)
        """

        self.xi = xi
        self.eta = eta
        self.weight_inplane = weight_inplane
        self.zeta = []
        self.weight_outerplane = []
        self.local_basis = []
        self.surface_tangents = []
        self.rotationmatrices = []
        self.Bmatrices = []
        self.strain_global = []
        self.strain_local = []
        self.stress_local = []
        self.sigma_vM = []
        self.grad_u = None
        self.C_loc = None
        self.C_glob = None
        self.E = None
        self.rho = None
        self.t = None
        self.N = None
        self.M = None
        self.Q = None


    def __hash__(self):
        return hash(tuple((self.xi, self.eta, self.weight_inplane)))

    def __repr__(self):
        return (f"Integrationpoint_inplane(xi={self.xi}, eta={self.eta}, weight_inplane={self.weight_inplane}")


