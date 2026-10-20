
class gauss_point_sbm:

    def __init__(self, u, v, s, w, n, t) -> None:
        """
        Initialize an Integrationpoint at the surrogate boundary via u and v.

        :param u: coordinate in u direction
        :param v: coordinate in v direction
        :param s: line coorinate
        :param n: normal of the active Edge
        :param t: tangent of the active Edge
        """

        self.u = u
        self.v = v
        self.s = s
        self.weight = w
        self.n = n
        self.t = t

        self.closest_line = None
        self.closest_point = None
        self.s_line_closest_point = None
        self.difference_vector = None
        self.n_line_closest_point = None
        self.t_line_closest_point = None



