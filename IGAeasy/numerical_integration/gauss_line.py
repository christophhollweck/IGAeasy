
class gauss_point:

    def __init__(self, u, v, s, w, line) -> None:
        """
        Initialize an Integrationpoint for a line integral.

        :param u: coordinate in u direction
        :param v: coordinate in v direction
        :param s: line coorinate
        :param n: normal
        :param t: tangent
        """

        self.u = u
        self.v = v
        self.s = s
        self.weight = w
        self.n = None
        self.t = None

        self.line = line



