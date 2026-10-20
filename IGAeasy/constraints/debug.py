import numpy as np
from shapely.geometry import LineString, LinearRing
import math


def compute_taylor_term(derivative: float, dx: float, n_k: int, dy: float, k: int) -> float:
    """Compute a single term in a 2D Taylor expansion."""
    return derivative * (dx ** n_k) * (dy ** k) / (math.factorial(n_k) * math.factorial(k))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from IGAeasy.geometry import get_geometry
    from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
    import numpy as np
    from shapely.geometry import LineString
    from IGAeasy.trimming import trim_elements
    from IGAeasy.geometry import get_geometry

    n = 1
    n_elU = n
    n_elV = n
    ax = 1.
    ay = 1.
    p = 1
    conti = p - 1
    ximax = 1
    etamax = 1
    Bspl = get_geometry.surface(n_elU, n_elV, ax, ay, ximax, etamax, conti, p)


    ku = Bspl.knot_vectors[0]
    kv = Bspl.knot_vectors[1]
    cps = Bspl.control_points
    p = Bspl.degrees[0]

    # initialize THB
    n_max_level = 2
    refinement_type = 'h'
    RS = HierarchicalSpace(ku, kv, p, cps, n_max_level, refinement_type)

    u=0.55
    v=0.45
    dx_vec = [0.3, 0.2]

    supported_bfs = RS.elements[0].supported_b_splines
    supported_bfs.sort(key=lambda bf: bf.id)

    n_nodes = len(supported_bfs)
    H_sum_vec = np.zeros(n_nodes)
    N = np.array([bf(u, v, r1=0, r2=0) for bf in supported_bfs])

    H_taylor_vec = np.zeros(n_nodes)
    H_exact_vec = np.zeros(n_nodes)

    for i, bf in enumerate(supported_bfs):
        H_taylor_term = 0.0
        for n in range(1, bf.p + 1):
            for k in range(n + 1):
                n_k = n - k
                derivative = bf(u, v, r1=n_k, r2=k)
                H_taylor_term += compute_taylor_term(derivative, dx_vec[0], n_k, dx_vec[1], k)
        H_taylor_vec[i] = H_taylor_term + N[i]
        # H_taylor_vec[i] = N[i] + dx_vec[0]*bf(u, v, r1=1, r2=0) + dx_vec[1]*bf(u, v, r1=0, r2=1) +dx_vec[0]*dx_vec[1]*bf(u, v, r1=1, r2=1)
        # H_exact_vec[i] = bf(u + dx_vec[0], v + dx_vec[1], r1=0, r2=0)

    for i in range(n_nodes):
        print(
            f"Taylor: {H_taylor_vec[i]:.12f}  Exact: {H_exact_vec[i]:.12f}  Error: {abs(H_taylor_vec[i] - H_exact_vec[i]):.2e}")



