import numpy as np
import scipy.sparse as sp
from IGAeasy.splines.THBSplines.b_spline import BSpline, _find_knot_interval, _evaluate_univariate_b_spline
from IGAeasy.splines.THBSplines.element import Element
from collections import defaultdict
import time



class TensorProductSpace2D():

    def __init__(self, ku, kv, p, cps, level, refinement_type):
        """
        Initialize a new TensorProductSpace.

        :param knots: a np.ndarray of knot vectors
        :param degrees: a np.ndarray of degrees corresponding to each parametric dimension
        :param dim: the number of parametric directions
        """
        self.p = p
        self.ku = ku
        self.kv = kv
        self.unique_ku = np.unique(ku)
        self.unique_kv = np.unique(kv)
        self.level = level
        self.refinement_type = refinement_type
        self.basis = []
        self.elements = []
        self.cps = cps
        self.nfuncs = None
        self.nelems = None
        self.construct_basis()
        self.init_elements_to_functions()



    def construct_basis(self):

        """
        Initialize the tensor product with basis functions with control points and elements.
        """

        k = 0
        ku_len = len(self.ku)
        kv_len = len(self.kv)
        unique_ku_len = len(self.unique_ku)
        unique_kv_len = len(self.unique_kv)

        # basis functions
        for i in range(kv_len - self.p - 1):
            for j in range(ku_len - self.p - 1):
                end_u = _at_end(self.ku, j + self.p + 1)
                end_v = _at_end(self.kv, i + self.p + 1)
                cp = self.cps[k]
                self.basis.append(
                    BSpline(self.p, k, self.level, self.ku[j:j + self.p + 2], self.kv[i:i + self.p + 2], cp,
                            end_u=end_u, end_v=end_v)
                )
                k += 1

        self.nfuncs = k

        # elements
        element_params = [(i, j, j + i * (unique_ku_len - 1)) for i in range(unique_kv_len - 1) for j in
                          range(unique_ku_len - 1)]

        for i, j, num in element_params:
            self.elements.append(
                Element(self.unique_ku[j], self.unique_ku[j + 1], self.unique_kv[i], self.unique_kv[i + 1], self.p, num, self.level)
            )

        self.nelems = num + 1

    def init_elements_to_functions(self):
        """
        Initialize the assignment of elements to functions based on their coordinates (connectivity).
        """

        tol = 1e-14

        # Convert self.ku and self.kv to numpy arrays for efficient numerical operations
        ku_array = np.array(self.ku)
        kv_array = np.array(self.kv)

        nel_u = len(self.unique_ku) - 1
        nel_v = len(self.unique_kv) - 1

        nfuns_u = len(self.ku) - self.p - 1
        nfuns_v = len(self.kv) - self.p - 1

        dic_u = {}
        dic_v = {}

        for elnum_u in range(nel_u):
            start_knot = self.unique_ku[elnum_u]
            end_knot = self.unique_ku[elnum_u + 1]

            # Use numpy to find the index where end_knot directly follows start_knot within tolerance
            start_idx = np.where((np.abs(ku_array[:-1] - start_knot) < tol) & (np.abs(ku_array[1:] - end_knot) < tol))[
                0]

            start_idx = start_idx[0]

            # Calculate the index where the first active basis function starts
            start_basis = start_idx - self.p
            end_basis = start_basis + self.p + 1

            # Make sure the indices are within valid range
            dic_u[elnum_u] = list(range(max(start_basis, 0), min(end_basis, len(self.basis))))

        for elnum_v in range(nel_v):
            start_knot = self.unique_kv[elnum_v]
            end_knot = self.unique_kv[elnum_v + 1]

            # Use numpy to find the index where end_knot directly follows start_knot within tolerance
            start_idx = np.where((np.abs(kv_array[:-1] - start_knot) < tol) & (np.abs(kv_array[1:] - end_knot) < tol))[
                0]

            start_idx = start_idx[0]

            # Calculate the index where the first active basis function starts
            start_basis = start_idx - self.p
            end_basis = start_basis + self.p + 1

            # Make sure the indices are within valid range
            dic_v[elnum_v] = list(range(max(start_basis, 0), min(end_basis, len(self.basis))))

        def get_element_number_2D(i, j, nel_u):
            return j * nel_u + i

        def get_function_number_2D(i, j, nfuns_u):
            return j * nfuns_u + i

        dic_uv = defaultdict(list)


        for elnum_v, funs_v in dic_v.items():
            for elnum_u, funs_u in dic_u.items():
                elnum = get_element_number_2D(elnum_u, elnum_v, nel_u)
                dic_uv[elnum] = [get_function_number_2D(fun_u, fun_v, nfuns_u) for fun_v in funs_v for fun_u in funs_u]


        for element in self.elements:
            el_loc_id = element.loc_id
            funs_idx = dic_uv[el_loc_id]
            element.supported_b_splines_init = [self.basis[idx] for idx in funs_idx]

            for function in element.supported_b_splines_init:
                function.elements_of_support.append(element)


        for element in self.elements:
            element.supported_b_splines = list(element.supported_b_splines_init)

        # Immutable level topology is separate from the current active support.
        self.support_by_basis = {b.loc_id: tuple(b.elements_of_support) for b in self.basis}

    def compute_projection_h_matrix(self, coarse_knots, fine_knots, p):
        """
        Computes the 1D projection matrix of the space corresponding to the
        fine knots with respect to the coarse knots and the spline degree for h refinement.

        :param coarse_knots: list/array of coarse knot vectors, one for each parametric direction
        :param fine_knots: list/array of fine knot vectors, one for each parametric direction
        :param degrees: list of spline degrees, one for each parametric direction.

        :return: a list of 1D projection matrices corresponding to each parametric dimension
        """
        matrices = []
        for fine, coarse, degree in zip(fine_knots, coarse_knots, [p, p]):
            coarse = augment_knots(coarse, degree)
            fine = augment_knots(fine, degree)
            m = len(fine) - (degree + 1)
            n = len(coarse) - (degree + 1)

            a = sp.lil_matrix((m, n), dtype=np.float64)
            fine = np.array(fine, dtype=np.float64)
            coarse = np.array(coarse, dtype=np.float64)
            for i in range(m):
                mu = find_knot_index(fine[i], coarse)
                b = 1
                for k in range(1, degree + 1):
                    tau1 = coarse[mu - k + 1:mu + 1]
                    tau2 = coarse[mu + 1:mu + k + 1]
                    omega = (fine[i + k] - tau1) / (tau2 - tau1)
                    b = np.append((1 - omega) * b, 0) + np.insert((omega * b), 0, 0)
                a[i, mu - degree:mu + 1] = b
            matrices.append(a[degree + 1:-degree - 1, degree + 1:-degree - 1])

        return matrices

    def compute_projection_p_matrix(self, coarse_knots, fine_knots, p):
        """
        Computes the 1D projection matrix of the space corresponding to the
        fine knots with respect to the coarse knots and the spline degree for p refinement.

        :param coarse_knots: list/array of coarse knot vectors, one for each parametric direction
        :param fine_knots: list/array of fine knot vectors, one for each parametric direction
        :param degrees: list of spline degrees, one for each parametric direction.

        :return: a list of 1D projection matrices corresponding to each parametric dimension
        """

        matrices = []
        for fine, coarse, degree in zip(fine_knots, coarse_knots, [p, p]):
            degree_old = degree
            degree_new = degree_old + 1
            kv0 = coarse
            kv1 = fine

            nbf_old = len(kv0) - degree_old - 1
            nbf_new = len(kv1) - degree_new - 1
            C = np.zeros((nbf_old, nbf_new))

            dict_children = {}
            id_counter = 0
            for i in range(len(kv1) - degree_new - 1):
                key = tuple(kv1[i:i + degree_new + 2])
                dict_children[key] = id_counter
                id_counter += 1

            dict_parents = {}
            hits = 0

            for i in range(nbf_old):
                bf_old = kv0[i:i + degree_old + 2]
                min_val = np.min(np.unique(bf_old))
                bf_old_norm = np.array(bf_old) - min_val
                bf_old_norm = [np.round(val, 10) for val in bf_old_norm]
                bf_old_norm_key = tuple(bf_old_norm)

                bfs_new = p_refine_knot_vector(bf_old)
                nbf_new_loc = len(bfs_new) - (degree_new + 2) + 1

                if bf_old_norm_key not in dict_parents:

                    # list of point to evaluate the new basis functions to get the scaling coefficients
                    p_eval = []
                    for k in range(nbf_new_loc):
                        p_eval.append(np.mean(bfs_new[k:k + degree_new + 2]))

                    B_old = []
                    for point in p_eval:
                        B_old.append(_evaluate_univariate_b_spline(point, bf_old, degree_old, r=0))

                    Mat = np.zeros((len(bfs_new) - degree_new - 1, len(bfs_new) - degree_new - 1))

                    for jk, point in enumerate(p_eval):
                        for ik in range(len(bfs_new) - degree_new - 1):
                            Mat[jk, ik] = _evaluate_univariate_b_spline(point, bfs_new[ik:ik + degree_new + 2], degree_new,
                                                                        r=0)

                    # solve for the coefficients
                    coeffs = np.linalg.solve(Mat, np.array(B_old))
                    dict_parents[bf_old_norm_key] = coeffs

                else:
                    hits += 1
                    coeffs = dict_parents[bf_old_norm_key]

                key = tuple(bfs_new[:degree_new + 2].tolist())
                id_bf = dict_children[key]
                C[i, id_bf: id_bf + nbf_new_loc] = coeffs
            matrices.append(sp.csr_matrix(C.T))

        return matrices

    def refine(self):
        """
        Refine the space by either k-refinement, h-refinement or p-refinement,
        depending on the refinement_type parameter.

        :param refinement_type: 'k' for k-refinement (p then h), 'h' for h-refinement only,
                                'p' for p-refinement only
        :return: refined TensorProductSpace2D and a projection matrix
        """

        refinement_type = self.refinement_type

        if refinement_type not in {'k', 'h', 'p'}:
            raise ValueError("refinement_type must be 'k', 'h', or 'p'")

        if refinement_type == 'k':
            # p-refinement
            coarse_knots_p = [self.ku, self.kv]
            fine_knots_p = [p_refine_knot_vector(knot_vector) for knot_vector in coarse_knots_p]
            projection_p_matrices_1D = self.compute_projection_p_matrix(coarse_knots_p, fine_knots_p, self.p)
            projection_p_matrix = sp.kron(projection_p_matrices_1D[1], projection_p_matrices_1D[0])

            # h-refinement
            coarse_knots_h = fine_knots_p
            fine_knots_h = [insert_midpoints(knot_vector) for knot_vector in coarse_knots_h]
            projection_h_matrices_1D = self.compute_projection_h_matrix(coarse_knots_h, fine_knots_h, self.p + 1)
            projection_h_matrix = sp.kron(projection_h_matrices_1D[1], projection_h_matrices_1D[0])

            projection_k_matrix = projection_h_matrix @ projection_p_matrix

            c_transpose = projection_k_matrix.T.tolil()

            # allocate cps with zeros
            cps_fine = np.zeros((c_transpose.shape[1], 3))
            fine_space = TensorProductSpace2D(fine_knots_h[0], fine_knots_h[1], self.p + 1, cps_fine, self.level + 1, refinement_type)

        elif refinement_type == 'h':
            # h-refinement
            coarse_knots_h = [self.ku, self.kv]
            fine_knots_h = [insert_midpoints(knot_vector) for knot_vector in coarse_knots_h]
            projection_h_matrices_1D = self.compute_projection_h_matrix(coarse_knots_h, fine_knots_h, self.p)
            projection_h_matrix = sp.kron(projection_h_matrices_1D[1], projection_h_matrices_1D[0])

            c_transpose = projection_h_matrix.T.tolil()

            # allocate cps with zeros
            cps_fine = np.zeros((c_transpose.shape[1], 3))
            fine_space = TensorProductSpace2D(fine_knots_h[0], fine_knots_h[1], self.p, cps_fine, self.level + 1,
                                              refinement_type)
        else:
            # p-refinement
            coarse_knots_p = [self.ku, self.kv]
            fine_knots_p = [p_refine_knot_vector(knot_vector) for knot_vector in coarse_knots_p]
            projection_p_matrices_1D = self.compute_projection_p_matrix(coarse_knots_p, fine_knots_p, self.p)
            projection_p_matrix = sp.kron(projection_p_matrices_1D[1], projection_p_matrices_1D[0])

            c_transpose = projection_p_matrix.T.tolil()

            # allocate cps with zeros
            cps_fine = np.zeros((c_transpose.shape[1], 3))
            fine_space = TensorProductSpace2D(fine_knots_p[0], fine_knots_p[1], self.p + 1, cps_fine, self.level + 1,
                                              refinement_type)

        return fine_space, c_transpose


    def __call__(self, xi, eta):
        """
        Evaluates the tensor product space at the point (xi, eta)

        :param xi: first component
        :param eta: second component
        :return: S(xi, eta)
        """

        def where_uv(xi, eta):
            for el in self.elements:
                if el.u_min <= xi <= el.u_max and el.v_min <= eta <= el.v_max:
                    return el

        el = where_uv(xi, eta)
        if el is None:
            return np.zeros(3)  # Return zero vector if point is outside any element

        total = np.zeros(3)
        for f in el.supported_b_splines_init:
            total += f.cp * f(xi, eta)
        return total


    def plot_surface(self, s=0.05):

        from mayavi import mlab

        def eval(u, v):
            x = np.zeros_like(u)
            y = np.zeros_like(u)
            z = np.zeros_like(u)
            for i in range(len(u)):
                eval = self(u[i], v[i])
                x[i] = eval[0]
                y[i] = eval[1]
                z[i] = eval[2]
            return x, y, z

        mlab.figure(bgcolor=(1, 1, 1))
        for b in self.basis:
            cp = b.cp
            mlab.points3d(cp[0], cp[1], cp[2], color=(1, 0, 0), scale_factor=s)

        u, v = np.meshgrid(np.linspace(self.ku[0], self.ku[-1], 20), np.linspace(self.kv[0], self.kv[-1], 20))
        x, y, z = eval(u.flatten(), v.flatten())
        x = x.reshape(u.shape)
        y = y.reshape(u.shape)
        z = z.reshape(u.shape)
        mlab.mesh(x, y, z, colormap='coolwarm')
        mlab.show()

    def plot_basis(self, pnts=20):

        from mayavi import mlab
        sc = 1
        Mat = []

        for b in self.basis:
            xi = np.linspace(b.knots_u[0], b.knots_u[-1], pnts)
            eta = np.linspace(b.knots_v[0], b.knots_v[-1], pnts)
            Xi, Eta = np.meshgrid(xi, eta)
            z = np.zeros((pnts, pnts))
            for i in range(pnts):
                for j in range(pnts):
                    z[i, j] = b(xi[i], eta[j])
            Mat.append([Xi.T, Eta.T, z])

        mlab.figure(bgcolor=(1, 1, 1))
        for i in range(len(Mat)):
            Xi = Mat[i][0]
            Eta = Mat[i][1]
            z = Mat[i][2]
            mlab.surf(sc * Xi, sc * Eta, z)

            # flat bottom surface
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        z = np.zeros((5, 5))

        X, Y = np.meshgrid(x, y)
        mlab.surf(sc * X.T, sc * Y.T, z)

        mlab.show()



def insert_midpoints(knots):
    """
    Inserts midpoints in all interior knot intervals of a p+1 regular knot
    vector.

    :param knots: p + 1 regular knot vector to be refined
    :return: refined_knots
    """

    knots = np.array(knots, dtype=np.float64)
    unique_knots = np.unique(knots)
    midpoints = (unique_knots[:-1] + unique_knots[1:]) / 2
    knots = np.sort(np.concatenate((knots, midpoints)))

    return knots

def p_refine_knot_vector(kv):
    """
    Increases the multiplicity of all knots by one.

    :param knots: p + 1 regular knot vector to be refined
    :return: refined_knots
    """

    knots_2_duplicate = np.unique(kv)
    return np.sort(np.concatenate((kv, knots_2_duplicate)))

def augment_knots(knots, degree):

    """
    Adds degree + 1 values to either end of the knot vector, in order to
    facilitate matrix based evaluation.

    :param knots: knot vector
    :param degree: polynomial degree
    :return: padded knot vector
    """

    return np.pad(knots, (degree + 1, degree + 1), 'constant',
                  constant_values=(knots[0] - 1, knots[-1] + 1))


def find_knot_index(x, knots, endpoint=False):

    """
    Finds the index i such that t_i <= x < t_i+1.
    If endpoint is True, then we check for t_n-1 <= x <= t_n
    for the final index.

    :param x: point in question
    :param knots: knot vector
    :param endpoint: Boolean, whether to include the final knot

    :return: index i
    """
    # if we have requested end point, and are at the end, return
    # corresponding index.
    knots = knots
    if endpoint and (knots[-2] <= x <= knots[-1]):
        i = max(np.argmax(knots < x) - 1, 0)
        return len(knots) - i - 2
    # if we are outside the domain, return -1
    if x < knots[0] or x >= knots[-1]:
        return -1
    # otherwise, return the corresponding index
    return np.max(np.argmax(knots > x) - 1, 0)

def _at_end(knots, index):
    return abs(knots[-1] - knots[index]) < 1.0E-13

if __name__ == '__main__':

    ku = [0, 0, 0, 1, 2, 3, 4, 5, 5, 5]
    kv = [0, 0, 0, 1, 2, 3, 4, 5, 5, 5]

    p = 2
    conti = 1
    cps = np.zeros((49,3))
    level = 0
    T = TensorProductSpace2D(ku, kv, p, cps, level)
    T1 = T.refine()
    print(T1.basis[3].grad(1,0.25))
    pass
