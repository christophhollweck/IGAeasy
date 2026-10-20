import typing
from typing import List
import numpy as np
from collections import defaultdict
from rtree import index
from IGAeasy.splines.LRSplines.aux_split_functions import split_single_basis_function
from IGAeasy.splines.LRSplines.b_spline import BSpline
from IGAeasy.splines.LRSplines.element import Element
from IGAeasy.splines.LRSplines.meshline import Meshline
from IGAeasy.splines.control_point_fields import control_point_fields

Vector = typing.Union[typing.List['float'], np.ndarray]


class LRSpline:
    def __init__(self, p: int, ku: Vector, kv: Vector, cps: Vector):
        self.p = p
        self.ku = ku
        self.kv = kv
        self.unique_ku = np.unique(self.ku)
        self.unique_kv = np.unique(self.kv)
        self.trimmed = False
        self.BReps = []

        # Data structures for elements, basis functions, and meshlines
        self.elements_dic = {}  # Use ID as keys
        self.basis_dic = {}  # Full basis keys; hash collisions cannot merge functions.
        self._basis_by_index = {}  # Integer object IDs for the spatial index.
        self.meshlines_h = defaultdict(list)  # Use const as keys
        self.meshlines_v = defaultdict(list)  # Use const as keys
        self.meshlines_id = {}

        # Initialization steps
        self._initialize_elements()
        self._initialize_basis_functions(cps)
        self._initialize_meshlines_h()
        self._initialize_meshlines_v()

        # Create R-trees
        self._basis_by_index = {id(b): b for b in self.basis_dic.values()}
        self.bfs_rtree = self._create_rtree(self.basis_dic, use_hash=False)
        self.els_rtree = self._create_rtree(self.elements_dic, use_hash=False)
        self.mls_h_rtree = self._create_rtree({id(ml): ml for sublist in self.meshlines_h.values() for ml in sublist}, use_hash=False)
        self.mls_v_rtree = self._create_rtree({id(ml): ml for sublist in self.meshlines_v.values() for ml in sublist}, use_hash=False)
        self.get_connectivity()

    def _initialize_elements(self):
        num = 0
        for i in range(len(self.unique_kv) - 1):
            for j in range(len(self.unique_ku) - 1):
                el = Element(
                    self.unique_ku[j],
                    self.unique_ku[j + 1],
                    self.unique_kv[i],
                    self.unique_kv[i + 1],
                    self.p,
                    num
                )
                self.elements_dic[id(el)] = el  # Use ID as key
                num += 1

    def _initialize_basis_functions(self, cps):
        k = 0
        for i in range(len(self.kv) - self.p - 1):
            for j in range(len(self.ku) - self.p - 1):
                cp = cps[k]
                k += 1
                bf = BSpline(
                    self.p,
                    self.ku[j:j+self.p+2],
                    self.kv[i:i+self.p+2],
                    cp,
                    weight=1
                )
                self.basis_dic[bf.basis_key] = bf

    def _initialize_meshlines_h(self):
        for i, val in enumerate(self.unique_kv):
            const_value = float(val)
            new_m = Meshline(
                start=self.unique_ku[0],
                stop=self.unique_ku[-1],
                constant_value=val,
                axis=1
            )
            new_m.set_multiplicity(self.kv)
            self.meshlines_h[const_value].append(new_m)
            self.meshlines_id[id(new_m)] = new_m

    def _initialize_meshlines_v(self):
        for j, val in enumerate(self.unique_ku):
            const_value = float(val)
            new_m = Meshline(
                start=self.unique_kv[0],
                stop=self.unique_kv[-1],
                constant_value=val,
                axis=0
            )
            new_m.set_multiplicity(self.ku)
            self.meshlines_v[const_value].append(new_m)
            self.meshlines_id[id(new_m)] = new_m

    @staticmethod
    def _create_rtree(objects, use_hash: bool):
        idx = index.Index()
        for key, obj in objects.items():
            # Retrieve the AABB from the get_AABB method
            aabb = obj.get_AABB()
            u_min, u_max = aabb[0][0], aabb[0][1]
            v_min, v_max = aabb[1][0], aabb[1][1]

            # Insert the AABB into the RTree using the appropriate key (hash or ID)
            idx.insert(hash(obj) if use_hash else id(obj), (u_min, v_min, u_max, v_max))

        return idx

    def get_connectivity(self):
        """
        Links all B-Splines and Elements by updating:
        - bf.elements_of_support: List of elements within the support of the B-Spline bf.
        - el.supported_b_splines: List of active B-Splines affecting the element el.

        The checks are performed efficiently using the RTree.
        """
        # Initialize support lists
        for bf in self.basis_dic.values():
            bf.elements_of_support = []

        for el in self.elements_dic.values():
            el.supported_b_splines = []

        # Iterate through all B-Splines
        for bf_hash, bf in self.basis_dic.items():
            # Find candidate elements from the RTree whose AABBs overlap with the B-Spline's AABB
            element_candidates_ids = list(self.els_rtree.intersection(self.flat_AABB(bf.get_AABB())))
            element_candidates_ids = np.unique(element_candidates_ids)

            # Filter candidates that are actually in support
            for element_id in element_candidates_ids:
                el = self.elements_dic.get(element_id)
                if el and bf.intersects(el):  # Check if element intersects
                    # Link B-Spline to element
                    bf.elements_of_support.append(el)
                    # Link element to B-Spline
                    el.supported_b_splines.append(bf)

        # Update the elements and their IDs
        self.elements = list(self.elements_dic.values())
        for i, el in enumerate(self.elements):
            el.id = i
            el._mesh_elements = self.elements

        # Update the basis functions and their IDs
        self.basis = list(self.basis_dic.values())
        for i, bf in enumerate(self.basis):
            bf.id = i
            bf.level = np.min([el.level for el in bf.elements_of_support])

    def get_funcs_in_rectangle(self, level, AABBs):
        """
        Efficiently retrieves all active basis function objects fully contained within the given AABBs for a specific level using an RTree.

        :param level: The level of basis functions to consider.
        :param AABBs: A list of AABBs, where each AABB is defined as [[u_min, u_max], [v_min, v_max]].
        :return: A list of unique active basis function objects fully contained within the AABBs and matching the specified level.
        """
        funcs = set()
        eps = 1e-10

        # Flatten AABBs to fit the RTree format and add a small tolerance
        flattened_AABBs = [
            (AABB[0][0] - eps, AABB[1][0] - eps, AABB[0][1] + eps, AABB[1][1] + eps)
            for AABB in AABBs
        ]

        # Retrieve all candidate basis functions for each AABB
        for flat_AABB, original_AABB in zip(flattened_AABBs, AABBs):
            # Find candidates whose AABBs intersect with the flattened AABB
            basis_candidate_hashes = list(self.bfs_rtree.intersection(flat_AABB))


            for basis_hash in basis_candidate_hashes:
                basis = self._basis_by_index.get(
                    basis_hash)
                if not basis:
                    continue

                # Check if the basis is fully contained within the original AABB and matches the specified level
                basis_aabb = basis.get_AABB()
                if (
                        basis.level == level
                        and basis_aabb[0][0] >= original_AABB[0][0] - eps
                        and basis_aabb[0][1] <= original_AABB[0][1] + eps
                        and basis_aabb[1][0] >= original_AABB[1][0] - eps
                        and basis_aabb[1][1] <= original_AABB[1][1] + eps
                ):
                    funcs.add(basis)

        return list(funcs)

    def get_funcs_ibc(self, level):
        """
        Return all basis functions on the given level whose support:
        1) lies inside at least one outer loop
        2) is not fully inside any inner loop
        3) does not cut a loop boundary
        """

        from shapely.geometry import box
        tol = 1e-13

        outerloops = [b.polygon for b in self.BReps if b.outerloop]
        innerloops = [b.polygon for b in self.BReps if not b.outerloop]

        result = []

        # Loop over all basis functions in the hierarchical space
        for fun in self.basis:

            if fun.level != level:
                continue

            # Build parametric support box (axis-aligned)
            u_min, u_max = fun.knots_u[0], fun.knots_u[-1]
            v_min, v_max = fun.knots_v[0], fun.knots_v[-1]

            support_box = box(
                u_min - tol,
                v_min - tol,
                u_max + tol,
                v_max + tol
            )

            # ---------------------------------------------
            # 1 fully outside all outer loops
            # ---------------------------------------------
            intersects_outer = False
            for poly in outerloops:
                # If no intersection, function lies completely outside this outer loop
                # We want at least one intersection -> otherwise: continue (skip)
                if poly.intersects(support_box):
                    intersects_outer = True
                    break

            if not intersects_outer:
                continue

            # ---------------------------------------------
            # 2 fully inside any inner loop
            # ---------------------------------------------
            inside_inner = False
            for poly in innerloops:
                if poly.contains(support_box):
                    inside_inner = True
                    break

            if inside_inner:
                continue

            # ---------------------------------------------
            # 3 function cuts any loop boundary
            # intersects but not fully contained
            # ---------------------------------------------
            cuts_boundary = False
            for poly in outerloops + innerloops:
                if poly.intersects(support_box) and not poly.contains(support_box):
                    cuts_boundary = True
                    break

            if cuts_boundary:
                continue

            # Valid basis function
            result.append(fun)

        return result


    def get_funcs_from_point(self, level, listofpoints):
        """
        Efficiently retrieves all active basis function objects containing the specified points for a specific level using an RTree.

        :param level: The hierarchical level of basis functions to consider.
        :param listofpoints: A list of points, where each point is defined as [u, v].
        :return: A list of unique active basis function objects containing the points.
        """
        funcs = set()
        eps = 1e-10

        # Iterate over all points
        for point in listofpoints:
            u, v = point

            # Expand the point into a very small AABB for intersection checking
            flat_point_AABB = (u - eps, v - eps, u + eps, v + eps)

            # Find candidate basis functions whose AABBs intersect this small AABB
            basis_candidate_hashes = list(self.bfs_rtree.intersection(flat_point_AABB))

            for basis_hash in basis_candidate_hashes:
                basis = self._basis_by_index.get(
                    basis_hash)
                if not basis:
                    continue

                # Check if the basis contains the point and matches the specified level
                if basis.level == level:
                    basis_aabb = basis.get_AABB()
                    u_min, u_max = basis_aabb[0]
                    v_min, v_max = basis_aabb[1]
                    h_u = u_max - u_min
                    h_v = v_max - v_min
                    p = self.p + 1

                    # if (
                    #         u_min + h_u / p <= u <= u_max - h_u / p
                    #         and v_min + h_v / p <= v <= v_max - h_v / p
                    # ):

                    if (u_min <= u <= u_max and v_min <= v <= v_max):
                        funcs.add(basis)

        return list(funcs)


    def refine_by_structured_mesh(self, bfs, multiplicity=1, span_rtol=1e-6) -> None:
        """
        Refines marked supports by bisecting their largest knot spans.
        Equidistant child knots are not guaranteed for arbitrary nonuniform inputs.

        :param glob_funcnum: function to refine
        :param multiplicity: multiplicity of the meshline to enforce desired continuity
        """
        meshlines = []
        meshlines.append([self.get_structured_refinement_meshlines(bf, multiplicity, span_rtol) for bf in bfs])
        meshlines = [meshline for sublist1 in meshlines for sublist2 in sublist1 for meshline in sublist2]
        [self.insert_line(meshline) for meshline in meshlines]

        #self.get_connectivity()


    def get_structured_refinement_meshlines(self, bf, multiplicity=1, span_rtol=1e-6):
        """
        Bisects the largest distinct knot spans in each parameter direction.
        This balances dyadically refined spans; arbitrary nonuniform inputs
        need not produce equidistant child knot vectors.
        span_rtol compares spans relative to the largest span, with an
        additional allowance for coordinate roundoff.
        1D Example: in [0,1,2,2.5] only the first two spans are divided inserting
        the knots [0.5, 1.5].
        :param glob_funcnum: function to refine
        :param multiplicity: multiplicity of the meshline to enforce desired continuity
        :return: meshlines
        """
        if not np.isfinite(span_rtol) or span_rtol < 0:
            raise ValueError("span_rtol must be finite and nonnegative.")

        def selected_midpoints(knots):
            knots = np.unique(knots)
            spans = np.diff(knots)
            if not len(spans):
                raise ValueError("Basis support must contain a positive knot span.")
            largest = np.max(spans)
            # Subtracting shifted coordinates introduces errors of a few ULPs.
            roundoff = 4 * np.max(np.abs(np.spacing(knots)))
            tolerance = span_rtol * largest + roundoff
            selected = spans >= largest - tolerance
            return (knots[:-1] + spans / 2)[selected]

        knotsu, knotsv = bf.knots_u, bf.knots_v
        horizontal = [Meshline(knotsu[0], knotsu[-1], c, 1, multiplicity)
                      for c in selected_midpoints(knotsv)]
        vertical = [Meshline(knotsv[0], knotsv[-1], c, 0, multiplicity)
                    for c in selected_midpoints(knotsu)]
        return horizontal + vertical

    def compute_enclosing_aabb(self, basis_candidates):
        """
        Computes the enclosing AABB for a collection of basis functions.

        :param basis_candidates: Dictionary containing basis functions (e.g., {hash: basis_function}).
        :return: The enclosing AABB as (min_u, min_v, max_u, max_v).
        """
        # Initialize bounds with extreme values
        if not basis_candidates:
            raise ValueError("Cannot compute bounds of an empty set of basis functions.")
        min_u, min_v = np.inf, np.inf
        max_u, max_v = -np.inf, -np.inf

        for basis in basis_candidates.values():
            # Assume each basis has a method `get_AABB()` that returns its AABB as (min_u, min_v, max_u, max_v)
            aabb = basis.get_AABB()

            # Update global minimum and maximum bounds for u and v
            min_u = min(min_u, aabb[0][0])  # Update minimum u
            min_v = min(min_v, aabb[1][0])  # Update minimum v
            max_u = max(max_u, aabb[0][1])  # Update maximum u
            max_v = max(max_v, aabb[1][1])  # Update maximum v

        # Return the enclosing AABB as a tuple
        return (min_u, min_v, max_u, max_v)

    def insert_line(self, meshline: Meshline) -> None:
        """
        Inserts a line in the mesh, splitting where necessary. Uses dictionaries for new_functions and functions_to_remove.
        """
        # Step 0: Merge meshlines and early exit if already present
        control_point_fields(self.basis_dic.values())
        meshline_already_exists, meshline = self.merge_meshlines(meshline)
        if meshline_already_exists:
            return

        # Step 1: Split basis functions affected by the new meshline
        new_functions = {}  # Hash -> Basisfunction
        function_ids_to_remove = []
        basis_candidates_hashes = list(self.bfs_rtree.intersection(self.flat_AABB(meshline.get_AABB())))
        basis_candidates = {
            bf_hash: basis
            for bf_hash in basis_candidates_hashes
            if (basis := self._basis_by_index.get(bf_hash))}

        for basis_hash, basis in basis_candidates.items():
            if meshline.splits_basis(basis):
                if meshline.number_of_knots_contained(basis) < meshline.multiplicity:
                    self.local_split(basis, meshline, new_functions, function_ids_to_remove)

        while function_ids_to_remove:
            bf_id = function_ids_to_remove.pop()
            basis = self.basis_dic.pop(bf_id)
            self.bfs_rtree.delete(id(basis), self.flat_AABB(basis.get_AABB()))
            self._basis_by_index.pop(id(basis), None)

        # step2 recursive test against old meshlines
        enclosing_aabb = (self.compute_enclosing_aabb(basis_candidates) if basis_candidates
                          else self.flat_AABB(meshline.get_AABB()))
        meshline_candidates_h_ids = list(self.mls_h_rtree.intersection(enclosing_aabb))

        # vertical meshlines
        meshline_candidates_v_ids = list(self.mls_v_rtree.intersection(enclosing_aabb))

        meshline_candidates_ids = meshline_candidates_h_ids + meshline_candidates_v_ids
        meshline_candidates = [
            self.meshlines_id[meshline_id] for meshline_id in meshline_candidates_ids
            if meshline_id in self.meshlines_id
        ]



        # Step 2: Recursive test against old meshlines
        while new_functions:
            basis_hash, basis = new_functions.popitem()
            split_more = False

            # Test against meshlines
            for old_meshline in meshline_candidates:
                if old_meshline.splits_basis(basis) and old_meshline.number_of_knots_contained(basis) < old_meshline.multiplicity:
                    split_more = True
                    self.local_split(basis, old_meshline, new_functions, function_ids_to_remove)
                    break

            if not split_more:
                # Add the basis function to the dictionary and RTree if it doesn't need further splitting
                self.basis_dic[basis_hash] = basis
                self._basis_by_index[id(basis)] = basis
                self.bfs_rtree.insert(id(basis), self.flat_AABB(basis.get_AABB()))


        # Step 3: Split elements affected by the new meshline
        new_elements = []
        element_candidates_ids = list(self.els_rtree.intersection(self.flat_AABB(meshline.get_AABB())))
        element_candidates = {
            element_id: element
            for element_id in element_candidates_ids
            if (element := self.elements_dic.get(element_id))}

        for element_id, element in element_candidates.items():
            if meshline.splits_element(element):
                split_val = meshline.constant_value
                split_elements = element.split(axis=meshline.axis, split_value=split_val)
                new_elements.extend(split_elements)
                self.els_rtree.delete(element_id, self.flat_AABB(element.get_AABB()))
                del self.elements_dic[element_id]

        for element in new_elements:
            element_id = id(element)
            self.elements_dic[element_id] = element
            self.els_rtree.insert(element_id, self.flat_AABB(element.get_AABB()))


        # old_elements = list(self.elements_dic.values())
        # new_elements = []
        # while old_elements:
        #     old_element = old_elements.pop()
        #     if meshline.splits_element(old_element):
        #         el1, el2 = old_element.split(axis=meshline.axis, split_value=meshline.constant_value)
        #         new_elements.append(el1)
        #         new_elements.append(el2)
        #         # self.els_rtree.insert(id(el1), self.flat_AABB(el1.get_AABB()))
        #         # self.els_rtree.insert(id(el2), self.flat_AABB(el2.get_AABB()))
        #     else:
        #         new_elements.append(old_element)
        #
        # el_dic = {}
        # for el in new_elements:
        #     el_dic[id(el)] = el
        # self.elements_dic = el_dic

        #self.get_connectivity()

    def local_split(self, basis, meshline, new_functions, function_ids_to_remove):
        """
        Handles splitting a single basis function into two parts due to a meshline.
        Updates dictionaries for new_functions and functions_to_remove.
        """
        b1, b2, alpha_1, alpha_2 = split_single_basis_function(meshline, basis)

        for b in (b1, b2):
            bh = b.basis_key
            if bh in self.basis_dic:
                self._update_old_basis_function(b, self.basis_dic)
            elif bh in new_functions:
                self._update_old_basis_function(b, new_functions)
            else:
                new_functions[bh] = b

        if basis.basis_key in self.basis_dic:
            function_ids_to_remove.append(basis.basis_key)


    @staticmethod
    def _update_old_basis_function(new_basis, basis_dict):
        """
        Updates the basis function in the given dictionary.
        """
        basis_hash = new_basis.basis_key
        basis_dict[basis_hash].update_weights(new_basis)


    def flat_AABB(self, aabb):
        # umin, vmin, umax, vmax
        return [aabb[0][0], aabb[1][0], aabb[0][1], aabb[1][1]]

    def __call__(self, xi, eta):
        """
        Evaluates the THBSpline at the point (xi, eta)

        :param xi: first component
        :param eta: secont component
        :return: L(xi, eta)
        """

        def where_uv(xi, eta):
            for el in self.elements:
                elAABB = el.get_AABB()
                xis = elAABB[0]
                etas = elAABB[1]
                if xis[0] <= xi <= xis[1] and etas[0] <= eta <= etas[1]:
                    return el

        el = where_uv(xi, eta)
        total = np.array([0.0,0.0,0.0])
        for f in el.supported_b_splines:
            total += f.cp * f(xi, eta)
        return total

    def merge_meshlines(self, meshline: Meshline) -> typing.Tuple[bool, Meshline]:
        """
        Tests the meshline against all currently stored meshlines, and combines, updates,
        and deletes meshlines as needed. Updates the RTree as necessary.
        Returns True if the meshline was already in the list, False otherwise.

        :param meshline: Meshline to test against previous meshlines.
        :return: (True, existing_meshline) if meshline already exists; (False, updated_meshline) otherwise.
        """
        meshlines_to_remove = []
        axis = meshline.axis
        const = float(meshline.constant_value)

        if axis == 1:  # Horizontal meshlines
            old_meshlines = self.meshlines_h.get(const, [])
            rtree = self.mls_h_rtree
        else:  # Vertical meshlines
            old_meshlines = self.meshlines_v.get(const, [])
            rtree = self.mls_v_rtree

        for old_meshline in old_meshlines:
            if not old_meshline._similar(meshline):
                continue  # Skip non-comparable meshlines

            if meshline == old_meshline:
                return True, meshline  # Exact match found

            if old_meshline.contains(meshline):
                # New meshline is contained within an old one
                if meshline.multiplicity > old_meshline.multiplicity:
                    if old_meshline.start == meshline.start and old_meshline.stop == meshline.stop:
                        meshlines_to_remove.append(old_meshline)  # Replace old meshline
                else:
                    return True, old_meshline

            elif old_meshline.overlaps(meshline):
                # Handle overlap by extending or merging
                if old_meshline.multiplicity < meshline.multiplicity:
                    old_bounds = self.flat_AABB(old_meshline.get_AABB())
                    rtree.delete(id(old_meshline), old_bounds)
                    old_meshline.start = min(old_meshline.start, meshline.start)
                    old_meshline.stop = max(old_meshline.stop, meshline.stop)
                    rtree.insert(id(old_meshline), self.flat_AABB(old_meshline.get_AABB()))
                elif old_meshline.multiplicity > meshline.multiplicity:
                    meshline.start = min(old_meshline.start, meshline.start)
                    meshline.stop = max(old_meshline.stop, meshline.stop)
                else:
                    meshline.start = min(old_meshline.start, meshline.start)
                    meshline.stop = max(old_meshline.stop, meshline.stop)
                    meshlines_to_remove.append(old_meshline)

        # Remove old meshlines and update RTree
        for old_meshline in meshlines_to_remove:
            constant_value = float(old_meshline.constant_value)
            if old_meshline.axis == 1:  # Horizontal meshlines
                old_meshlines = self.meshlines_h
                old_meshlines_id = self.meshlines_id
                rtree = self.mls_h_rtree
            else:  # Vertical meshlines
                old_meshlines = self.meshlines_v
                old_meshlines_id = self.meshlines_id
                rtree = self.mls_v_rtree

            old_meshlines[constant_value].remove(old_meshline)  # Remove from internal storage
            del old_meshlines_id[id(old_meshline)]
            rtree.delete(id(old_meshline), self.flat_AABB(old_meshline.get_AABB()))

        # Insert or update the new meshline in the RTree
        rtree.insert(id(meshline), self.flat_AABB(meshline.get_AABB()))  # Add to RTree

        # Add meshline to the corresponding internal list
        if axis == 1:
            self.meshlines_h.setdefault(const, []).append(meshline)
            self.meshlines_id[id(meshline)] = meshline
        else:
            self.meshlines_v.setdefault(const, []).append(meshline)
            self.meshlines_id[id(meshline)] = meshline

        return False, meshline


    def eval_displacement(self, xi, eta):
        """
        Evaluates the displacement at the point (xi, eta)

        :param xi: first component
        :param eta: second component
        :return: T(xi, eta)
        """

        def where_uv(xi, eta):
            for el in self.elements:
                if el.u_min <= xi <= el.u_max and el.v_min <= eta <= el.v_max:
                    return el

        el = where_uv(xi, eta)
        total = np.array([0.0, 0.0, 0.0])
        for f in el.supported_b_splines:
            total += f.displacements * f(xi, eta)
        return total

    def eval_scalar_field(self, xi, eta):
        """
        Evaluates the scalar field (e.g. temperature) at the point (xi, eta).

        :param xi: First parametric coordinate
        :param eta: Second parametric coordinate
        :return: Scalar value at (xi, eta)
        """

        def where_uv(xi, eta):
            for el in self.elements:
                if el.u_min <= xi <= el.u_max and el.v_min <= eta <= el.v_max:
                    return el
            raise ValueError(f"No element found for (xi, eta) = ({xi}, {eta})")

        el = where_uv(xi, eta)
        value = 0.0
        for bf in el.supported_b_splines:
            value += bf.scalar_value * bf(xi, eta)
        return value


if __name__ == "__main__":
    # Example data
    import numpy as np
    p = 2  # Grad
    ku_glob = 3*[0] + [1, 2, 3, 4, 5] + 3*[6]
    kv_glob = 3*[0] + [1, 2, 3, 4, 5] + 3*[6]
    cps = np.random.rand(64, 3)


    LR = LRSpline(p, ku_glob, kv_glob, cps)

    # Check knot vectors.
    print("Knotensequenz u:", LR.ku)
    print("Knotensequenz v:", LR.kv)


    # Optionale zusÃ¤tzliche Tests:
    # Check control-point counts against degree and knot vectors.
    expected_num_cps = (len(ku_glob) - p - 1) * (len(kv_glob) - p - 1)
    print("Erwartete Anzahl der Kontrollpunkte:", expected_num_cps)

    # Check the spline degree.
    print("Grad p:", LR.p)

    LR.get_connectivity()
    pass
