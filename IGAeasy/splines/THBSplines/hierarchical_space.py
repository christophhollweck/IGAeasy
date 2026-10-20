import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as plp
import scipy.sparse as sp
from scipy.special import roots_legendre
import typing
from IGAeasy.splines.THBSplines.tensor_product_space import TensorProductSpace2D
import time
import pyvista as pv
from scipy.interpolate import RBFInterpolator
from IGAeasy.splines.control_point_fields import control_point_fields


class HierarchicalSpace():

    def __init__(self, ku, kv, p, cps, n_max_level=None, refinement_type='h'):
        """
        Initialize a hierarchical space with a base mesh and space over the
        given knot vectors

        :param knots: tensor product knot vectors
        :param degrees: correpsonding tensor product degrees
        :param dim: parametric dimension
        :param dim: controlpoints
        :param n_max_level: Maximum total number of levels, including the base
            level. None means no fixed limit. Levels are built only on demand.
        """
        if n_max_level is not None and (not isinstance(n_max_level, (int, np.integer)) or n_max_level < 1):
            raise ValueError("n_max_level must be a positive integer or None.")
        self.nlevels = 1
        self.n_max_level = n_max_level
        self.ku = ku
        self.kv = kv
        self.p = p
        self.refinement_type = refinement_type
        self.trimmed = False
        self.BReps = []

        self.spaces = [TensorProductSpace2D(ku, kv, p, cps, 0, refinement_type)]
        self.afunc_level = {0: set(range(self.spaces[0].nfuncs))}  # active functions on level
        self.dfunc_level = {0: set()}  # deactivated functions on level
        self.nfuncs_level = {0: self.spaces[0].nfuncs}
        self.nfuncs = self.nfuncs_level[0]

        self.aelem_level = {0: set(range(self.spaces[0].nelems))}  # active elements on level
        self.delem_level = {0: set()}  # deactivated elements on level
        self.nelems_level = {0: self.spaces[0].nelems}
        self.nel = self.spaces[0].nelems

        self.consecutive_matrix = []


        for f in self.spaces[0].basis:
            f.active = True
        for el in self.spaces[0].elements:
            el.active = True
        self.basis = [f for f in self.spaces[0].basis]
        self.elements = [el for el in self.spaces[0].elements]

        for id, b in enumerate(self.basis):
            b.id = id
        for id, e in enumerate(self.elements):
            e.id = id
            e._mesh_elements = self.elements

    def get_funcs_ibc(self, level):
        """
        Compute IBC-valid basis functions on the given level:
        - Remove functions fully outside all outer loops.
        - Remove functions fully inside any inner loop.
        - Remove functions whose support intersects any loop boundary.
        """

        from shapely.geometry import box
        tol = 1e-13

        if level not in self.afunc_level:
            return []

        result = set()

        # Split BReps into outer and inner loops
        outerloops = [b.polygon for b in self.BReps if b.outerloop]
        innerloops = [b.polygon for b in self.BReps if not b.outerloop]

        for funcnum_loc in self.afunc_level[level]:
            fun = self.spaces[level].basis[funcnum_loc]

            u_min, u_max = fun.knots_u[0], fun.knots_u[-1]
            v_min, v_max = fun.knots_v[0], fun.knots_v[-1]

            # Support box of the basis function
            func_box = box(u_min - tol, v_min - tol, u_max + tol, v_max + tol)

            # ------------------------------------------------------------
            # 1 fully outside all outer loops
            # ------------------------------------------------------------
            outside_outer = True
            for poly in outerloops:
                if poly.intersects(func_box):
                    outside_outer = False
                    break

            if outside_outer:
                continue

            # ------------------------------------------------------------
            # 2 fully inside any inner loop
            # ------------------------------------------------------------
            inside_inner = False
            for poly in innerloops:
                if poly.contains(func_box):
                    inside_inner = True
                    break

            if inside_inner:
                continue

            # ------------------------------------------------------------
            # 3 function cuts a loop boundary
            #    intersects but is not fully contained
            # ------------------------------------------------------------
            cuts_loop = False
            for poly in outerloops + innerloops:
                if poly.intersects(func_box) and not poly.contains(func_box):
                    cuts_loop = True
                    break

            if cuts_loop:
                continue

            result.add(funcnum_loc)

        return list(result)

    def get_active_funcs(self, level):
        """
        Mark all active functions on the given level:
        - Active if the support lies inside or intersects any outer loop.
        - Not active if fully contained in any inner loop.
        """

        from shapely.geometry import box
        tol = 1e-13

        if level not in self.afunc_level:
            return []

        active = set()

        outerloops = [b.polygon for b in self.BReps if b.outerloop]
        innerloops = [b.polygon for b in self.BReps if not b.outerloop]

        for funcnum_loc in self.afunc_level[level]:
            fun = self.spaces[level].basis[funcnum_loc]

            u_min, u_max = fun.knots_u[0], fun.knots_u[-1]
            v_min, v_max = fun.knots_v[0], fun.knots_v[-1]

            func_box = box(u_min - tol, v_min - tol, u_max + tol, v_max + tol)

            # Check intersection with outerloops
            on_or_inside_outer = False
            for poly in outerloops:
                if poly.intersects(func_box):
                    on_or_inside_outer = True
                    break

            if not on_or_inside_outer:
                continue

            # Remove if fully inside an innerloop
            inside_inner = False
            for poly in innerloops:
                if poly.contains(func_box):
                    inside_inner = True
                    break

            if inside_inner:
                continue

            active.add(funcnum_loc)

        return list(active)

    def get_funcs_in_rectangle(self, level, AABBs):
        """
        returns functions with full support within the given rectangle
        :param level: level on which functions should be searched
        :param AABB: Rectangle defined as AABB ([[u_min, u_max],[v_min, v_max]])
        """
        funcs = set()
        tol = 1e-13
        if level in self.afunc_level:
            for AABB in AABBs:
                for funcnum_loc in self.afunc_level[level]:
                    fun = self.spaces[level].basis[funcnum_loc]
                    u_min = fun.knots_u[0]
                    u_max = fun.knots_u[-1]
                    v_min = fun.knots_v[0]
                    v_max = fun.knots_v[-1]
                    if (u_min >= AABB[0][0] - tol and u_max <= AABB[0][1] + tol and
                            v_min >= AABB[1][0] - tol and v_max <= AABB[1][1] + tol):
                        funcs.add(funcnum_loc)

        return funcs

    def get_funcs_from_loop(self, level, brep):
        """
        Returns all basis functions whose support is intersected by the given BRep line,
        but excludes those fully contained within the BRep polygon.

        :param level: Level on which functions should be searched.
        :param brep: Brep (Shapely LineString or Polygon)
        """
        from shapely.geometry import box
        funcs = set()
        tol = 1e-13

        if level not in self.afunc_level:
            return list(funcs)

        for funcnum_loc in self.afunc_level[level]:
            fun = self.spaces[level].basis[funcnum_loc]
            u_min, u_max = fun.knots_u[0], fun.knots_u[-1]
            v_min, v_max = fun.knots_v[0], fun.knots_v[-1]

            # Create a box representing the support of the basis function
            func_box = box(u_min - tol, v_min - tol, u_max + tol, v_max + tol)

            # Check if the support intersects the BRep line
            if brep.polygon.intersects(func_box):
                # Exclude supports fully contained within the polygon
                if not brep.polygon.contains(func_box):
                    funcs.add(funcnum_loc)

        return list(funcs)

    def get_funcs_from_point(self, level, listofpoints):
        """
        Efficiently retrieves all active basis function objects containing the specified points for a specific level using an RTree.

        :param level: The hierarchical level of basis functions to consider.
        :param listofpoints: A list of points, where each point is defined as [u, v].
        :return: A list of unique active basis function objects containing the points.
        """
        funcs = set()

        if level in self.afunc_level:
            for point in listofpoints:
                u, v = point
                for funcnum_loc in self.afunc_level[level]:
                    fun = self.spaces[level].basis[funcnum_loc]
                    u_min = fun.knots_u[0]
                    u_max = fun.knots_u[-1]
                    v_min = fun.knots_v[0]
                    v_max = fun.knots_v[-1]

                    if (u_min <= u <= u_max and v_min <= v <= v_max):
                        funcs.add(funcnum_loc)

        return list(funcs)

    def get_funcs_from_point_local(self, level, listofpoints, alpha):
        """
        Retrieves unique active basis function indices from a specific level,
        only if at least one provided point falls within the central region of the function's domain.
        The central region is defined as a box centered in the function's domain,
        with half-widths determined by alpha (recommended 0 < alpha < 0.5).

        :param level: The hierarchical level of the basis functions.
        :param listofpoints: A list of points, each defined as [u, v].
        :param alpha: Fraction of the overall domain width from the center for the central region.
        :return: A list of unique active basis function indices whose central region contains at least one point.
        """
        funcs = set()

        # Check if level exists in the active function levels
        if level in self.afunc_level:
            # Iterate over each active basis function index at the given level
            for funcnum_loc in self.afunc_level[level]:
                fun = self.spaces[level].basis[funcnum_loc]

                # Determine overall domain boundaries for the function
                u_min = fun.knots_u[0]
                u_max = fun.knots_u[-1]
                v_min = fun.knots_v[0]
                v_max = fun.knots_v[-1]

                # Compute the center of the domain
                u_center = (u_min + u_max) / 2
                v_center = (v_min + v_max) / 2

                # Determine half-width of the central region using alpha
                # This defines a box [u_center - alpha*(u_max-u_min), u_center + alpha*(u_max-u_min)]
                # and similarly in v direction.
                half_width_u = alpha * (u_max - u_min)
                half_width_v = alpha * (v_max - v_min)
                u_left = u_center - half_width_u
                u_right = u_center + half_width_u
                v_left = v_center - half_width_v
                v_right = v_center + half_width_v

                # Check if any point falls within the overall bounds and the central region.
                # First, a quick check if the point is in the overall bounds (cheaper test).
                # If true, then check if it is within the smaller central region.
                for point in listofpoints:
                    u, v = point
                    if not (u_min <= u <= u_max and v_min <= v <= v_max):
                        continue
                    if u_left <= u <= u_right and v_left <= v <= v_right:
                        funcs.add(funcnum_loc)
                        break  # No need to check further points for this function

        return list(funcs)

    def refine(self, marked_functions: dict):
        """
        Refines the hierarchical space.
        :param marked_functions: marked functions of level l to refine
        """
        # Validate the entire request before changing any active state.
        for level, indices in marked_functions.items():
            if not isinstance(level, (int, np.integer)) or not 0 <= level < self.nlevels:
                raise ValueError("Refinement level is outside the available hierarchy.")
            if any(index not in self.afunc_level[level] for index in indices):
                raise ValueError("Only active basis functions can be refined.")

        control_point_fields(self.basis)  # Validate field shapes before mutation.
        required_levels = max((level + 2 for level, indices in marked_functions.items() if len(indices)),
                              default=self.nlevels)
        self.add_level(required_levels)

        # contains all deactivated functions during refinement, even when the functions
        # are not explicitely marked e.g. when a functions doesnt contain a single
        # element of the same level anymore. Gets updated within the algorithm.
        deactivated_functions = {key: set() for key in range(self.nlevels)}
        afunc_old = {key: value.copy() for key, value in self.afunc_level.items()}
        activated_functions =  {key: set() for key in range(self.nlevels)}

        for level in marked_functions.keys():
            funcs2deact = marked_functions[level]
            basis = self.spaces[level].basis
            elements = self.spaces[level].elements
            els_list = set()

            # deactivate marked function
            for index in funcs2deact:
                # print(index)
                b = basis[index]
                b.active = False
                if b.loc_id in self.afunc_level[level]:
                    self.afunc_level[level].remove(b.loc_id)
                self.dfunc_level[level].add(b.loc_id)
                deactivated_functions[level].add(b.loc_id)

                # deactivate all elements in the support of deactivated functions
                for el in self.spaces[b.level].support_by_basis[b.loc_id]:
                    el.active = False
                    if el.loc_id in self.aelem_level[level]:
                        self.aelem_level[level].remove(el.loc_id)
                    self.delem_level[level].add(el.loc_id)

                    # collect element object in list
                    els_list.add(el)
                    # activate children of deactivated elements if it hasnt been already deactivated
                    for child in el.children:
                        if child.active == None:
                            child.active = True
                            self.aelem_level[level + 1].add(child.loc_id)

                # activate children of basis function if at least one element of the childrens support
                # is still active
                for child in b.children:
                    boollist_fun = [el.active for el in self.spaces[child.level].support_by_basis[child.loc_id]]
                    if any(boollist_fun):
                        child.active = True
                        self.afunc_level[level + 1].add(child.loc_id)
                        activated_functions[level + 1].add(child.loc_id)
                    # if all elements of the child we aim to active have been deactivated through previous
                    # refinements, the child is activated and directely deactivated and replaced by its children
                    else:
                        self.dfunc_level[level + 1].add(child.loc_id)
                        deactivated_functions[level + 1].add(child.loc_id)
                        for childchild in child.children:
                            childchild.active = True
                            self.afunc_level[level + 2].add(childchild.loc_id)
                            activated_functions[level + 2].add(childchild.loc_id)



            funs = set()
            for el in els_list:
                for fun in el.supported_b_splines_init:
                    funs.add(fun)
                    fun.trunc = True

            # clear space; deactivate functions of level l if all elements of level l
            # in its support are deactivated and replace it by its children
            for fun in funs:
                boollist_el = [el.active for el in self.spaces[fun.level].support_by_basis[fun.loc_id]]
                # Execute the code block if all elements in boollist_el are False
                if all(not val for val in boollist_el) and fun.active:
                    fun.active = False
                    self.afunc_level[level].remove(fun.loc_id)
                    self.dfunc_level[level].add(fun.loc_id)
                    deactivated_functions[level].add(fun.loc_id)
                    for child in fun.children:
                        child.active = True
                        self.afunc_level[level + 1].add(child.loc_id)
                        activated_functions[level + 1].add(child.loc_id)



        self.update_cps(deactivated_functions, afunc_old, activated_functions)

        # collect active functions and elements
        basis = []
        elements = []

        for level in range(self.nlevels):
            afuncs = self.afunc_level[level]
            aels = self.aelem_level[level]
            basis_lvl = self.spaces[level].basis
            elements_lvl = self.spaces[level].elements
            for afunc in afuncs:
                basis.append(basis_lvl[afunc])
            for ael in aels:
                elements.append(elements_lvl[ael])

        self.basis = basis
        self.elements = elements
        self.nfuncs = len(basis)
        self.nel = len(elements)

        self.update_connectivity()


    def update_connectivity(self):
        """
        Updates the connectivity btw. elements and functions.
        """

        # update connectivity for active and truncated functions
        for element in self.elements:
            element.supported_b_splines = {b for b in element.supported_b_splines_init if b.active}
        for function in self.basis:
            function.elements_of_support = []


        for fun in self.basis:
            active_elements = set()
            if fun.trunc:
                active_elements = self.recursive_search(fun, active_elements)
                for el in active_elements:
                    el.supported_b_splines.add(fun)


        for element in self.elements:
            for b in element.supported_b_splines:
                b.elements_of_support.append(element)

        for id, b in enumerate(self.basis):
            b.id = id
        for id, e in enumerate(self.elements):
            e.id = id
            e._mesh_elements = self.elements


    def recursive_search(self, f, active_elements):
        """
        collects all active elements of all levels within the support of a active function
        """
        # add active children of support
        for e in self.spaces[f.level].support_by_basis[f.loc_id]:
            if e.active:
                active_elements.add(e)

        # Recursive loop through children
        for child in f.children:
            if not child.active:
                for e in self.spaces[child.level].support_by_basis[child.loc_id]:
                    if e.active:
                        active_elements.add(e)

                if child.trunc:
                    self.recursive_search(child, active_elements)

        return active_elements


    def add_level(self, n_max_level):
        """
        Ensure the requested total number of levels exists and is connected.
        Existing levels and their parent-child links are preserved.
        """

        if not isinstance(n_max_level, (int, np.integer)) or n_max_level < 1:
            raise ValueError("Requested level count must be a positive integer.")
        if self.n_max_level is not None and n_max_level > self.n_max_level:
            raise ValueError("Refinement would exceed n_max_level.")
        while self.nlevels < n_max_level:
            parent_level = self.nlevels - 1
            refined_space, c = self.spaces[-1].refine()
            self.nlevels += 1
            self.spaces.append(refined_space)
            self.consecutive_matrix.append(c)

            self.afunc_level[self.nlevels - 1] = set()
            self.dfunc_level[self.nlevels - 1] = set()
            self.nfuncs_level[self.nlevels - 1] = self.spaces[-1].nfuncs

            self.aelem_level[self.nlevels - 1] = set()
            self.delem_level[self.nlevels - 1] = set()
            self.nelems_level[self.nlevels - 1] = self.spaces[-1].nelems

            self.connect_levels(start_level=parent_level)


    def connect_levels(self, start_level=0):
        """
        Connects the different refinement levels by connecting parent elements/ functions
        with their respective children
        """
        refinement_type = self.refinement_type
        for lvl in range(start_level, self.nlevels - 1):
            elements_lvl0 = self.spaces[lvl].elements
            unique_ku0 = self.spaces[lvl].unique_ku
            unique_kv0 = self.spaces[lvl].unique_kv
            n_el_u_0 = len(unique_ku0) - 1
            n_el_v_0 = len(unique_kv0) - 1

            elements_lvl1 = self.spaces[lvl + 1].elements
            unique_ku1 = self.spaces[lvl + 1].unique_ku
            unique_kv1 = self.spaces[lvl + 1].unique_kv
            n_el_u_1 = len(unique_ku1) - 1
            n_el_v_1 = len(unique_kv1) - 1

            elnum = 0
            for j in range(n_el_v_0):
                for i in range(n_el_u_0):
                    if refinement_type in {'k', 'h'}:
                        # k- and h-refinement: 4 child elements
                        children = [
                            (2 * j) * n_el_u_1 + (2 * i),
                            (2 * j) * n_el_u_1 + (2 * i + 1),
                            (2 * j + 1) * n_el_u_1 + (2 * i),
                            (2 * j + 1) * n_el_u_1 + (2 * i + 1)
                        ]
                    else:
                        #p-refinement: 1 child element
                        children = [j * n_el_u_1 + i]

                    elements_lvl0[elnum].children = [elements_lvl1[child] for child in children]
                    elnum += 1

            basis_functions0 = self.spaces[lvl].basis
            basis_functions1 = self.spaces[lvl + 1].basis
            consecutive_matrix = self.consecutive_matrix[lvl]

            for i, bf0 in enumerate(basis_functions0):
                indices_bf1 = consecutive_matrix.rows[i]
                bf0.children = [basis_functions1[j] for j in indices_bf1]
                bf0.coefficient = np.array(consecutive_matrix.data[i])


    def update_cps(self, marked_functions, afunc_old, activated_functions):
        """Transfer the current active geometry through the old truncation tree.

        Inactive control points may be stale after deformation. Reconstruct their
        coefficients from active ancestors instead of using stored values.
        Previously active children already contain their own coefficients;
        old truncation removes their parents' contribution at those children.
        """
        old_basis = [self.spaces[level].basis[i] for level, indices in afunc_old.items() for i in indices]
        for field, shape in control_point_fields(old_basis).items():
            width = int(np.prod(shape)) if shape else 1
            coefficients = np.zeros((len(self.spaces[0].basis), width))
            for index in afunc_old[0]:
                coefficients[index] = np.asarray(getattr(self.spaces[0].basis[index], field, np.zeros(shape))).reshape(-1)
            for level in range(self.nlevels):
                for index in self.afunc_level[level]:
                    value = coefficients[index].copy()
                    setattr(self.spaces[level].basis[index], field, float(value[0]) if not shape else value.reshape(shape))
                if level + 1 == self.nlevels:
                    break
                transferred = self.consecutive_matrix[level].T @ coefficients
                for index in afunc_old[level + 1]:
                    transferred[index] = np.asarray(getattr(self.spaces[level + 1].basis[index], field, np.zeros(shape))).reshape(-1)
                coefficients = transferred

    def __call__(self, xi, eta):
        """
        Evaluates the THBSpline at the point (xi, eta)

        :param xi: first component
        :param eta: second component
        :return: T(xi, eta)
        """

        def where_uv(xi, eta):
            for el in self.elements:
                if el.u_min <= xi <= el.u_max and el.v_min <= eta <= el.v_max:
                    return el

        el = where_uv(xi, eta)
        total = np.array([0.0,0.0,0.0])
        for f in el.supported_b_splines:
            total += f.cp * f(xi, eta)
        return total

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
        total = np.array([0.0,0.0,0.0])
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

    def rigid_body_rotation(self, angles, P):
        """
           Rotates a 3D vector around the x, y, and z axes by given angles (in degrees).

           :param vector: The vector to be rotated (as a list or numpy array).
           :param angles: The rotation angles around the x, y, and z axes in degrees (as a tuple or list: [angle_x, angle_y, angle_z]).
           :return: The rotated vector as a numpy array.
           """

        # Convert angles from degrees to radians
        angle_x, angle_y, angle_z = np.radians(angles)

        # Rotation matrix for the x-axis
        rotation_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])

        # Rotation matrix for the y-axis
        rotation_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])

        # Rotation matrix for the z-axis
        rotation_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix in the order x -> y -> z
        rotation_matrix = rotation_z @ rotation_y @ rotation_x

        # Calculate the rotated vector
        P_rot = rotation_matrix @ P
        for bf in self.basis:
            bf.cp = rotation_matrix @ bf.cp

        return P_rot

    def rotate_geometry(self, angles):

        """
                   Rotates the geometry around the x, y, and z axes by given angles (in degrees).

                   :param angles: The rotation angles around the x, y, and z axes in degrees (as a tuple or list: [angle_x, angle_y, angle_z]).
                   """

        # Convert angles from degrees to radians
        angle_x, angle_y, angle_z = np.radians(angles)

        # Rotation matrix for the x-axis
        rotation_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])

        # Rotation matrix for the y-axis
        rotation_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])

        # Rotation matrix for the z-axis
        rotation_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix in the order x -> y -> z
        rotation_matrix = rotation_z @ rotation_y @ rotation_x

        for bf in self.basis:
            bf.cp = rotation_matrix @ bf.cp


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from mayavi import mlab
    import time
    import  splinepy


    def get_Bspl2D(n_elU, n_elV, ax, ay, ximax, etamax, conti, degrees):

        # knots to insert
        uins = np.linspace(0, ximax, n_elU + 1)
        uins = uins[1:-1]
        vins = np.linspace(0, etamax, n_elV + 1)
        vins = vins[1:-1]

        if len(degrees) != 2 or degrees[0] != degrees[1]:
            raise ValueError('Only 2D Geometries and homogeneous degrees in u and v-direction supported')
        # degree
        p = degrees[0]

        # knotvectors
        U = [0, 0, ximax, ximax]
        V = [0, 0, etamax, etamax]

        # coords(x, y)
        cps0 = np.array([
            [0., 0., 0.],
            [ax, 0., 0.],
            [0, ay, 0.],
            [ax, ay, 0.]
        ])

        # create Bspl Surface
        Bspl = splinepy.BSpline(
            degrees=[1, 1],
            knot_vectors=[U, V],
            control_points=cps0
        )

        # Order Elevation
        for i in range(p - 1):
            Bspl.elevate_degrees([0, 1])

        # after k-refinement, reduce continuity to desired value
        # multiplicity of knot (e.g. m=2 -> double knot values)
        m = p - conti
        for i in range(m):
            # Knot Insertion
            if len(uins) > 0:
                Bspl.insert_knots(0, uins)
            if len(vins) > 0:
                Bspl.insert_knots(1, vins)

        return Bspl


    ku = [0, 0, 0, 0.5, 1, 1, 1]
    kv = [0, 0, 0, 1, 1, 1]
    p = 2
    conti = p - 1
    n_max_level = 3

    cps = np.array(
        [
            [-1, 0, 1],
            [-1, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [-1, 0.5, 1],
            [-1, 0.5, 0],
            [1, 0.5, 0],
            [1, 0.5, 1],
            [-1, 1, 1],
            [-1, 1, 0],
            [1, 1, 0],
            [1, 1, 1]
        ]
    )

    Bspl = splinepy.BSpline(
        degrees=[p, p],
        knot_vectors=[ku, kv],
        control_points=cps,
    )

    Bspl.insert_knots(0, [0.25, 0.75])
    Bspl.insert_knots(1, [0.25, 0.5, 0.75])

    # n_elU = 6
    # n_elV = 6
    # p = 2
    # ax = 1
    # ay = 1
    # ximax = 1
    # etamax = 1
    # conti = p - 1
    # degrees = [p, p]
    # n_max_level = 3
    #
    # Bspl = get_Bspl2D(n_elU, n_elV, ax, ay, ximax, etamax, conti, degrees)

    cps = Bspl.cps
    ku = Bspl.knot_vectors[0]
    kv = Bspl.knot_vectors[1]

    # measure time
    start_time = time.time()

    T = HierarchicalSpace(ku, kv, p, cps, n_max_level)

    # refine_area = {0: [[[0, 0.7], [0, 0.7]]]}
    # for level, AABBs in refine_area.items():
    #     list_of_funcs = T.get_funcs_in_rectangle(level, AABBs)
    #     T.refine({level: list_of_funcs})

    refine_area = {0: [[[0, 0.5], [0, 1]]]}
    for level, AABBs in refine_area.items():
        list_of_funcs = T.get_funcs_in_rectangle(level, AABBs)
        T.refine({level: list_of_funcs})

    refine_area = {0: [[[0.5, 1], [0, 1]]]}
    for level, AABBs in refine_area.items():
        list_of_funcs = T.get_funcs_in_rectangle(level, AABBs)
        T.refine({level: list_of_funcs})

    refine_area = {1: [[[0.5, 1], [0, 1]]]}
    for level, AABBs in refine_area.items():
        list_of_funcs = T.get_funcs_in_rectangle(level, AABBs)
        T.refine({level: list_of_funcs})

    refine_area = {1: [[[0.25, 0.75], [0.25, 0.75]]]}
    for level, AABBs in refine_area.items():
        list_of_funcs = T.get_funcs_in_rectangle(level, AABBs)
        T.refine({level: list_of_funcs})

    end_time = time.time()
    execution_time = end_time - start_time
    print("AusfÃƒÆ’Ã‚Â¼hrungszeit:", execution_time, "Sekunden")

    T.plot_elements()
    T.plot_basis()
    T.plot_surface()
