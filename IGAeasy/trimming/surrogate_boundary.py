from shapely.geometry import LineString, Point
import numpy as np
from IGAeasy.numerical_integration.gauss_sbm import gauss_point_sbm
from IGAeasy.loads import LineLoad
from IGAeasy.constraints.dirichlet_weak import WeakLineConstraint

def gauss_legendre(n):
    """Returns Gauss-Legendre points and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(n)

class ActiveEdge:
    def __init__(self, line: LineString, is_horizontal: bool, side: int, ngp: int, cell):
        """
        Represents an active edge (surrogate boundary) of a trimmed element.

        Parameters:
        - line (LineString): The shifted line used for integration.
        - is_horizontal (bool): Orientation of the line.
        - side (int): Side index (0=bottom, 1=right, 2=top, 3=left).
        - ngp (int): Number of Gauss Points.
        - cell: The parent cell associated with this edge.
        """
        self.line = line
        self.is_horizontal = is_horizontal
        self.side = side
        self.ngp = ngp
        self.cell = cell

        self.dirichlet_data = []
        self.neumann_data = []
        self.surrogate_data =[]
        self._compute_integration_data()


    def _compute_integration_data(self):
        """Compute integration points, weights, tangents and normals on the surrogate edge."""
        self.surrogate_data = []
        self.dirichlet_data = []
        self.neumann_data = []
        xi, wi = gauss_legendre(self.ngp)
        length = self.line.length

        # p0 = np.array(self.line.interpolate(0).coords[0])
        # p1 = np.array(self.line.interpolate(length).coords[0])
        # midpoint = 0.5 * (p0 + p1)

        # Define tangent and normal depending on side ID
        if self.side == 0:  # bottom
            t = np.array([1.0, 0.0])  # horizontal (→)
            n = np.array([0.0, -1.0])  # normal down
        elif self.side == 1:  # right
            t = np.array([0.0, 1.0])  # vertical (↑)
            n = np.array([1.0, 0.0])  # normal right
        elif self.side == 2:  # top
            t = np.array([1.0, 0.0])  # horizontal (→)
            n = np.array([0.0, 1.0])  # normal up
        elif self.side == 3:  # left
            t = np.array([0.0, 1.0])  # vertical (↑)
            n = np.array([-1.0, 0.0])  # normal left
        else:
            raise ValueError(f"Unknown side ID: {self.side}")

        for x, w in zip(xi, wi):
            s = 0.5 * (x + 1) * length
            pt = self.line.interpolate(s)
            gw = 0.5 * w * length
            self.surrogate_data.append(gauss_point_sbm(pt.x, pt.y, s, gw, n, t))


    def assign_bc(self, bc: list):
        """
        Assigns the closest BC object (Dirichlet or Neumann) to each integration point on this edge.

        :param bc: List of LineLoad or WeakLineConstraint objects with a .line attribute
        """
        self.dirichlet_data = []
        self.neumann_data = []
        h = min([self.cell.u_max - self.cell.u_min, self.cell.v_max - self.cell.v_min])

        for gp in self.surrogate_data:
            p = Point(gp.u, gp.v)

            min_dist = np.inf
            closest_bc = None

            for b in bc:
                dist = b.line.distance(p)
                if dist < min_dist and dist <= h:
                    min_dist = dist
                    closest_bc = b

            if closest_bc is not None:
                gp.closest_line = closest_bc

                # classify and store
                if isinstance(closest_bc, LineLoad):
                    self.neumann_data.append(gp)
                elif isinstance(closest_bc, WeakLineConstraint):
                    self.dirichlet_data.append(gp)
            else:
                gp.closest_line = None

    # def _find_closest_brep(self):
    #     """Find the BREP with the **minimal** distance to any integration point."""
    #     if len(self.brep_candidates) == 1:
    #         self.closest_brep = self.brep_candidates[0]
    #         return
    #
    #     min_dist = np.inf
    #     closest = None
    #     for brep in self.brep_candidates:
    #         target = brep.get_linestring()
    #         distances = [
    #             pnt.distance(target.interpolate(target.project(pnt)))
    #             for pnt, _ in self.integration_points
    #         ]
    #         min_dist_candidate = min(distances)
    #         if min_dist_candidate < min_dist:
    #             min_dist = min_dist_candidate
    #             closest = brep
    #
    #     self.closest_brep = closest


    # def _compute_closest_points(self):
    #     """Project integration points on the closest BREP."""
    #     target = self.closest_brep.get_linestring()
    #     self.closest_points = []
    #     self.difference_vectors = []
    #
    #     for pnt, _ in self.integration_points:
    #         s = target.project(pnt)  # line parameter (distance along the line)
    #         closest = target.interpolate(s)
    #         self.closest_points.append((closest, s))
    #
    #         dx = closest.x - pnt.x
    #         dy = closest.y - pnt.y
    #         self.difference_vectors.append(np.array([dx, dy]))


    # def _compute_brep_normals_and_tangents(self):
    #     """Compute normals and tangents at the closest points on the true boundary."""
    #     target = self.closest_brep.get_linestring()
    #     eps = 1e-10
    #
    #     for pt, s in self.closest_points:
    #         s = target.project(pt)
    #         s0 = max(0.0, s - eps)
    #         s1 = min(target.length, s + eps)
    #         p0 = np.array(target.interpolate(s0).coords[0])
    #         p1 = np.array(target.interpolate(s1).coords[0])
    #         t = p0 - p1
    #         t /= np.linalg.norm(t)
    #         n = np.array([-t[1], t[0]])
    #         self.tangents_brep.append(t)
    #         self.normals_brep.append(n)



def assign_bc_to_active_sides_sbm(RS, bc_list):
    """
    Assign boundary conditions to all active sides of active elements.

    Parameters:
    - elements: List of elements (must have .active and .active_sides)
    - bc_list: List of BC objects (LineLoad, WeakLineConstraint, ...)
    """
    for element in RS.elements:
        if getattr(element, 'active', True) and not element.void and getattr(element, 'active_sides', []):
            for active_side in element.active_sides:
                active_side.assign_bc(bc_list)

def find_closest_brep_for_element(BReps, el):
    p_el = Point((el.u_min + el.u_max) / 2, (el.v_min + el.v_max) / 2)

    min_dist = float('inf')
    closest_brep = None

    for brep in BReps:
        if brep.polygon is None:
            continue
        linestring = brep.get_linestring()
        # Project element point onto BRep boundary curve
        proj_dist = linestring.project(p_el)
        closest_point = linestring.interpolate(proj_dist)
        dist = p_el.distance(closest_point)
        # Debug output
        # print(f"BRep ID={brep.ID}, distance={dist}")
        if dist < min_dist:
            min_dist = dist
            closest_brep = brep

    return closest_brep

def opposite_side(sidenum):
    """Returns the opposite side number."""
    return (sidenum + 2) % 4

def get_neighbor_center(el, sidenum, shift_count=1):
    """Returns the center of a neighbor element shifted in given direction."""
    hx = el.u_max - el.u_min
    hy = el.v_max - el.v_min

    cx = (el.u_min + el.u_max) / 2
    cy = (el.v_min + el.v_max) / 2

    if sidenum == 0:  # bottom
        cy -= shift_count * hy
    elif sidenum == 1:  # right
        cx += shift_count * hx
    elif sidenum == 2:  # top
        cy += shift_count * hy
    elif sidenum == 3:  # left
        cx -= shift_count * hx
    else:
        raise ValueError(f"Invalid side number: {sidenum}")

    return (cx, cy)
