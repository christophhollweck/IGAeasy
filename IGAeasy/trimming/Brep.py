import numpy as np
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
from shapely import affinity
from shapely.geometry.polygon import orient

class BRep:
    """Parametric trim polygon; CCW denotes material, CW denotes a cutout.

    Set outerloop explicitly to orient a supplied polygon, including its holes.
    Otherwise its exterior orientation determines the role, as in legacy setups.
    """
    def __init__(self, Brep_id, bsplines=None, polygon=None, coarsen=False, outerloop=None):
        self.ID = Brep_id
        self.bsplines = bsplines if bsplines else []  # Default to empty list if no B-Splines provided
        self.polygon = polygon  # Polygon is directly passed if available
        self.outerloop = False  # Default orientation (not computed until BRep is complete)
        self.area = 0.0

        # If B-Splines are provided, calculate the polygon and its area and orientation
        if self.bsplines:
            self.get_polygon()
            self.compute_area_and_orientation()
        elif self.polygon is not None:
            self.compute_area_and_orientation()  # Calculate area and orientation directly from the polygon

        if coarsen == True:
            self.polygon = self.adaptive_sampling(self.polygon)
            self.compute_area_and_orientation()

        if outerloop is not None and self.polygon is not None:
            self.polygon = orient(self.polygon, sign=1. if outerloop else -1.)
            self.compute_area_and_orientation()


    def eval_BRep(self, num_points=100):
        """Polygonizes the BRep by sampling the B-Splines."""
        xlist = []
        ylist = []
        for Bspl in self.bsplines:
            uvec = np.linspace(0, 1, num_points)
            for u in uvec:
                xy = Bspl.obj.evaluate([[u]])[0]
                xlist.append(xy[0])
                ylist.append(xy[1])
        return xlist, ylist

    def get_polygon(self, num_points=100):
        """Calculates the polygon from the B-Splines."""
        xlist, ylist = self.eval_BRep(num_points)
        points = list(zip(xlist, ylist))

        # Create Shapely Polygon and store it
        self.polygon = Polygon(points)

    def compute_area_and_orientation(self):
        """Calculates the area and determines the orientation (CW or CCW)."""
        if self.polygon is not None:
            if not isinstance(self.polygon, Polygon) or self.polygon.is_empty or not self.polygon.is_valid:
                raise ValueError('BRep requires a nonempty valid Polygon')
            self.area = self.polygon.area  # Interior rings subtract from the enclosed area.
            self.outerloop = bool(self.polygon.exterior.is_ccw)


    def simplify_polygon_douglas_peucker(self, poly, tolerance=0.05):
        """Simplifies a polygon using the Douglas-Peucker algorithm."""
        simplified = LineString(poly.exterior.coords).simplify(tolerance, preserve_topology=True)
        return Polygon(simplified, holes=[ring.coords for ring in poly.interiors])


    def adaptive_sampling(self, poly, threshold=175, max_iter=20):
        """Reduces the number of points adaptively based on the angles between adjacent line segments."""

        points = list(poly.exterior.coords[:-1])  # Remove the last point (closed loop)
        prev_len = len(points)

        for j in range(max_iter):
            temp_points_iter = [points[0]]  # Keep the first point

            for i in range(0, len(points) - 2, 2):  # Iterate through the points except the last one
                p1 = points[i]
                p2 = points[i + 1]
                p3 = points[(i + 2) % len(points)]  # If i+2 is out of bounds, take the first point

                angle = self.calculate_angle(p1, p2, p3)

                if angle < threshold:
                    temp_points_iter.append(p2)  # Keep p2 if the angle is below the threshold

                temp_points_iter.append(p3)  # Always add p3

            if temp_points_iter[-1] != points[-1]:
                temp_points_iter.append(points[-1])  # Ensure the last point is retained

            current_len = len(temp_points_iter)
            if prev_len == current_len:
                break  # Stop if no points were removed

            prev_len = current_len
            points = temp_points_iter  # Update the points for the next iteration

        return Polygon(points + [points[0]], holes=[ring.coords for ring in poly.interiors])


    def calculate_angle(self, p1, p2, p3):
        """Calculates the angle between the line segments (p1->p2) and (p2->p3)."""

        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)

        # Vectors from p1 to p2 and from p2 to p3
        v1 = p2 - p1
        v2 = p3 - p2

        # Compute the angle
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = dot_product / (norm_v1 * norm_v2)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Angle in radians

        return 180 - np.degrees(angle)  # Convert to degrees

    def plot(self):
        if self.polygon:
            x, y = self.polygon.exterior.xy
            plt.plot(x, y)

        plt.show()

    # def get_linestring(self):
    #     if self.polygon:
    #         return self.polygon.exterior
    #     raise RuntimeError("BRep does not contain a valid polygon.")

    def get_linestring(self):
        if self.polygon:
            return LineString(self.polygon.exterior.coords)
        raise RuntimeError("BRep does not contain a valid polygon.")


    def rotate(self, angle_deg, origin='center'):
        """
        Rotates the polygon.

        Parameters
        ----------
        angle_deg : float
            Rotation angle in degrees (counter-clockwise).
        origin : str or tuple
            Center of rotation. E.g., 'center', 'centroid', or (x, y).
        """
        if self.polygon is None:
            raise ValueError("No polygon to rotate.")
        self.polygon = affinity.rotate(self.polygon, angle_deg, origin=origin)
        return self

    def translate(self, dx=0.0, dy=0.0):
        """
        Translates the polygon.

        Parameters
        ----------
        dx : float
            Shift in x-direction.
        dy : float
            Shift in y-direction.
        """
        if self.polygon is None:
            raise ValueError("No polygon to translate.")
        self.polygon = affinity.translate(self.polygon, xoff=dx, yoff=dy)
        return self

    def scale(self, sx=1.0, sy=1.0, origin='center'):
        """
        Scales the polygon.

        Parameters
        ----------
        sx : float
            Scaling factor in x-direction.
        sy : float
            Scaling factor in y-direction.
        origin : str or tuple
            Center of scaling.
        """
        if self.polygon is None:
            raise ValueError("No polygon to scale.")
        self.polygon = affinity.scale(self.polygon, xfact=sx, yfact=sy, origin=origin)
        return self

    def mirror_x(self, x=0.0):
        """
        Mirrors the polygon about a vertical axis (parallel to y-axis).

        Parameters
        ----------
        x : float
            x-coordinate of mirror axis.
        """
        return self.scale(sx=-1, sy=1, origin=(x, 0.0))

    def mirror_y(self, y=0.0):
        """
        Mirrors the polygon about a horizontal axis (parallel to x-axis).

        Parameters
        ----------
        y : float
            y-coordinate of mirror axis.
        """
        return self.scale(sx=1, sy=-1, origin=(0.0, y))
