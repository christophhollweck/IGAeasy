import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from IGAeasy.trimming import Brep


def circle(center_x, center_y, radius, brep_id, outerloop: bool, num_points=51):
    """Creates a circular polygon."""
    angles = np.linspace(0+0.3, 2 * np.pi+0.3, num_points, endpoint=False)
    if not outerloop:
        angles = np.flip(angles)

    x_coords = center_x + radius * np.cos(angles)
    y_coords = center_y + radius * np.sin(angles)
    polygon = Polygon(zip(x_coords, y_coords))
    return Brep.BRep(brep_id, polygon=polygon)


def rectangle(x_min, y_min, x_max, y_max, brep_id, outerloop: bool):
    """Creates a rectangular polygon."""
    coords = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    if not outerloop:
        coords.reverse()
    polygon = Polygon(coords)
    return Brep.BRep(brep_id, polygon=polygon)


def ellipse(center_x, center_y, radius_x, radius_y, brep_id, outerloop: bool, num_points=51):
    """Creates an elliptical polygon."""
    angles = np.linspace(0.033, 2 * np.pi + 0.033, num_points, endpoint=False)
    if not outerloop:
        angles = np.flip(angles)

    x_coords = center_x + radius_x * np.cos(angles)
    y_coords = center_y + radius_y * np.sin(angles)

    polygon = Polygon(zip(x_coords, y_coords))
    return Brep.BRep(brep_id, polygon=polygon)


def regular_polygon(center_x, center_y, radius, num_sides, brep_id, outerloop: bool):
    """Creates a regular n-sided polygon."""
    angles = np.linspace(0.033, 2 * np.pi + 0.033, num_sides, endpoint=False)
    if not outerloop:
        angles = np.flip(angles)

    x_coords = center_x + radius * np.cos(angles)
    y_coords = center_y + radius * np.sin(angles)

    polygon = Polygon(zip(x_coords, y_coords))
    return Brep.BRep(brep_id, polygon=polygon)


def heart(center_x, center_y, size, brep_id, outerloop: bool, num_points=400):
    """Creates a heart-shaped polygon.

    The polygon orientation is CCW if outerloop=True, else CW.
    """
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    # Parametric heart curve
    x_coords = 16 * np.sin(t) ** 3
    y_coords = (13 * np.cos(t) - 5 * np.cos(2 * t) -
                2 * np.cos(3 * t) - np.cos(4 * t))

    # Normalize by max extent so that 'size' behaves like a radius
    x_coords = size * x_coords / 16.0
    y_coords = size * y_coords / 16.0

    # Shift to center
    x_coords = center_x + x_coords
    y_coords = center_y + y_coords

    # Enforce orientation
    if outerloop:
        x_coords = np.flip(x_coords)
        y_coords = np.flip(y_coords)

    polygon = Polygon(zip(x_coords, y_coords))
    return Brep.BRep(brep_id, polygon=polygon)


def custom_geometry(polygon: Polygon, brep_id, outerloop: bool):
    """Takes a Shapely polygon, checks and enforces orientation."""
    return Brep.BRep(brep_id, polygon=polygon, outerloop=outerloop)


# Test and plot shapes
if __name__ == "__main__":
    fig, ax = plt.subplots()

    # Circle for comparison
    circle_shape = circle(0.55, 0.43, 0.05, brep_id=1, outerloop=True)
    x_c, y_c = circle_shape.polygon.exterior.xy
    ax.plot(x_c, y_c, "b-", label="Circle")


    # Heart innerloop (CW)
    heart_cw = heart(0.55, 0.43, 0.05, brep_id=3, outerloop=False)
    x_h2, y_h2 = heart_cw.polygon.exterior.xy
    ax.plot(x_h2, y_h2, "g-", label="Heart CW")

    ax.set_aspect("equal", "box")
    ax.legend()
    ax.set_title("Circle and Heart with orientation")

    plt.show()

