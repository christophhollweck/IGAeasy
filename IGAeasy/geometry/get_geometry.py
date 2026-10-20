import numpy as np
import matplotlib.pyplot as plt
import splinepy
from scipy.interpolate import make_interp_spline


def raasch_hook(refinements=0, width_elements=4, trimmed=True):
    """Polynomial C1 fit of Raasch's two circular arcs, extended for trimming.

    Active material occupies u in [0, 1], v in [1/12, 11/12].
    Dimensions follow the Abaqus Raasch benchmark in inches (radii 14 and 46).
    Knot insertion refines a fixed fit; the curvature jump is retained at the join.
    With trimmed=False, extract exactly that material rectangle as a fitted patch.
    """
    from scipy.interpolate import BSpline
    from scipy.linalg import null_space
    if refinements < 0 or int(refinements) != refinements or width_elements < 1:
        raise ValueError('Use nonnegative integer refinements and positive width_elements')
    small_length, large_length = 14*np.pi/3, 46*5*np.pi/6
    length = small_length+large_length
    join = small_length/length

    def centerline(u):
        small = np.asarray(u) <= join
        alpha = -np.pi/2+length*np.asarray(u)/14
        beta = 5*np.pi/6-length*(np.asarray(u)-join)/46
        return np.column_stack((np.where(small, 14*np.cos(alpha),
            30*np.sqrt(3)+46*np.cos(beta)), np.where(small, 14+14*np.sin(alpha),
            -16+46*np.sin(beta))))

    breaks = np.r_[-.05, np.linspace(0., join, 5), np.linspace(join, 1., 13)[1:], 1.05]
    knots = np.sort(np.r_[[breaks[0]]*4, breaks[1:-1], join, [breaks[-1]]*4])
    basis = BSpline(knots, np.eye(len(knots)-4), 3)
    parameters = np.linspace(breaks[0], breaks[-1], 3001)
    # Enforce the physical end positions and tangents exactly during fitting.
    constraints = np.vstack((basis([0., 1.]), basis.derivative()([0., 1.])))
    targets = np.vstack((centerline([0., 1.]), [[length, 0.], [0., -length]]))
    particular = np.linalg.lstsq(constraints, targets, rcond=None)[0]
    kernel = null_space(constraints)
    coefficients = particular+kernel@np.linalg.lstsq(basis(parameters)@kernel,
        centerline(parameters)-basis(parameters)@particular, rcond=None)[0]
    v_knots = np.r_[[0.]*4, [1.]*4]
    cps = [[x, y, z] for z in (-2., 6., 14., 22.) for x, y in coefficients]
    geometry = splinepy.BSpline(degrees=[3, 3], knot_vectors=[knots, v_knots], control_points=cps)
    geometry.insert_knots(1, np.linspace(0., 1., width_elements+1)[1:-1])
    for _ in range(int(refinements)):
        for direction in (0, 1):
            spans = np.unique(geometry.knot_vectors[direction])
            geometry.insert_knots(direction, (spans[:-1]+spans[1:])/2)
    if not trimmed:
        geometry = splinepy.helpme.extract.spline(geometry, 0, [0., 1.])
        geometry = splinepy.helpme.extract.spline(geometry, 1, [1/12, 11/12])
    return geometry


def cooks_membrane(n_elements=8, degree=3):
    """Fitted planar Cook panel: corners (0,0), (48,44), (48,60), (0,44)."""
    if not isinstance(n_elements, (int, np.integer)) or n_elements < 1:
        raise ValueError('Use a positive integer element count')
    if not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError('Use a positive integer degree')
    geometry = splinepy.BSpline(degrees=[1, 1],
        knot_vectors=[[0., 0., 1., 1.], [0., 0., 1., 1.]],
        control_points=[[0., 0., 0.], [48., 44., 0.], [0., 44., 0.], [48., 60., 0.]])
    for _ in range(degree-1):
        geometry.elevate_degrees([0, 1])
    for direction in (0, 1):
        geometry.insert_knots(direction, np.linspace(0., 1., n_elements+1)[1:-1])
    return geometry


def open_hemisphere(n_elements=8, radius=10., opening_degrees=18.):
    """Cubic polynomial approximation of one regular quarter of an open hemisphere.

    The two meridians are symmetry cuts. Exact endpoint derivatives preserve
    reflection planes. Refine the returned spline by knot insertion to keep
    the fitted geometry fixed throughout a convergence study.
    """
    if not isinstance(n_elements, (int, np.integer)) or n_elements < 2:
        raise ValueError('Use at least two elements per direction')
    if not np.isfinite([radius, opening_degrees]).all() or radius <= 0 or not 0 < opening_degrees < 90:
        raise ValueError('Use positive radius and an opening strictly between 0 and 90 degrees')
    parameters = np.linspace(0., 1., n_elements+1)

    def arc(start, stop):
        angles = start+(stop-start)*parameters
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        derivatives = [(stop-start)*np.array([-np.sin(a), np.cos(a)]) for a in (start, stop)]
        return make_interp_spline(parameters, points, k=3,
            bc_type=([(1, derivatives[0])], [(1, derivatives[1])]))

    azimuth = arc(0., np.pi/2)
    polar = arc(np.pi/2, np.radians(opening_degrees))
    cps = np.array([[radius*cos_t*sin_p, radius*sin_t*sin_p, radius*cos_p]
                    for cos_p, sin_p in polar.c for cos_t, sin_t in azimuth.c])
    # Remove roundoff at exact reflection/equator coordinates.
    cps[np.abs(cps) < 1e-14*radius] = 0.
    return splinepy.BSpline(degrees=[3, 3], knot_vectors=[azimuth.t, polar.t], control_points=cps)


def paraboloid(n_elx, n_ely, degree=3, curvature=(.2, .3), centered=False):
    """Exact paraboloid; centered=True gives a cap with its crest at the origin."""
    return polynomial_shell(n_elx, n_ely, degree, curvature, centered=centered)


def polynomial_shell(n_elx, n_ely, degree=3, curvature=(.2, .3),
                     mixed=(0., 0.), centered=True, bounds=(0., 0., 1., 1.)):
    """Exact polynomial graph on bounds, preserving the original u/v coordinates."""
    if (not all(isinstance(value, (int, np.integer)) for value in (n_elx, n_ely, degree))
            or degree < 2 or min(n_elx, n_ely) < 1):
        raise ValueError('Use degree >= 2 and positive element counts')
    a, b = np.asarray(curvature, dtype=float)
    c, d = np.asarray(mixed, dtype=float)
    if not np.isfinite([a, b, c, d]).all():
        raise ValueError('Geometry coefficients must be finite')
    u0, v0, u1, v1 = np.asarray(bounds, dtype=float)
    if not np.isfinite(bounds).all() or u0 >= u1 or v0 >= v1:
        raise ValueError('bounds must describe a finite, nonempty parameter rectangle')
    shift, sign = (.5, -1.) if centered else (0., 1.)
    x0, x1, y0, y1 = u0-shift, u1-shift, v0-shift, v1-shift
    linear_u, linear_v = (x0, (x0+x1)/2, x1), (y0, (y0+y1)/2, y1)
    square_u, square_v = (x0*x0, x0*x1, x1*x1), (y0*y0, y0*y1, y1*y1)
    # Products of Bernstein coefficients preserve the polynomial geometry exactly.
    cps = [[linear_u[i], linear_v[j], sign*(a*square_u[i]+b*square_v[j])
            +c*square_u[i]*linear_v[j]+d*linear_u[i]*square_v[j]] for j in range(3) for i in range(3)]
    geometry = splinepy.BSpline(degrees=[2, 2],
        knot_vectors=[[u0]*3+[u1]*3, [v0]*3+[v1]*3], control_points=cps)
    for _ in range(degree-2):
        geometry.elevate_degrees([0, 1])
    geometry.insert_knots(0, np.linspace(u0, u1, n_elx+1)[1:-1])
    geometry.insert_knots(1, np.linspace(v0, v1, n_ely+1)[1:-1])
    return geometry


def surface(
    n_elU,
    n_elV,
    ax,
    ay,
    ximax,
    etamax,
    conti,
    p,
    distortion=False,
    n_waves=5,
    rotation_deg=0.0,
):
    # Define interior knots to insert.
    uins = np.linspace(0, ximax, n_elU + 1)[1:-1]
    vins = np.linspace(0, etamax, n_elV + 1)[1:-1]

    # Define the initial linear knot vectors.
    U = [0, 0, ximax, ximax]
    V = [0, 0, etamax, etamax]

    # Define control points with the u-direction varying fastest.
    cps0 = np.array([
        [0.0, 0.0, 0.0],
        [ax, 0.0, 0.0],
        [0.0, ay, 0.0],
        [ax, ay, 0.0],
    ])

    Bspl = splinepy.BSpline(
        degrees=[1, 1],
        knot_vectors=[U, V],
        control_points=cps0,
    )

    # Elevate the degree in both parametric directions.
    for _ in range(p - 1):
        Bspl.elevate_degrees([0, 1])

    # Insert knots with the multiplicity required for the desired continuity.
    multiplicity = p - conti
    for _ in range(multiplicity):
        if len(uins) > 0:
            Bspl.insert_knots(0, uins)
        if len(vins) > 0:
            Bspl.insert_knots(1, vins)

    if distortion:
        # Distort interior control points while preserving the boundary.
        n_u = len(Bspl.knot_vectors[0]) - Bspl.degrees[0] - 1
        n_v = len(Bspl.knot_vectors[1]) - Bspl.degrees[1] - 1

        cps = Bspl.control_points.copy().reshape((n_v, n_u, 3))

        for j in range(1, n_v - 1):
            for i in range(1, n_u - 1):
                u_phase = n_waves * np.pi * i / (n_u - 1)
                v_phase = n_waves * np.pi * j / (n_v - 1)

                dx = (
                    distortion
                    * np.sin(u_phase)
                    * np.cos(v_phase)
                    * ax / n_elU
                )
                dy = (
                    distortion
                    * np.cos(u_phase)
                    * np.sin(v_phase)
                    * ay / n_elV
                )

                cps[j, i, 0] += dx
                cps[j, i, 1] += dy

        Bspl.control_points = cps.reshape(-1, 3)

    if rotation_deg != 0.0:
        # Rotate the complete geometry about the global z-axis.
        theta = np.deg2rad(rotation_deg)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0.0],
            [sin_theta,  cos_theta, 0.0],
            [0.0,        0.0,       1.0],
        ])

        # Control points are stored as row vectors.
        Bspl.control_points = (
            Bspl.control_points.copy() @ rotation_matrix.T
        )

    return Bspl


def halfpipe():
    ku = [0, 0, 0, 0.5, 1, 1, 1]
    kv = [0, 0, 0, 1, 1, 1]
    p = 2
    conti = p - 1

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

    cps = Bspl.cps
    ku = Bspl.knot_vectors[0]
    kv = Bspl.knot_vectors[1]

    return Bspl


def get_roof1d(phi0=50, phi1=90, R=25, n_elx=4, degree=2, n=100):
    """
    Creates 1D spline data by fitting a curve to points on a circular arc.

    Parameters:
    - phi0: Start angle (in degrees).
    - phi1: End angle (in degrees).
    - R: Radius of the circular arc.
    - n_elx: Number of elements along the x-direction.
    - degree: Degree of the spline curve.
    - n: Number of points used to approximate the arc.

    Returns:
    - data: Original data points on the circular arc (as an array).
    - curve: Fitted spline curve object.
    """
    # Convert angles to radians
    phi0 = np.radians(phi0)
    phi1 = np.radians(phi1)

    # Generate points along the circular arc
    phi = np.linspace(phi1, phi0, n)
    data = [R * np.array([np.cos(phi[i]), np.sin(phi[i])]) for i in range(len(phi))]

    # Fit a spline curve to the data
    n_cps0 = n_elx + degree  # Number of control points
    curve1d, residual_curve = splinepy.helpme.fit.curve(data, degree=degree, n_control_points=n_cps0)
    print(f"Residual of Bspline Curve Fitting: {residual_curve:.5f}")  # Print the fitting residual

    return np.array(data), curve1d


def plot_roof1d(data, curve, n=100):
    """
    Visualizes the 1D roof spline: the original data points, the fitted spline curve,
    control points, knot points, and the control polygon.

    Parameters:
    - data: Original data points on the circular arc.
    - curve: Fitted spline curve object.
    - n: Number of evaluation points along the curve.
    """
    # Evaluate the spline curve at n points
    u = np.linspace(0, 1, n)
    curve_eval = np.array([curve.evaluate([[ui]])[0] for ui in u])

    # Extract unique knot points and evaluate the curve at these positions
    knots = np.unique(curve.knot_vectors[0])
    knot_points = np.array([curve.evaluate([[k]])[0] for k in knots])

    # Control points of the spline
    cps = np.array(curve.cps)

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], label="Data Points", color="blue", s=15)  # Data points
    plt.plot(curve_eval[:, 0], curve_eval[:, 1], label="Spline Curve", color="red", linewidth=2)  # Spline curve
    plt.scatter(knot_points[:, 0], knot_points[:, 1], label="Knot Points", color="green", s=50, marker="x")  # Knots
    plt.scatter(cps[:, 0], cps[:, 1], label="Control Points", color="orange", s=50, marker="o")  # Control points
    plt.plot(cps[:, 0], cps[:, 1], color="orange", linestyle="--", linewidth=1,
             label="Control Polygon")  # Control polygon

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Spline Approximation of a Circular Arc with Knots and Control Points")
    plt.show()


def extrude_geometry(curve1d, n_ely, L):
    """
    Extrudes a 1D curve into 2D by adding a second parameter direction (e.g., height).

    Parameters:
    - curve1d: 1D spline curve to be extruded.
    - n_ely: Number of elements in the extruded direction.
    - L: Length of extrusion.

    Returns:
    - Bspl2d: 2D B-spline geometry created by extrusion.
    """
    # Convert 1D control points to 3D for extrusion
    cps1d = curve1d.cps
    cps1d = np.array([[cp[0], 0, cp[1]] for cp in cps1d])  # Add z-coordinate
    ku = curve1d.knot_vectors[0]
    pu = curve1d.degrees[0]

    # Create 1D B-spline
    Bspl1d = splinepy.BSpline(degrees=[pu], knot_vectors=[ku], control_points=cps1d)

    # Extrude by duplicating control points in the second parameter direction
    kv = [0, 0, 1, 1]  # Knot vector for the extruded direction
    pv = 1  # Degree in the extruded direction
    cps2d = []
    for i in range(2):
        for cp in cps1d:
            cps2d.append([cp[0], i * L, cp[2]])

    # Create 2D B-spline
    Bspl2d = splinepy.BSpline(degrees=[pu, pv], knot_vectors=[ku, kv], control_points=cps2d)

    # Perform degree elevation if necessary
    for i in range(pu - 1):
        Bspl2d.elevate_degrees([1])

    # Refine the mesh by inserting knots along the extruded direction
    vins = np.linspace(0, 1, n_ely + 1)
    vins = vins[1:-1]
    Bspl2d.insert_knots(1, vins)

    return Bspl2d


def global_refine_geometry(Bspl2d, reftimes):
    """
    Refines a 2D B-spline geometry globally by inserting knots.

    Parameters:
    - Bspl2d: 2D B-spline geometry to refine.
    - reftimes: Number of global refinement iterations.

    Returns:
    - Bspl2d: Refined 2D B-spline geometry.
    """
    for i in range(reftimes):
        ku = Bspl2d.knot_vectors[0]
        kv = Bspl2d.knot_vectors[1]
        ku_unique = np.unique(ku)
        kv_unique = np.unique(kv)

        # Insert midpoints of knot spans in both parameter directions
        uins = 0.5 * (ku_unique[:-1] + ku_unique[1:])
        vins = 0.5 * (kv_unique[:-1] + kv_unique[1:])

        Bspl2d.insert_knots(0, uins)
        Bspl2d.insert_knots(1, vins)

    return Bspl2d


# def scordelis_lo_roof(n_elx, n_ely, degree):
#     """
#     Creates a quarter Scordelis-Lo roof geometry (extruded circular arc as a 2D spline).
#
#     Returns:
#     - Bspl2d: 2D B-spline representation of the roof.
#     """
#     n = 100001
#     phi0 = 50
#     phi1 = 90
#     R = 25
#     L = 25
#
#     # Create the base arc and extrude it into 2D
#     data, curve = get_roof1d(phi0, phi1, R, n_elx, degree, n=n)
#     old_kv = curve.knot_vectors[0]
#     old_knots = np.unique(old_kv)
#     new_knots = np.linspace(0, 1, n_elx+1)
#     deviation = np.linalg.norm(old_knots-new_knots)
#     if deviation < 1e-4:
#         new_kv = [0] * degree + new_knots.tolist() + [1] * degree
#         curve.knot_vectors[0] = np.array(new_kv)
#     Bspl2d = extrude_geometry(curve, n_ely, L)
#     return Bspl2d

def scordelis_lo_roof(n_elx, n_ely, degree, full=False):
    """
    Creates a Scordelis-Lo roof geometry (extruded circular arc as a 2D spline).

    If full=False, returns only a quarter of the roof (default, exploiting symmetry).
    If full=True, returns the full half-cylinder roof geometry.

    Parameters:
    - n_elx: Number of elements in circumferential direction (for quarter or half arc).
    - n_ely: Number of elements in longitudinal direction.
    - degree: Degree of the B-spline basis.
    - full: Whether to generate full half-cylinder geometry (default: False).

    Returns:
    - Bspl2d: 2D B-spline representation of the roof geometry.
    """
    n = 100001
    R = 25


    if full:
        phi0 = 130
        phi1 = 50
        nel_phi = 2 * n_elx
        L = 50
        n_ely *= 2
    else:
        phi0 = 50
        phi1 = 90
        nel_phi = n_elx
        L = 25

    # Create the base arc and extrude it into 2D
    data, curve = get_roof1d(phi0, phi1, R, nel_phi, degree, n=n)
    old_kv = curve.knot_vectors[0]
    old_knots = np.unique(old_kv)
    new_knots = np.linspace(0, 1, nel_phi + 1)
    deviation = np.linalg.norm(old_knots - new_knots)
    if deviation < 1e-4:
        new_kv = [0] * degree + new_knots.tolist() + [1] * degree
        curve.knot_vectors[0] = np.array(new_kv)

    Bspl2d = extrude_geometry(curve, n_ely, L)
    return Bspl2d


def tcm_case_block_only(
    ax=120,
    ay=100,
    p=2,
    conti=1,
    n_elU=10,
    n_elV=6,
    distortion=False
):
    """
    Returns ONLY the deformable block as a single BSpline surface.
    Walls must be applied as boundary conditions during analysis.
    
    Returns:
    - Bspl2d: Single BSpline surface representing the deformable block
    """
    Bspl2d = surface(
        n_elU=n_elU,
        n_elV=n_elV,
        ax=ax,
        ay=ay,
        ximax=1,
        etamax=1,
        conti=conti,
        p=p,
        distortion=distortion
    )

    return Bspl2d


def tcm_case_with_walls(
    ax=320,
    ay=100,
    wall_thickness=5,
    p=2,
    conti=1,
    n_elU=10,
    n_elV=6,
    distortion=False,
    return_multipatch=True
):
    """
    Option A: Returns the deformable block + rigid walls.
    
    Parameters:
    - ax, ay: Dimensions of the deformable block
    - wall_thickness: Thickness of rigid walls
    - p: Polynomial degree
    - conti: Continuity (p - conti = multiplicity)
    - n_elU, n_elV: Number of elements in each direction
    - distortion: Whether to apply distortion to the block
    - return_multipatch: If True, returns splinepy.Multipatch object
                         If False, returns dict of BSpline surfaces
    
    Geometry:
        ┌────────── top wall ──────────┐
        │                              │
     left wall   deformable block      │
        │                              │
        └───────── bottom wall ────────┘
    
    Returns:
    - If return_multipatch=True: splinepy.Multipatch object (Bspl2d)
    - If return_multipatch=False: dict with keys ["block", "wall_left", "wall_bottom", "wall_top"]
    """

    # --- Deformable block ---
    Bspl2d_block = surface(
        n_elU=n_elU,
        n_elV=n_elV,
        ax=ax,
        ay=ay,
        ximax=1,
        etamax=1,
        conti=conti,
        p=p,
        distortion=distortion
    )

    # --- Rigid walls ---

    # Left wall: vertical strip at x = 0
    Bspl2d_wall_left = surface(
        n_elU=1,
        n_elV=n_elV,
        ax=wall_thickness,
        ay=ay,
        ximax=1,
        etamax=1,
        conti=1,
        p=1,
        distortion=False
    )
    # Shift it to x = 0 - wall_thickness
    cps = Bspl2d_wall_left.cps.copy()
    cps[:, 0] = cps[:, 0] #- wall_thickness
    Bspl2d_wall_left.cps = cps

    # Bottom wall: horizontal strip at y = 0
    Bspl2d_wall_bottom = surface(
        n_elU=n_elU,
        n_elV=1,
        ax=ax,
        ay=wall_thickness,
        ximax=1,
        etamax=1,
        conti=1,
        p=1,
        distortion=False
    )
    cps = Bspl2d_wall_bottom.cps.copy()
    cps[:, 1] = cps[:, 1] - wall_thickness
    Bspl2d_wall_bottom.cps = cps

    # Top wall: horizontal strip at y = ay
    Bspl2d_wall_top = surface(
        n_elU=n_elU,
        n_elV=1,
        ax=ax,
        ay=wall_thickness,
        ximax=1,
        etamax=1,
        conti=1,
        p=1,
        distortion=False
    )
    cps = Bspl2d_wall_top.cps.copy()
    cps[:, 1] = cps[:, 1] + ay
    Bspl2d_wall_top.cps = cps

    if return_multipatch:
        # Return as splinepy Multipatch object
        import splinepy
        Bspl2d = splinepy.Multipatch(
            splines=[Bspl2d_block, Bspl2d_wall_left, Bspl2d_wall_bottom, Bspl2d_wall_top]
        )
        return Bspl2d
    else:
        # Return as dictionary (original behavior)
        return {
            "block": Bspl2d_block,
            "wall_left": Bspl2d_wall_left,
            "wall_bottom": Bspl2d_wall_bottom,
            "wall_top": Bspl2d_wall_top
        }


# Example usage:
if __name__ == "__main__":
    import splinepy

    # Option 1: Get only the block (like other geometry functions)
    block = tcm_case_block_only()
    block.show()

    # Option 2: Get multipatch with walls (returns Bspl2d)
    Bspl2d = tcm_case_with_walls(return_multipatch=True)
    Bspl2d.show()

    # Option 3: Get dict with individual surfaces (original behavior)
    geo_dict = tcm_case_with_walls(return_multipatch=False)
    geo_dict["block"].show()
    geo_dict["wall_left"].show()
