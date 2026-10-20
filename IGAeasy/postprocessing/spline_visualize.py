import matplotlib.pyplot as plt
import matplotlib.patches as plp
from shapely.geometry import box, Polygon, Point, LineString
import pyvista as pv
from scipy.interpolate import RBFInterpolator
import triangle
from IGAeasy.postprocessing.postprocessing import update_cps
from IGAeasy.loads import LineLoad
import numpy as np
from IGAeasy.constraints.dirichlet_weak import WeakLineConstraint

# THB
def plot_mesh_overloading(RS) -> None:
    """
    Plots the mesh.
    """
    fig = plt.figure(figsize=(10, 6))
    axs = fig.add_subplot(1, 1, 1)
    overloaded_patch = plp.Patch(color='red', label='Overloaded Elements')
    underloaded_patch = plp.Patch(color='green', label='Underloaded Elements')
    normal_patch = plp.Patch(color='black', label='Normal Elements')

    for m in RS.elements:
        w = m.u_max - m.u_min
        h = m.v_max - m.v_min

        if len(m.supported_b_splines) == (m.p + 1) ** 2:
            axs.add_patch(plp.Rectangle((m.u_min, m.v_min), w, h, fill=True, color='black',
                                        alpha=0.2))
        elif len(m.supported_b_splines) < (m.p + 1) ** 2:
            axs.add_patch(plp.Rectangle((m.u_min, m.v_min), w, h, fill=True, color='green',
                                        alpha=0.2))
        else:
            axs.add_patch(plp.Rectangle((m.u_min, m.v_min), w, h, fill=True, color='red',
                                        alpha=0.2))


    if RS.BReps:
        for brep in RS.BReps:
            x, y = brep.polygon.exterior.xy
            plt.plot(x, y, color='red')

    max_u = RS.ku[-1]
    max_v = RS.kv[-1]
    axs.set_xlim(0, max_u)
    axs.set_ylim(0, max_v)

    # Adjust layout to fit legend
    plt.subplots_adjust(right=0.75)  # Increase right margin further

    # Add legend outside the plot
    axs.legend(handles=[overloaded_patch, underloaded_patch, normal_patch],
               loc='center left', bbox_to_anchor=(1.05, 0.5))

    plt.show()


def plot_parametric_space(RS, plot_gauss_points=False, savedir=False, name=None, SBM=False):
    """
    Plots non-void elements in the parametric space, their material/void regions, BReps,
    optional Gauss points, and (if enabled) SBM and BRep normals or line normals.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from shapely.geometry import box

    # fig, ax = plt.subplots()
    fig = plt.gcf()  # fig
    ax  = plt.gca()  # ax

    def plot_arrow(ax, x, y, dx, dy, color, dashed=False, scale=0.05, width=0.4):
        linestyle = (0, (3, 3)) if dashed else 'solid'
        arrow = FancyArrowPatch(
            (x, y), (x + scale * dx, y + scale * dy),
            color=color,
            arrowstyle='-|>',
            mutation_scale=4,
            linewidth=width,
            linestyle=linestyle
        )
        ax.add_patch(arrow)

    # # --- Element bounds and material/void regions ---
    # for el in RS.elements:
    #     element_polygon = box(el.u_min, el.v_min, el.u_max, el.v_max)
    #     x, y = element_polygon.exterior.xy
    #     ax.plot(x, y, linestyle=':', color='grey', linewidth=1.5)
    #
    #     # Plot untrimmed non-void elements
    #     if not el.void and not el.trimmed:
    #     # if not el.void:
    #     #     ax.fill(x, y, facecolor='#FFFACD', edgecolor='grey', alpha=0.99)
    #         ax.fill(x, y, facecolor='grey', edgecolor='black', alpha=0.5)
    #
    #     # Plot trimmed elements
    #     if not el.void and el.trimmed:
    #         # Plot material polygons (alternative visualizations, commented)
    #         for poly in el.material_polygons:
    #             plot_polygon_custom(poly, ax=ax, facecolor='grey', edgecolor='black', alpha=0.5,
    #                                 point_color='black', point_size=0)

            # for poly in el.coarse_material_polygons:
            #     x, y = poly.exterior.xy
            #     ax.fill(x, y, facecolor='grey', edgecolor='black', alpha=0.5)

            # # Plot material triangles (to show triangulation)
            # for triangle_list in el.coarse_material_triangles:
            #     for tri in triangle_list:
            #         plot_triangle_custom(tri, ax=ax, facecolor='black', edgecolor='black', alpha=0.1)

            # Plot void polygons (commented)
            # for poly in el.void_polygons:
            #     plot_polygon_custom(poly, ax=ax, facecolor='red', edgecolor='black', alpha=0.5,
            #                         point_color='red', point_size=15)

    # --- Element bounds and material/void regions ---
    for el in RS.elements:
        element_polygon = box(el.u_min, el.v_min, el.u_max, el.v_max)
        x, y = element_polygon.exterior.xy
        ax.plot(x, y, color='black', linewidth=0.5, alpha=0.5)

        # Plot untrimmed non-void elements
        if not el.void and not el.trimmed:
        # if not el.void:
        #     ax.fill(x, y, facecolor='#FFFACD', edgecolor='grey', alpha=0.99)
            ax.fill(x, y, facecolor='grey', edgecolor='black', alpha=0.5, linewidth=0.5)

        # Plot trimmed elements
        if not el.void and el.trimmed:
            # Plot material polygons (alternative visualizations, commented)
            for poly in el.material_polygons:
                plot_polygon_custom(poly, ax=ax, facecolor='grey', edgecolor='black', alpha=0.5, linewidth=0.0)

        if plot_gauss_points:
            for gp in getattr(el, "gausspoints", []):
                ax.scatter(gp.xi, gp.eta, color='blue', s=8)

    # --- SBM Plot ---
    if SBM:
        def plot_gp_normals(edge, ax, scale=0.05, width=0.4):
            def plot_group(gp_list, color):
                for gp in gp_list:
                    cp = gp.closest_point
                    plot_arrow(ax, gp.u, gp.v, gp.n[0], gp.n[1], color=color, dashed=True, scale=scale, width=width)
                    ax.plot([gp.u, cp.x], [gp.v, cp.y], color=color, linestyle='dotted', linewidth=width)
                    plot_arrow(ax, cp.x, cp.y, gp.n_line_closest_point[0], gp.n_line_closest_point[1],
                               color=color, dashed=False, scale=scale, width=width)
            # plot_group(edge.dirichlet_data, 'green')
            # plot_group(edge.neumann_data, 'red')

        for el in RS.elements:
            # if not el.active or not hasattr(el, "active_sides"):
            #     continue
            for edge in el.active_sides:
                x, y = edge.line.xy
                ax.plot(x, y, color='darkred', linewidth=2.5)
                plot_gp_normals(edge, ax, scale=0.04, width=1)

            if len(el.active_sides) > 0:
                element_polygon = box(el.u_min, el.v_min, el.u_max, el.v_max)
                x, y = element_polygon.exterior.xy
                ax.fill(x, y, facecolor='red', edgecolor='grey', alpha=0.8)
                # if x[0] > 0.1 and y[0] > 0.1 and x[0] < 0.9 and y[0] < 0.9:
                #     ax.fill(x, y, facecolor='red', edgecolor='grey', alpha = 0.8)

                # # rectangle boundaries
                # x1, y1 = 0.45, 0.32
                # x2, y2 = 0.66, 0.52
                #
                # # check if the element centroid (x[0], y[0]) is outside the rectangle
                # if not (x1 <= x[0] <= x2 and y1 <= y[0] <= y2):
                #     ax.fill(x, y, facecolor='red', edgecolor='grey', alpha=0.8)


            # if el.level>0 and not el.void:
            #     element_polygon = box(el.u_min, el.v_min, el.u_max, el.v_max)
            #     x, y = element_polygon.exterior.xy
            #     ax.fill(x, y, facecolor='orange', edgecolor='grey', alpha=0.5)



    # # --- Plot for SBM=False ---
    # if not SBM:
    #     for el in RS.elements:
    #         if not el.active or not hasattr(el, "gps_line"):
    #             continue
    #
    #         if el.level > 0 and not el.void:
    #             element_polygon = box(el.u_min, el.v_min, el.u_max, el.v_max)
    #             x, y = element_polygon.exterior.xy
    #             ax.fill(x, y, facecolor='orange', edgecolor='grey', alpha=0.5)
    #
    #         gp_list = el.gps_line
    #
    #         for gp in gp_list:
    #             line = gp.line
    #
    #             if isinstance(line, WeakLineConstraint):
    #                 color = 'green'
    #             else:
    #                 color = 'red'
    #
    #             ax.plot(gp.u, gp.v, 'o', color=color, markersize=4)
    #
    #             n = gp.n[:2] if len(gp.n) > 2 else gp.n
    #             plot_arrow(ax, gp.u, gp.v, n[0], n[1], color=color, dashed=False, scale=0.04, width=1)

    # # # --- Basisfunktionen-Plot ---
    # bf_material = set()
    # bf_void = set()

    # for el in RS.elements:
    #     if el.void:
    #         bf_void.update(el.supported_b_splines)
    #     else:
    #         bf_material.update(el.supported_b_splines)

    # # for bf in bf_material:
    # #     u, v = bf.cp[0], bf.cp[1]
    # #     ax.scatter(u, v, color='deepskyblue', s=8)

    # # for bf in bf_void:
    # #     u, v = bf.cp[0], bf.cp[1]
    # #     ax.scatter(u, v, color='deepskyblue', alpha=0.1, s=8)

    # # # --- BRep-Plot ---
    # for Brep in RS.BReps:
    #     x, y = Brep.polygon.exterior.xy
    #     ax.plot(x, y, linewidth=2, color='blue')

    # --- BRep-Plot ---
    for Brep in RS.BReps:
        poly = Brep.polygon
        if poly.geom_type == "Polygon":
            x, y = poly.exterior.xy
            ax.plot(x, y, linewidth=1.5, color='red')
        elif poly.geom_type == "MultiPolygon":
            for p in poly.geoms:
                x, y = p.exterior.xy
                ax.plot(x, y, linewidth=1.5, color='red')

    # p = 2
    # step_size = 1 / 20
    # level_x = 6
    # level_y = 13
    # xxx = [step_size*level_x, step_size*(level_x+p+1), step_size*(level_x+p+1), step_size*level_x, step_size*level_x]
    # yyy = [step_size*level_y, step_size*level_y, step_size*(level_y+p+1), step_size*(level_y+p+1), step_size*level_y]
    # ax.plot(xxx, yyy, "--", color='black', linewidth=3)

    # level_x = 5
    # level_y = 12
    # xxx = [step_size*level_x, step_size*(level_x+p+1), step_size*(level_x+p+1), step_size*level_x, step_size*level_x]
    # yyy = [step_size*level_y, step_size*level_y, step_size*(level_y+p+1), step_size*(level_y+p+1), step_size*level_y]
    # ax.plot(xxx, yyy, "--", color='black',linewidth=3)

    # level_x = 4
    # level_y = 11
    # xxx = [step_size*level_x, step_size*(level_x+p+1), step_size*(level_x+p+1), step_size*level_x, step_size*level_x]
    # yyy = [step_size*level_y, step_size*level_y, step_size*(level_y+p+1), step_size*(level_y+p+1), step_size*level_y]
    # ax.plot(xxx, yyy, "--", color='black', linewidth=3)

    #*******
    # level_x = 5
    # level_y = 6
    # xxx = [1/12*level_x, 1/12*(level_x+p+1), 1/12*(level_x+p+1), 1/12*level_x, 1/12*level_x]
    # yyy = [1/12*level_y, 1/12*level_y, 1/12*(level_y+p+1), 1/12*(level_y+p+1), 1/12*level_y]
    # ax.plot(xxx, yyy, "--", color='black', linewidth=4)

    # level_x = 4
    # level_y = 5
    # xxx = [1/12*level_x, 1/12*(level_x+p+1), 1/12*(level_x+p+1), 1/12*level_x, 1/12*level_x]
    # yyy = [1/12*level_y, 1/12*level_y, 1/12*(level_y+p+1), 1/12*(level_y+p+1), 1/12*level_y]
    # ax.plot(xxx, yyy, "--", color='black',linewidth=4)

    # level_x = 3
    # level_y = 4
    # xxx = [1/12*level_x, 1/12*(level_x+p+1), 1/12*(level_x+p+1), 1/12*level_x, 1/12*level_x]
    # yyy = [1/12*level_y, 1/12*level_y, 1/12*(level_y+p+1), 1/12*(level_y+p+1), 1/12*level_y]
    # ax.plot(xxx, yyy, "--", color='black', linewidth=4)

    # --- Plot appearance ---
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


    # if savedir:
    #     import tikzplotlib
    #     tikz_file_path = f'{savedir}/{name}.tex'
    #     pdf_file_path = f'{savedir}/{name}.pdf'
    #
    #     tikzplotlib.save(tikz_file_path)
    #     print(f'TikZ graphic saved to {tikz_file_path}')
    #     plt.savefig(pdf_file_path, bbox_inches='tight')
    #     print(f'PDF graphic saved to {pdf_file_path}')
    #
    # plt.show()

    if savedir:
        pdf_file_path = f"{savedir}/{name}.pdf"
        plt.savefig(pdf_file_path, bbox_inches="tight")
        print(f"PDF graphic saved to {pdf_file_path}")

        try:
            import tikzplotlib

            tikz_file_path = f"{savedir}/{name}.tex"
            tikzplotlib.save(tikz_file_path)
            print(f"TikZ graphic saved to {tikz_file_path}")

        except ImportError as exc:
            print(f"TikZ export skipped because tikzplotlib is incompatible: {exc}")
        except Exception as exc:
            print(f"TikZ export failed: {exc}")

    plt.show()


def plot_polygon_custom(poly, ax, facecolor, edgecolor, alpha=0.5, linewidth=0.5):
    """Plots a polygon with edges and corner points."""
    x, y = poly.exterior.xy
    ax.fill(x, y, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)


def plot_triangle_custom(triangle, ax=None, edgecolor='black', facecolor='none', alpha=1):
    """Plots a single triangle with edges and corner points."""
    x, y = triangle.exterior.xy
    ax.plot(x, y, color=edgecolor, lw=1)  # Triangle edges
    ax.fill(x, y, facecolor=facecolor, alpha=alpha)


def plot_ips(RS):
    fig = plt.figure(figsize=(15, 15))
    for cell in RS.elements:
        AABB = [[cell.u_min, cell.u_max], [cell.v_min, cell.v_max]]
        cellAABB = np.array(AABB)
        x = cellAABB[0, [0, 1, 1, 0, 0]]
        y = cellAABB[1, [0, 0, 1, 1, 0]]
        plt.plot(x, y, color='black')
        for gp in cell.gausspoints:
            plt.scatter(gp.xi, gp.eta, color='red', s=8)
    plt.show()


import numpy as np
import pyvista as pv
import os


def plot_basis(RS, sc=5, N=40, view=None):
    # Data calculation
    U, V = RS.ku, RS.kv
    mat = []
    for fun in RS.basis:
        xi_dir, eta_dir = fun.knots_u, fun.knots_v
        xi = np.linspace(xi_dir[0], xi_dir[-1], N)
        eta = np.linspace(eta_dir[0], eta_dir[-1], N)
        Xi, Eta = np.meshgrid(xi, eta)
        z = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                z[i, j] = fun(xi[i], eta[j])
        mat.append([Xi.T, Eta.T, z])

    p = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    p.set_background("white")

    for Xi, Eta, Z in mat:
        surface = pv.StructuredGrid(sc * Xi, sc * Eta, Z)
        p.add_mesh(surface, scalars=Z.ravel(order='F'), cmap="viridis",
                   smooth_shading=True, lighting=True, show_scalar_bar=False)

    bx, by = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    flat = pv.StructuredGrid(sc * bx.T, sc * by.T, np.zeros((5, 5)))
    p.add_mesh(flat, color="lightgrey", opacity=0.2, show_scalar_bar=False)

    if view == "iso":
        p.view_isometric()
        p.camera.azimuth += 180
    elif view == "top":
        p.view_xy()

    # Fit the camera to the geometry.
    p.reset_camera()

    return p


import numpy as np
import imageio


def save_plot(filename, path, plotter):
    if not os.path.exists(path):
        os.makedirs(path)

    full_path = os.path.join(path, filename)

    # Kamera optimieren
    plotter.reset_camera()
    plotter.camera.zoom(1.1)  # Leicht reduziert, damit Cropping Platz hat

    # Screenshot als RGB-Array (statt Datei)
    img = plotter.screenshot(None)

    if img is not None:
        # Locate non-white pixels against the white background.

        non_white = np.any(img < 255, axis=2)


        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)

        if np.any(rows) and np.any(cols):
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]

            # Keep a two-pixel margin when cropping.
            img_cropped = img[max(0, ymin - 2):ymax + 2, max(0, xmin - 2):xmax + 2]


            imageio.imsave(full_path, img_cropped)
        else:
            # Keep the original image if no foreground was found.
            plotter.screenshot(full_path)

    plotter.close()
    print(f"Image saved and cropped to: {full_path}")

def visualize_modes(frequencies, eigenvectors, RS, dof_per_cp=2, dimensions=2, scale_factor=0.25):
    """
    Visualize specific modes of a EVP by selecting the mode number and plotting the corresponding displacements.
    The displacement vectors are scaled by the provided scale factor.

    Parameters:
    - frequencies   : ndarray, frequencies of the eigenmodes
    - eigenvectors  : ndarray, eigenvectors of the modes
    - RS            : object, the geometric model to visualize
    - dof_per_cp    : int, degrees of freedom per control point (default 2)
    - dimensions    : int, problem dimensions (default 2)
    - scale_factor  : float, scaling factor for the displacements (default 1.0)
    """
    mode = 0  # Start at mode 0

    while True:
        # Prompt user for input or press Enter to increase the mode by 1
        user_input = input(f"Choose a mode number (0 to {len(frequencies) - 1}, -1 to exit): ")

        if user_input == "":  # If Enter is pressed without any input, increase the mode
            mode += 1
            if mode >= len(frequencies):
                print("You've reached the last mode.")
                break
        elif user_input == "-1":
            break
        else:
            try:
                mode = int(user_input)
                if mode < 0 or mode >= len(frequencies):
                    print("Invalid input. Please try again.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a valid number.")
                continue

        print(f"Mode {mode}, Frequency: {frequencies[mode]:.4f}")
        displacements = eigenvectors[:, mode] * scale_factor  # Scale the displacements
        print(f"Scaled displacements: {displacements.shape}")

        # Update control points based on scaled displacements
        update_cps(RS, displacements, dof_per_cp=dof_per_cp, dimensions=dimensions)

        # Visualize the surface with the mode
        plot_trimmed_surface(
            RS, n_points=10, s=0.5, display_mode='displacement_magnitude'
        )


def plot_trimmed_surface(RS, n_points=10, s=0.05, evalpnt=None, stressinterp=False, output_path=None, display_mode=None):
    """
    Plot the surface with options to display stress or displacement in different components.

    Parameters:
    - RS: The refined space object.
    - n_points: Number of points for mesh refinement.
    - s: Scaling factor for arrows.
    - evalpnt: Evaluation point.
    - stressinterp: Flag for stress interpolation.
    - output_path: Path to save the plot (optional).
    - display_mode: Mode for color display ('displacement_x', 'displacement_y', 'displacement_z',
                   'displacement_magnitude', 'stress_von_mises').
    """

    available_modes = [
        'displacement_x', 'displacement_y', 'displacement_z',
        'displacement_magnitude', 'stress_von_mises'
    ]

    if display_mode is None:
        print("Available display modes:")
        for i, mode in enumerate(available_modes):
            print(f"{i + 1}: {mode}")

        try:
            choice = int(input("Select a display mode (enter number): ")) - 1
            if 0 <= choice < len(available_modes):
                display_mode = available_modes[choice]
            else:
                print("Invalid selection. Using default: displacement_z")
                display_mode = 'displacement_z'
        except ValueError:
            print("Invalid input. Using default: displacement_z")
            display_mode = 'displacement_z'

    def eval(u, v):
        x = np.zeros_like(u)
        y = np.zeros_like(u)
        z = np.zeros_like(u)
        ux = np.zeros_like(u)
        uy = np.zeros_like(u)
        uz = np.zeros_like(u)
        u_mag = np.zeros_like(u)
        scalar_field = np.zeros_like(u)
        for i in range(len(u)):
            eval_point = RS(u[i], v[i])
            eval_disp = RS.eval_displacement(u[i], v[i])
            eval_scalar_field = RS.eval_scalar_field(u[i], v[i])
            geo = eval_point + eval_disp
            x[i] = geo[0]
            y[i] = geo[1]
            z[i] = geo[2]
            ux[i] = eval_disp[0]
            uy[i] = eval_disp[1]
            uz[i] = eval_disp[2]
            u_mag[i] = np.linalg.norm(eval_disp)
            scalar_field[i] = eval_scalar_field
        return x, y, z, ux, uy, uz, u_mag, scalar_field

    points = []
    faces = []
    scalars = []
    horizontal_lines = set()
    vertical_lines = set()
    point_offset = 0


    line_points = []
    line_cells = []

    line_offset = 0

    if display_mode == 'stress_von_mises':
        from IGAeasy.postprocessing.shell_results import interpolate_shell_results
        interpolate_shell_results(RS, 'vM', available=('vM',))

    for element in RS.elements:
        if not getattr(element, 'active', True):
            continue
        if element.void == False and element.trimmed == False:
            u_min, v_min, u_max, v_max = element.u_min, element.v_min, element.u_max, element.v_max
            u, v = np.meshgrid(np.linspace(u_min, u_max, n_points), np.linspace(v_min, v_max, n_points))
            x, y, z, ux, uy, uz, u_mag, scalar_field = eval(u.flatten(), v.flatten())

            horizontal_lines.add((u_min, u_max, v_min))
            horizontal_lines.add((u_min, u_max, v_max))
            vertical_lines.add((v_min, v_max, u_min))
            vertical_lines.add((v_min, v_max, u_max))

            element_points = np.column_stack((x, y, z))
            points.append(element_points)

            for i in range(n_points - 1):
                for j in range(n_points - 1):
                    p1 = point_offset + i * n_points + j
                    p2 = point_offset + i * n_points + (j + 1)
                    p3 = point_offset + (i + 1) * n_points + j
                    p4 = point_offset + (i + 1) * n_points + (j + 1)
                    faces.append([3, p1, p2, p3])
                    faces.append([3, p3, p2, p4])

            if display_mode == 'displacement_x':
                scalars.extend(ux)
            elif display_mode == 'displacement_y':
                scalars.extend(uy)
            elif display_mode == 'displacement_z':
                scalars.extend(uz)
            elif display_mode == 'displacement_magnitude':
                scalars.extend(u_mag)
            elif display_mode == 'scalar_field':
                scalars.extend(scalar_field)
            elif display_mode == 'stress_von_mises':
                if stressinterp and len(element.gausspoints) > 0:
                    interpolated_stress = element.vM_interp(np.column_stack((u.flatten(), v.flatten())))
                    scalars.extend(interpolated_stress)
                else:
                    avg_stress = np.mean([gp.sigma_vM for gp in element.gausspoints]) if len(element.gausspoints) > 0 else 0
                    scalars.extend([avg_stress] * element_points.shape[0])

            point_offset += len(element_points)

        if element.void == False and element.trimmed == True:
            # triangles = element.material_triangles
            polygons = element.material_polygons
            polygons, triangles = refine(polygons, element, n_points)

            # Process material_triangles to create mesh
            uv_to_index = {}
            element_points = []
            element_scalars = []

            for triangle in triangles:
                # Extract vertices of the triangle
                vertices = list(triangle.exterior.coords[:-1])  # Exclude the last point (duplicate of the first)
                for uv in vertices:
                    u_vertex, v_vertex = uv
                    key = (u_vertex, v_vertex)
                    if key not in uv_to_index:
                        uv_to_index[key] = len(element_points)
                        # Evaluate geometry and displacement
                        eval_point = RS(u_vertex, v_vertex)
                        eval_disp = RS.eval_displacement(u_vertex, v_vertex)
                        eval_scalar = RS.eval_scalar_field(u_vertex, v_vertex)
                        geo = eval_point + eval_disp
                        x = geo[0]
                        y = geo[1]
                        z = geo[2]
                        element_points.append([x, y, z])

                        # Compute scalar based on display_mode
                        if display_mode == 'displacement_x':
                            scalar = eval_disp[0]
                        elif display_mode == 'displacement_y':
                            scalar = eval_disp[1]
                        elif display_mode == 'displacement_z':
                            scalar = eval_disp[2]
                        elif display_mode == 'displacement_magnitude':
                            scalar = np.linalg.norm(eval_disp)
                        elif display_mode == 'scalar_field':
                            scalar = eval_scalar
                        elif display_mode == 'stress_von_mises':
                            if stressinterp and len(element.gausspoints) > 0:
                                stress = element.vM_interp(np.array([[u_vertex, v_vertex]]))
                                scalar = stress[0]
                            else:
                                # Use average stress
                                scalar = np.mean(
                                    [gp.sigma_vM for gp in element.gausspoints]) if element.gausspoints else 0
                        element_scalars.append(scalar)

            # Add points and scalars if any
            if element_points:
                points.append(np.array(element_points))
                scalars.extend(element_scalars)

                # Create faces for each triangle
                for triangle in triangles:
                    vertices = list(triangle.exterior.coords[:-1])  # Exclude the last point
                    indices = [uv_to_index[(uv[0], uv[1])] for uv in vertices]
                    faces.append([3, point_offset + indices[0], point_offset + indices[1], point_offset + indices[2]])

                point_offset += len(element_points)

            # Process material_polygons to add boundary lines
            for polygon in polygons:
                # Extract u and v coordinates from the polygon's exterior
                u_poly = [uv[0] for uv in polygon.exterior.coords]
                v_poly = [uv[1] for uv in polygon.exterior.coords]

                # Evaluate the polygon's points
                x_poly, y_poly, z_poly, _, _, _, _, _ = eval(u_poly, v_poly)
                current_start = len(line_points)
                # Add each point to line_points
                for x, y, z in zip(x_poly, y_poly, z_poly):
                    line_points.append([x, y, z])
                # Create line segments connecting consecutive points and close the loop
                n = len(x_poly)
                for i in range(n):
                    line_cells.append([2, current_start + i, current_start + ((i + 1) % n)])
                line_offset += n

    points = np.vstack(points)
    faces = np.hstack(faces)
    scalars = np.array(scalars)

    surface = pv.PolyData(points)
    surface.faces = faces
    surface[display_mode] = scalars

    plotter = pv.Plotter()
    plotter.add_mesh(surface, scalars=display_mode, cmap="coolwarm", show_scalar_bar=True, show_edges=False, point_size=0)


    # Horizontale Linien
    for line in horizontal_lines:
        u_start, u_end, v_const = line
        ul = np.linspace(u_start, u_end, n_points)
        vl = np.full_like(ul, v_const)  # Konstantes v
        xl, yl, zl, _, _, _, _, _ = eval(ul, vl)
        for i in range(len(xl)):
            line_points.append([xl[i], yl[i], zl[i]])
            if i > 0:
                line_cells.append([2, line_offset + i - 1, line_offset + i])
        line_offset += len(xl)

    # Vertikale Linien
    for line in vertical_lines:
        v_start, v_end, u_const = line
        vl = np.linspace(v_start, v_end, n_points)
        ul = np.full_like(vl, u_const)  # Konstantes u
        xl, yl, zl, _, _, _, _, _ = eval(ul, vl)
        for i in range(len(xl)):
            line_points.append([xl[i], yl[i], zl[i]])
            if i > 0:
                line_cells.append([2, line_offset + i - 1, line_offset + i])
        line_offset += len(xl)


    line_mesh = pv.PolyData(np.array(line_points))
    line_mesh.lines = np.array(line_cells)  # Line connectivity

    plotter.add_mesh(line_mesh, color="black", line_width=0.5, point_size=0)

    axis_length = 1.0

    x_arrow = pv.Arrow(start=[0, 0, 0], direction=[axis_length, 0, 0], scale=s * axis_length)
    plotter.add_mesh(x_arrow, color="red", label="X-Axis", point_size=0)

    y_arrow = pv.Arrow(start=[0, 0, 0], direction=[0, axis_length, 0], scale=s * axis_length)
    plotter.add_mesh(y_arrow, color="green", label="Y-Axis", point_size=0)

    z_arrow = pv.Arrow(start=[0, 0, 0], direction=[0, 0, axis_length], scale=s * axis_length)
    plotter.add_mesh(z_arrow, color="blue", label="Z-Axis", point_size=0)

    # for b in RS.basis:
    #     cp = b.cp + b.displacements
    #     plotter.add_points(cp, color="orange", point_size=20)

    plotter.show()


def insert_intersections_along_edges(poly, refinement_lines):
    """
    Insert intersections with refinement lines in order along each polygon edge.
    """
    new_coords = []
    coords = list(poly.exterior.coords)

    # The closing point repeats the first vertex.
    for i in range(len(coords) - 1):
        start = coords[i]
        end = coords[i + 1]
        edge_line = LineString([start, end])
        intersections = []

        for line in refinement_lines:
            inter = edge_line.intersection(line)
            if inter.is_empty:
                continue
            elif inter.geom_type == 'Point':
                intersections.append((inter.x, inter.y))
            elif inter.geom_type == 'MultiPoint':
                intersections.extend([(pt.x, pt.y) for pt in inter.geoms])
            # Handle intersections returned as LineString objects.
            # Include both endpoints.
            elif inter.geom_type == 'LineString':
                intersections.append(tuple(inter.coords[0]))
                intersections.append(tuple(inter.coords[-1]))

        # Sort intersections along the edge
        # by distance from its starting point.
        edge_length = edge_line.length

        def t_val(pt):
            dx = pt[0] - start[0]
            dy = pt[1] - start[1]
            return np.sqrt(dx ** 2 + dy ** 2) / edge_length if edge_length != 0 else 0

        intersections_sorted = sorted(intersections, key=t_val)


        new_coords.append(start)
        # Insert sorted intersection points, excluding
        # points coinciding with either endpoint within tolerance.
        for pt in intersections_sorted:
            if not np.allclose(pt, start) and not np.allclose(pt, end):
                new_coords.append(pt)

    new_coords.append(coords[-1])

    return Polygon(new_coords)


def refine(polygons, el, n):
    # Offset samples from knot boundaries.
    epsx = (el.u_max - el.u_min) / 100
    epsy = (el.v_max - el.v_min) / 100

    # Sample both parameter directions.
    u_vals = np.linspace(el.u_min, el.u_max, n)
    v_vals = np.linspace(el.v_min, el.v_max, n)

    # Sample element edges as line segments.
    # Use horizontal ticks on vertical edges.
    left_lines = [LineString([(el.u_min - epsx, v), (el.u_min + epsx, v)]) for v in v_vals]
    right_lines = [LineString([(el.u_max - epsx, v), (el.u_max + epsx, v)]) for v in v_vals]
    # Use vertical ticks on horizontal edges.
    bottom_lines = [LineString([(u, el.v_min - epsy), (u, el.v_min + epsy)]) for u in u_vals]
    top_lines = [LineString([(u, el.v_max - epsy), (u, el.v_max + epsy)]) for u in u_vals]


    all_lines = left_lines + right_lines + bottom_lines + top_lines

    # Refine polygon edges by inserting their intersection points.
    refined_polygons = []
    for poly in polygons:
        refined_poly = insert_intersections_along_edges(poly, all_lines)
        refined_polygons.append(refined_poly)

    # Triangulate the refined polygons.
    triangulated_polygons = []
    for poly in refined_polygons:
        points = np.array(poly.exterior.coords[:-1])  # Omit the repeated closing vertex.
        segments = [(i, i + 1) for i in range(len(points) - 1)] + [(len(points) - 1, 0)]

        # Triangle-Input vorbereiten
        tri_input = {"vertices": points, "segments": np.array(segments)}
        tri_output = triangle.triangulate(tri_input, "p")


        if "triangles" in tri_output:
            for tri in tri_output["triangles"]:
                tri_points = [tuple(tri_output["vertices"][i]) for i in tri]
                triangulated_polygons.append(Polygon(tri_points))

    return refined_polygons, triangulated_polygons


def plot_max_omega_el(RS):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import numpy as np

    fe_omega = 1
    max_omega = max([el.omega_el_max for el in RS.elements])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.axis('off')

    cells = RS.elements

    # Normalize the omega values to the range [0, 1]
    # norm = Normalize(vmin=0.5, vmax=4)
    norm = Normalize(vmin=20, vmax=70)
    cmap = plt.get_cmap('Purples')

    for cell in cells:
        AABB = [[cell.u_min, cell.u_max], [cell.v_min, cell.v_max]]
        cellAABB = np.array(AABB)
        x = cellAABB[0, [0, 1, 1, 0, 0]]
        y = cellAABB[1, [0, 0, 1, 1, 0]]

        # Rahmen zeichnen
        ax.plot(x, y, color='black', linewidth=0.5)

        # Populate only the element with the highest eigenfrequency.
        if cell.omega_el_max == max_omega:
            ax.fill(x, y, color='red')
            # plt.text(np.mean(cellAABB[0]), np.mean(cellAABB[1]), round(max_omega, 2),
            #          color='black', ha='center', va='center', fontsize=10)

    # print('Max El Omega:', np.max(omegas_max))
    for Brep in RS.BReps:
        x, y = Brep.polygon.exterior.xy
        ax.plot(x, y, linewidth=1, color='black')

    # # Create the colorbar
    # sm = ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    # # cbar.set_label('Maximale Eigenfrequenz')

    #import tikzplotlib
    plt.show()
