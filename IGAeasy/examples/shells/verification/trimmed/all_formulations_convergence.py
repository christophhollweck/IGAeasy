"""Manufactured solutions on a moderately trimmed curved shell with a hole."""
from shapely.geometry import box
from IGAeasy.examples.shells.verification.manufactured.paraboloid_convergence import main as run_study


# --- Geometry & trimming ---
curvature = (.4, -.2)
mixed_terms = (1.2, -.8)

# Twelfths keep genuine cut cells on every dyadic mesh, with fractions >= 1/9.
domain = box(1/12, 1/12, 11/12, 11/12).difference(box(5/12, 5/12, 7/12, 7/12))

# --- Discretization & formulations ---
net_sizes = (2, 4, 8, 16)
family = 'THB'  # Change to 'LR' for the same geometry and reference fields.
formulations = ('RM5', 'RM5_hier', 'KL')
stabilization = 'split'  # Separate displacement/director bounds for KL and RM5_hier.


if __name__ == '__main__':
    run_study(
        sizes=net_sizes,
        surface_curvature=curvature,
        mixed=mixed_terms,
        formulation_names=formulations,
        domain=domain,
        family=family,
        stabilization=stabilization
    )
