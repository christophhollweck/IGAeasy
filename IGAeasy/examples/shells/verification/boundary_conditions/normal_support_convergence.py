"""Moving normal support: tangential translations and director moments stay natural."""
from IGAeasy.examples.shells.verification.boundary_conditions.mixed_shell_convergence import main as run_study


# --- Discretization & optional trimming ---
net_sizes = (4, 8, 16)
domain = None
# from IGAeasy.examples.shells.verification.trimmed.all_formulations_convergence import domain
local_refinements = 0

# Geometry, material and reference amplitudes match mixed_shell_convergence.
# Only n.u is prescribed; matching tangential forces and moments are supplied.
if __name__ == '__main__':
    run_study(sizes=net_sizes, trim_domain=domain, local_steps=local_refinements, support='normal')
