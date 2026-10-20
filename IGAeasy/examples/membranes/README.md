# Membranes

- `benchmarks/cooks_membrane.py`: complete fitted Cook panel with weak support.
- `trimmed/cooks_membrane.py`: the identical physical panel cut from a rectangular
  background mesh. Its slanted edges cross elements; the straight root and loaded
  end coincide with the background boundary to avoid unnecessary tiny root cuts.
- `verification/benchmarks/cooks_membrane_convergence.py`: refinement comparison of both.

Both Cook examples use plane stress, E=100, nu=1/3, thickness=1 and a uniform
end shear with total force P=1. They report the loaded-edge midpoint response,
the normalized value u_y E t/P, the upper-corner response separately, the
rounded reference value and its difference. The numerical midpoint reference
23.91 lies in the reported literature range 23.90–23.96; none is an analytical
error reference. Differences need not decrease monotonically towards a rounded
comparison value. See [the documented reference](https://dcr-st.com/verification/SB3.html).

Each single setup opens one window; color, deformation, element boundaries and
weak supports can be toggled. The trimmed variant uses Nitsche by default;
both also accept `method='penalty'`. No automatic exports are written.

The same organization as for shells applies:

- `benchmarks`: complete, fitted single-mesh setups with reference comparisons.
- `trimmed`: complete single-mesh setups with cut boundaries.
- `verification/benchmarks`: reference-based mesh studies.
- `verification/manufactured`: manufactured continuum fields and error norms.
- `verification/boundary_conditions`: penalty and weak-boundary checks.

The old `clamped_membrane`, `membrane_eigenmodes` and
`nonlinear_clamped_membrane` solver modules are preserved under
`investigation_setups/legacy_examples/membranes`. Their existing investigation
callers now import them there. They are not presented as standalone validated
benchmarks. No solver behavior or paper-study parameters were changed.
