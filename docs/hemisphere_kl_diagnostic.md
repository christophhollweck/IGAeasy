# KL open-hemisphere investigation

## Finding

No implementation error explaining the approximately 0.08 percent difference
from 0.093597 was found in these checks. The independently assembled, exact-sphere
comparison approaches about 0.09352 too. This narrows the remaining question to
reference/model differences and any effects not covered by these checks; it is
not proof that the production formulation is free of all errors. The comparison
value in the public example has not been changed to match our result.

## Reproduce

Run `IGAeasy/examples/shells/verification/boundary_conditions/hemisphere_kl_diagnostic.py`.
Its `run(n_elements=..., degree=..., quadrature_order=..., geometry_elements=...,
exact_sphere=...)` returns diagnostics without plotting or exporting.

The diagnostic uses Cartesian even/odd reflection by coefficient elimination.
This is a tensor-grid comparison, not a general strong KL boundary interface.
It has no Nitsche terms and fixes one vertical rigid translation. Default bulk
assembly uses the production KL element. Degree elevation preserves the fixed
geometry; knot insertion then refines the field.

With `exact_sphere=True`, an independent Cartesian plane-stress integrator uses
analytical spherical tangents and second derivatives. Displacement functions
remain polynomial. It does not add rational-basis support or alter the production
geometry/basis kernel. Normal variation directly gives curvature variation.

## Results, 2026-10-16

All values below use radius 10, opening 18 degrees, thickness .04, E=6.825e7,
nu=.3 and unit quarter forces. The exact-sphere comparison uses the same angular
coordinate convention and physical force locations.

| Mesh | Degree | Points/direction | Geometry | Strong radial displacement |
| --- | ---: | ---: | --- | ---: |
| 16x16 | 3 | 4 | Fixed cubic fit, base 8 | .093408134065 |
| 16x16 | 3 | 6 | Same fit | .093408086689 |
| 16x16 | 4 | 5 | Same fit, elevated degree | .093505425491 |
| 32x32 | 3 | 4 | Fixed cubic fit, base 8 | .093513152322 |
| 32x32 | 4 | 5 | Same fit, elevated degree | .093520651687 |
| 64x64 | 3 | 4 | Fixed cubic fit, base 8 | .093521724768 |
| 32x32 | 3 | 4 | Fixed cubic fit, base 16 | .093511003906 |
| 16x16 | 3 | 4 | Analytical sphere | .093402000335 |
| 32x32 | 3 | 4 | Analytical sphere | .093510897594 |
| 64x64 | 3 | 4 | Analytical sphere | .093519539998 |

- Increasing bulk quadrature from 4 to 6 at 16x16 changes displacement by 4.74e-8.
  It cannot explain a discrepancy of roughly 7.5e-5.
- At 64x64 the fixed fit changes the response by 2.18e-6 relative to the exact
  sphere. A finer fit at 32x32 moves toward the exact-sphere result, rather than
  toward 0.093597. Radius error alone was not used as an accuracy argument.
- Graderaising and mesh refinement both approach approximately .09352.
  Richardson estimates from the three cubic levels are roughly .09352249 for
  the fixed fit and .09352029 for the analytical sphere. These are estimates,
  not exact limits or certified error bounds.
- Reduced-system relative residuals in the strong comparison are below 3.1e-9;
  opposite radial responses agree to about 1e-11. Conditioning still limits
  numerical precision; extra printed digits are for reproducibility.

The strong/Nitsche difference decreases with refinement. At 8x8 the two radial
responses are .090281729439 and .090414853166; their difference is 1.33e-4.
At 16x16 the difference is 7.76e-6; at 32x32 it is 5.04e-7 (Nitsche .093513655977).
These are different finite-dimensional
boundary treatments; equality at a fixed coarse mesh is not expected.

## Independent operator checks

`tests/shells/test_kl_hemisphere_operators.py` checks:

- production curvilinear element stiffness against independent Cartesian
  plane-stress stiffness on the same fitted geometry, including interior and
  polar/equatorial cells (relative matrix difference below 2e-14);
- membrane and bending operators against finite variations of the first and
  second fundamental forms;
- exact-sphere principal curvatures of magnitude 1/10 and Cartesian reflection;
- convergence of Nitsche toward the separately eliminated reflection space.

These checks share the polynomial basis evaluator but do not reuse the
production geometric or constitutive transformations in the Cartesian assembly.
They therefore test more than agreement between two callers of the same operator.

## Literature interpretation

[Krysl and Chen (2023), section 2.2](https://link.springer.com/article/10.1007/s11831-022-09798-5)
give .093597 as their extrapolated numerical KL value for the open hemisphere.
They explicitly distinguish it from an analytical solution and discuss uncertainty
in commonly used benchmark values.

[Wenzel (2003), section 5.1.8, printed pages 91–93](https://d-nb.info/969772203/34)
reports a Richardson-extrapolated .09352 with a separate director/DKT formulation
for the open hemisphere, alongside other published results. It is useful
independent corroboration of this scale of response, not an exact solution of
our particular Koiter energy. Differences in shell model and extrapolation
must be resolved before declaring either literature value wrong.

For RM formulations, the ideal point-loaded displacement is not a finite-limit
benchmark (Krysl and Chen, section 4). Agreement of their finite-mesh values with
the published KL number cannot establish RM convergence. A distributed-load
variant would be a different problem and needs its own reference.
