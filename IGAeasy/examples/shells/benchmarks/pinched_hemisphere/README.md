# Pinched open hemisphere

Start with [rm5.py](rm5.py). Each of `rm6.py`, `rm5.py`, `rm5_hierarchical.py`
and `kirchhoff_love.py` contains a complete single-mesh setup.
[convergence study](../../verification/benchmarks/pinched_hemisphere_convergence.py) compares their Nitsche solutions on 8/16/32 grids.
The linear quarter model has radius 10,
thickness 0.04, E=6.825e7 and Poisson ratio 0.3. Its polar opening is 18 degrees.
The geometry/material convention follows the isotropic case in
[Munglani et al., section 5.1](https://arxiv.org/pdf/1508.00347).

The quarter carries +1 in the x direction at (10,0,0), and -1 in the y direction
at (0,10,0). Each force is shared across a reflection cut; the reconstructed full
model therefore has alternating equatorial forces of magnitude 2. The equator
and opening are free. Both meridian cuts have reflection symmetry. A single
vertical displacement removes the remaining rigid translation; it is not a ring
support. Geometry, loads, symmetry planes and this gauge rotate together.
The displacement-magnitude plot depends on this vertical gauge; the radial
benchmark displacements do not. A free shell has no unique absolute translation.

The cited comparison value 0.093597 is a published extrapolated **KL** response
for the open hemisphere, not an exact analytical solution. The familiar 0.0924
belongs to the closed variant. Older open-shell values 0.093 and 0.094 also occur;
the distinction is discussed in
[Benchmarking Computational Shell Models, section 2.2](https://link.springer.com/article/10.1007/s11831-022-09798-5).
Consequently the plot says “difference from published KL value”, not total error.

For RM formulations, displacement directly under an ideal concentrated force is
not a finite-limit convergence benchmark. The cited shell benchmarking study
(section 4) discusses the point-load singularity of shear-flexible shells. The
RM curves here remain finite-mesh comparisons; apparent agreement with the KL
number does not validate their asymptotic limit. Distributed loading would define
a different, regularized problem and requires its own reference.

## Discretization and checks

The basis kernel is polynomial. A cubic tensor-product approximation of the
sphere is built once on the base 8x8 grid, using exact endpoint tangents. Its
maximum sampled radial error is about 5.07e-5. Subsequent meshes only insert
knots, so the geometry does not change with refinement. No rational weights
or singular pole have been introduced. Changing `base_elements` starts a different
fixed-geometry study; keep geometry error separate from solution error.

RM5 and RM5_hier constrain displacement/director perpendicular to each reflection
plane. RM6 instead constrains perpendicular displacement and the two in-plane
rotation components, because its rotation is an axial vector. The KL reflection
operator is specialized for a planar symmetry cut and checks the surface frame;
it is not a generic mask for arbitrary partial KL boundary conditions.

Strong coefficient reflection is included for RM5/RM6 as a comparison. KL/hierarchical
strong reflection would require physical derivative constraints and is not imitated
by fixing their extra coefficients. All formulations use full bulk quadrature.
Unit tests check radius error, regularity, geometry preservation, rotation,
opposite radial responses and approach of weak to strong RM solutions.

The single-formulation scripts plot one geometry view with switchable displacement
field. Visibility controls remain available. `verification/benchmarks/pinched_hemisphere_convergence.py` compares all four
Nitsche solutions; reduce `net_sizes` or `setups` for a quick run. Individual
`run(...)` functions also accept `method='penalty'`. Nothing is exported automatically.

## Observed comparison

At 32x32 with Nitsche the radial responses are about .0935812 (RM6), .0935815
(RM5), .0935891 (RM5_hier), and .0935137 (KL). KL reaches .0935218 at 64x64;
the remaining approximately .08 percent discrepancy from the published value is
not resolved. An independent strong Cartesian reflection space gives .0935132
at 32x32, close to the weak result. Increasing the fixed geometry resolution
changes KL 32 by only about 2.2e-6. The benchmark is therefore a useful diagnostic,
not evidence that every formulation already matches the literature limit.

Further investigation and reproducible independent KL checks are described in
[the diagnostic report](../../../../../docs/hemisphere_kl_diagnostic.md).
The corresponding script is
[`hemisphere_kl_diagnostic.py`](../../verification/boundary_conditions/hemisphere_kl_diagnostic.py).
