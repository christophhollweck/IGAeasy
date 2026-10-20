# Benchmark reference audit

Checked on 2026-10-16. A small algebraic residual checks the linear solve;
it does not establish correctness of kinematics, loading or boundary conditions.
Agreement between fitted and trimmed meshes is likewise an equivalence check,
not an independent reference solution.

| Problem | Quantity and reference | Interpretation |
| --- | --- | --- |
| Simply supported unit plate | Center normal deflection and displacement L2 norm against Navier series; RM includes shear, KL omits it | Analytical series, evaluated with finite truncation. Material, thickness and pressure now come from the setup. |
| Scordelis roof | Free-edge midpoint vertical deflection: conventional RM -0.3024, KL -0.3006 | Rounded numerical comparisons; fixed fitted-circle geometry and shell theory affect the limiting difference. |
| Open pinched hemisphere | Loaded equatorial radial displacement: 0.093597 | Published extrapolated KL value. The KL discrepancy is investigated in hemisphere_kl_diagnostic.md. RM point-load responses are finite-mesh comparisons; the ideal shear-flexible point-load response has no finite displacement limit. |
| Raasch hook | Loaded-edge centerline displacement in the load direction: 5.020 inches | Refined 3D continuum result, not an exact RM shell result. |
| Cook membrane | Loaded-edge midpoint: u_y E t / P approximately 23.91 for plane stress, nu=1/3 | Rounded numerical comparison. The existing 23.90–23.96 band is a comparison range, not a certified error bound. |
| Trimmed flat cantilever | End deflection and displacement L2 norm against bending-plus-shear polynomial | Applies to the specified nu=0 strip and loading; do not generalize it to arbitrary plates or Poisson ratios. |

## Sources and matching assumptions

- [Altair Scordelis verification](https://help.altair.com/hwsolvers/os/topics/solvers/os/scordelis_lo_roof_verification_r_2.htm):
  gravity load 90, nu=0, target free-edge midpoint displacement 0.3024.
- [Krysl, Benchmarking Computational Shell Models](https://link.springer.com/article/10.1007/s11831-022-09798-5):
  model-dependent roof and open-hemisphere comparison context. The detailed
  hemisphere conventions are recorded in its benchmark README.
- [Abaqus Raasch benchmark](https://abaqus.uclouvain.be/English/SIMACAEBMKRefMap/simabmk-c-raasch.htm),
  Table 1: 5.020 for a C3D20R 20x144x2 continuum mesh. Radii 14/46 inches,
  arcs 60/150 degrees, width 20, thickness 2, E=3300, nu=0.35, resultant 1.
  The source itself reports a small shell/continuum limiting difference.
- [STRIX Cook verification](https://dcr-st.com/verification/SB3.html):
  the same (0,0), (48,44), (48,60), (0,44) panel, plane stress and loaded-edge
  midpoint normalization. This software verification page supports the existing
  approximate comparison; it is not an analytical solution or a rigorous interval.

## Reference checks

`tests/shells/test_plate_reference.py` checks the independent unit-rigidity
square-plate coefficient, Fourier load amplitude, load/modulus/thickness scaling,
and 200-versus-400-mode truncation at the center and interior points.
For the default setup the center truncation difference is about 5.04e-12 (RM)
and 1.23e-15 (KL). Very small discretization differences must therefore not be
interpreted without considering series truncation and floating-point error.

Nitsche plate study, cubic basis, default rotation about (1,2,3):

| Formulation | L2, 2x2 | L2, 4x4 | L2, 8x8 | Rate 4x4 to 8x8 |
| --- | ---: | ---: | ---: | ---: |
| RM6 | 1.671e-5 | 1.893e-6 | 9.531e-8 | 4.31 |
| RM5 | 1.671e-5 | 1.893e-6 | 9.531e-8 | 4.31 |
| RM5 hierarchical | 2.349e-5 | 2.896e-6 | 2.151e-7 | 3.75 |
| KL | 2.694e-5 | 3.611e-6 | 2.117e-7 | 4.09 |

The rates are compatible with the cubic displacement comparison h^4 under the
usual regularity and stability assumptions. Three coarse meshes do not establish
the asymptotic rate. Linear residuals were below 4e-14. No output figures were
saved and no interactive windows were opened during this check.

The existing shell tests separately compare rotations, strong/weak supports,
and refinement. Passing these finite checks cannot exclude every implementation
error. In particular, rounded benchmark differences should not be used to demand
an asymptotic convergence rate to zero.
