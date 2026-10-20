# Linear structural release execution

Source regression completed on 2026-10-19 using Python 3.11.9.
Publication checks updated on 2026-10-20.

- Isolated source-export regression: 298 tests, 0 failures,
  0 errors, 0 skipped, 842.154 seconds.
- Editable development import fallback was disabled for this run.
- Wheel and source archive built and inspected for excluded research files,
  nonlinear modules and private storage paths: no findings.
- All 44 installed example modules imported without development fallback.
- Installed Cook membrane: normalized displacement 23.9387901395,
  relative algebraic residual 1.850e-13; within the documented numerical
  reference interval [23.90, 23.96]. This interval is not an analytical solution.

Scope: linear static membranes and the documented RM6, RM5, hierarchical RM5
and canonical KL shell cases. See [validation matrix](shell_validation_matrix.md)
for coverage, the open KL hemisphere discrepancy, RM point-load limitations,
geometry approximation and extreme-cut limits. Passing tests is not a guarantee
for arbitrary inputs or a replacement for independent engineering verification.

Source line endings are normalized to LF for reproducible repository hashes;
Python syntax trees are checked unchanged by normalization. Licensing and
repository publication are separate decisions from these numerical checks.

## Tested dependency versions

- numpy: 1.26.4
- scipy: 1.15.2
- matplotlib: 3.7.5
- shapely: 2.1.0
- splinepy: 0.1.3
- pyvista: 0.44.2
- Rtree: 1.3.0
- triangle: 20250106
- sympy: 1.14.0
- tqdm: 4.67.1
