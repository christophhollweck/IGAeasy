# Trimmed shell benchmarks

Start with `flat_cantilever_rm5.py`. Each script contains its complete setup and
opens one result window. Nitsche is the default; penalty is also available.

- `flat_cantilever_rm5.py`: slanted, flat strip with a uniform end line load.
  Reports the analytical RM tip displacement, absolute/relative tip error and
  the displacement L2 error. With nu=0 there is no transverse Poisson coupling:
  the cylindrical bending field is an exact RM plate solution, not merely a
  slender-beam approximation. Both bending and transverse shear are included.
  The cubic solution is represented exactly by degree 3; errors reach roundoff
  even on coarse meshes. This is a consistency test, not a rate study.
- `raasch_hook_rm5.py`: two tangential cylindrical segments, with radii 14 and 46,
  width 20 and thickness 2 (inches), E=3300 psi, nu=.35. A total end shear of
  one pound acts along the width. The physical strip is cut from an extended
  background. A fixed C1 polynomial fit retains the curvature jump at the join;
  refinement inserts knots without changing that geometry. Radius approximation
  error is reported separately. The root clamp and load rotate with the shell.
  The numerical 3D comparison value is 5.020 inches, not an exact shell solution.
  Current RM5 values for refinement levels 0/1/2 are 5.00362/5.03611/5.03788;
  the remaining 0.36% difference is unresolved and is not called an exact error.
  Source: [Abaqus Raasch benchmark](https://abaqus.uclouvain.be/English/SIMACAEBMKRefMap/simabmk-c-raasch.htm).

Refinement studies are in `../verification/trimmed/flat_cantilever_convergence.py`
and `../verification/trimmed/raasch_hook_convergence.py`.
The previous manufactured curved-hole setup remains useful for independent L2
convergence checks and now lives in `../verification/trimmed/manufactured_hole_rm5.py`.
The previous curved cantilever with no reference is retained under
`investigation_setups/legacy_examples/shell_studies`.

For the flat strip, w(x)=F*x^2*(3*L-x)/(6*E*I)+F*x/(kappa*G*A),
with I=b*t^3/12 and A=b*t. Changing nu requires reconsidering this plate reference.
No export runs automatically. Checkbox controls toggle deformation, supports,
trim boundaries, result fields and element edges in the same window.
