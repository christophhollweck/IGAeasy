# Raasch hook

`rm5.py` is the fitted, untrimmed counterpart of
`../../trimmed/raasch_hook_rm5.py`. Both use exactly the same polynomial surface
on u=[0,1], v=[1/12,11/12], material, load, weak support and observation point.
The fitted variant extracts that rectangle by knot insertion; it does not call
the trimming routine. Parameter coordinates stay unchanged.

The original trimmed variant cuts this rectangle out of an extended background.
It therefore has the same visible outline. Trimming describes the relation to
the background elements, not a visibly different physical structure.

The numerical 3D reference is 5.020 inches; it is not an exact shell solution.
It is the C3D20R 20x144x2 continuum result in
[Abaqus' Raasch benchmark, Table 1](https://abaqus.uclouvain.be/English/SIMACAEBMKRefMap/simabmk-c-raasch.htm).
That comparison uses the loaded-edge centerline, E=3300 psi, nu=0.35,
thickness 2 inches, width 20 inches and resultant shear 1 pound. The source also
reports a small limiting difference between shell and continuum responses;
refinement alone need not drive the shell difference to zero.
Both report the reference difference and the independent radius-fit diagnostic.
Run `../../verification/trimmed/raasch_hook_convergence.py` to compare both variants.
