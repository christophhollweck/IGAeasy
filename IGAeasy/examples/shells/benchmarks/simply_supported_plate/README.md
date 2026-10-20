# Simply supported unit plate

Each of `rm6.py`, `rm5.py`, `rm5_hierarchical.py` and `kirchhoff_love.py` is a
complete single-mesh setup. `verification/benchmarks/simply_supported_plate_convergence.py` calls those same setups with Nitsche
supports on progressively finer grids.

The square has side length 1, thickness 0.1, E=10920, nu=0.3 and uniform
transverse load -1. Cubic basis functions and full bulk quadrature are used.
Hard simple supports prescribe translations and the tangential physical
director; the normal bending moment is naturally zero. KL imposes the
compatible tangential director through the displacement trace.

The setup rotates about (1,2,3), including loads and supports. Evaluation uses
the original plate-normal direction. `reference.py` contains only the independent
Navier series, with shear for RM and without shear for KL. Each calculation
reports both center displacement difference and a physical displacement L2 norm.
The reference uses the material, thickness and pressure specified at the top of
each setup. Its geometry remains the unit square. The 200 odd modes per direction
are checked against 400 modes at the center and interior points; truncation is
below 1e-10 there, not an assertion of an exact infinite-series evaluation.

Select `nitsche` or `penalty` in `run(...)`. The safety factors are respectively
4 and 400 times a computed local bound. A finite penalty adds boundary error.
The study compares all formulations against their respective theory.
