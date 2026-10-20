# Scordelis–Lo roof

Start with `rm5.py`. Each formulation script contains its complete geometry,
material, loading, support, integration, assembly, solve and plotting sequence.
The same `run(...)` function is called by `verification/benchmarks/scordelis_roof_convergence.py`; no separate problem
solver is hidden behind the setup.

- `rm6.py`: six-parameter Reissner–Mindlin shell.
- `rm5.py`: five-parameter Reissner–Mindlin shell.
- `rm5_hierarchical.py`: hierarchical five-parameter shell.
- `kirchhoff_love.py`: canonical Kirchhoff–Love shell.
- `verification/benchmarks/scordelis_roof_convergence.py`: all four formulations with symmetric Nitsche diaphragms.
- `rm6_stresses.py`: RM6 section-stress view with interactive field/layer selection.

The full roof has radius 25, length 50, opening angle 80 degrees and thickness
0.25. E=4.32e8, nu=0, uniform vertical load=-90 per physical surface area.
End diaphragms fix the original x/z translations. Director rotations remain
free. One longitudinal translation is fixed only to remove the remaining rigid
motion. Geometry, load direction and diaphragm directions rotate together by
0.6 radians about (1,2,3).

The polynomial arc is fitted once on a 4x4 grid. Knot insertion gives nested
4/8/16/32 grids without changing that fitted geometry. Full bulk quadrature is
used; RM6 retains the original three-point Lobatto thickness rule.

Compare free-edge midpoint displacement with the rounded numerical reference
-0.3024 for RM and -0.3006 for KL. These are benchmark differences, not exact
error norms or guaranteed asymptotic convergence rates. The fitted geometry
and the distinction between RM and KL remain relevant at fine meshes.

Single setups support `nitsche`, `penalty` and `strong`. The default Nitsche
safety factor is 4 times the local computed bound; penalty uses 400. No arbitrary
fixed dimensional penalty parameter is supplied. Strong diaphragms provide an
independent comparison. Nothing is exported automatically.

Reference context: the conventional 0.3024 value is also used in [Altair verification](https://help.altair.com/hwsolvers/os/topics/solvers/os/scordelis_lo_roof_verification_r_2.htm). [Krysl's benchmark study](https://link.springer.com/article/10.1007/s11831-022-09798-5) discusses a more refined shear-flexible estimate near 0.30200-0.30204 and about 0.3006 for shear-rigid shells. We retain the conventional comparison value; its difference is not an exact discretization error.
