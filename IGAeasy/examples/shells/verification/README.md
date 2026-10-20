# Verification and convergence studies

The [validation matrix](../../../../docs/shell_validation_matrix.md) links each
numerical claim to its tests and separates remaining work from current coverage.

Only the optional `run_benchmarks.py` launcher remains directly in this folder.
Single-mesh physical setups belong in `../benchmarks` or `../trimmed`.

| Folder | What it checks |
|---|---|
| `benchmarks` | Plate, roof and hemisphere refinement against reference responses |
| `trimmed` | Cut-domain norms, fitted/trimmed comparisons, flat strip and Raasch hook |
| `manufactured` | Independent continuum fields and genuine local THB/LR refinement |
| `boundary_conditions` | Strong/penalty/Nitsche, mixed moments and moving normal supports |

Start with `manufactured/paraboloid_convergence.py`. Its curvature and mixed-term
settings cover caps, saddles and the former `polynomial_shell_convergence.py`
variant. That redundant starter is retained privately in legacy shell studies.
`manufactured/local_shell_convergence.py` adds actual local refinement;
`manufactured/rm_curved_convergence.py` retains the cylindrical RM6/RM5 check.
`manufactured/rm6_cap_convergence.py` adds independent double-curvature RM6 loads,
including body moments and a zero-drilling continuum field.

Boundary studies are retained because they exercise different operators:
`mixed_shell_convergence.py` prescribes displacement and supplies natural director
moments; `normal_support_convergence.py` prescribes only moving-normal displacement;
`plate_boundary_comparison.py` checks strong versus weak conditions, energy and
force balance. The `gradient_*` and `hierarchical_*` studies isolate KL and
hierarchical director boundary operators. These are diagnostic verification
studies, not additional physical benchmark geometries.
`boundary_conditions/free_edge_convergence.py` leaves three curved edges
kinematically unconstrained and applies independently derived forces/moments.
It also checks energy and continuum force/moment balance; see
[the conventions and results](../../../../docs/curved_natural_edges.md).

The manufactured loads come from independent continuum equilibrium, never from
multiplying the assembled stiffness by desired coefficients. Passing a study
cannot exclude all implementation errors; use norms, refinement, rotation checks
and conventional benchmark responses together.
