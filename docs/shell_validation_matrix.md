# Shell validation: evidence and remaining work

Current index, 2026-10-16. This is the authoritative coverage summary, not a
blanket release approval. Follow the linked tests for exact parameters and
assertions. A test with THB and LR plus another with Nitsche and penalty does
not establish every combination of those settings.

Primary formulations: RM6, RM5, RM5_hier and canonical KL. RM5_v2 remains an
internal kinematics comparison. All claims concern linear static shells,
regular polynomial geometry and the supported isotropic material law.

## Physical benchmarks

| Problem | Formulations | Boundary conditions | Geometry/discretization | Evidence and limit |
| --- | --- | --- | --- | --- |
| Simply supported plate | All four | Public Nitsche/penalty setups; automated Nitsche refinement/rotation | Untrimmed cubic THB | [Setup tests](../tests/shells/test_standalone_shell_examples.py), [Navier tests](../tests/shells/test_plate_reference.py). Independent L2 and center response, RM/KL-specific series; not a penalty error guarantee. |
| Scordelis roof | All four; RM5_v2 internally | Strong, penalty, Nitsche; same-mesh field/orientation comparisons | THB; selected LR and trimmed-hole cases in reference fixture | [Roof tests](../tests/shells/test_scordelis_weak.py), [conventions](../IGAeasy/examples/shells/benchmarks/scordelis_roof/README.md). Rounded references apply only without a hole; strong/weak agreement is not an independent reference. |
| Open hemisphere | KL finite-limit benchmark; RM6/RM5/RM5_hier finite-mesh comparisons | Nitsche/penalty setups, automated RM strong comparison, independent strong KL reflection | Untrimmed cubic THB, fixed fit, rotation | [Hemisphere tests](../tests/shells/test_hemisphere.py), [KL checks](../tests/shells/test_kl_hemisphere_operators.py). KL approaches about .09352; .093597 discrepancy remains open. Ideal RM point-load displacement has no finite limit. |
| Flat cantilever | RM5 | Nitsche root clamp, end traction | Moderately trimmed rotated THB | [Strip tests](../tests/shells/test_trimmed_shell_benchmarks.py). Analytical bending-plus-shear field for this nu=0 strip only. |
| Raasch hook | RM5 | Nitsche root clamp, end shear | Fitted/trimmed versions of same polynomial fit | [Hook tests](../tests/shells/test_trimmed_shell_benchmarks.py), [conventions](../IGAeasy/examples/shells/benchmarks/raasch_hook/README.md). 5.020 is a numerical 3D reference. Fitted/trimmed agreement checks equivalence, not exactness. |

Reference provenance and restrictions:
[benchmark references](benchmark_references.md),
[KL hemisphere diagnosis](hemisphere_kl_diagnostic.md).

## Analytical and operator evidence

These are selected cases. Local THB/LR means actual local refinement in the
linked tests, not a performance comparison or coverage of every refinement path.

| Check | Formulations | Boundary conditions | Geometry/discretization | Evidence |
| --- | --- | --- | --- | --- |
| Rigid motion, director kinematics, rotation of resultants | All four; RM5_v2 internally | Bulk/operator checks | Curved skew THB; exact discrete rotations when representable | [Rigid motion](../tests/shells/test_shell_rigid_motion.py), [RM5 kinematics](../tests/shells/test_rm5_kinematics.py). Curved RM5 director coefficients need not represent a rigid rotation exactly. |
| Nonzero membrane/bending patches | RM6, RM5 | Primarily Nitsche; selected penalty rigid/energy checks | Rotation, moderate trimming, nonuniform/local THB/LR | [RM6 patches](../tests/shells/test_shell_weak.py), [RM5 patches](../tests/shells/test_rm5_weak.py). Curved rigid motion is not a loaded manufactured solution. |
| Clamped plate: physical errors, reactions, energy | RM6, RM5; RM5_v2 internally | Strong/penalty/Nitsche | Affine THB/LR study; automated balance assertions on the default THB case | [Plate comparison](../IGAeasy/examples/shells/verification/boundary_conditions/plate_boundary_comparison.py), [balance/convergence tests](../tests/shells/test_penalty_free_shells.py). Despite its historical filename, the latter tests the three standard methods too. Manufactured loading includes matching body moments. |
| Loaded parabolic surface | RM6, RM5; RM5_v2 internally | Nitsche; penalty bias/refinement | Rotation; local trimmed THB/LR loaded patches explicitly cover RM5 variants | [Curved RM convergence](../tests/shells/test_rm_curved_convergence.py). Separate RM6 curved local patches do not establish this entire loaded matrix. |
| Clamped KL/gradient surface and homogeneous plate | KL, RM5_hier | Nitsche; selected penalty/strong plate comparisons | Curvature, rotation, holes, local THB/LR | [KL curved](../tests/shells/test_kl_curved_weak.py), [hierarchical curved](../tests/shells/test_rm5_hier_clamped_weak.py), [homogeneous reference](../tests/shells/test_homogeneous_shell_reference.py), [gradient penalty](../tests/shells/test_shell_gradient_penalty.py). |
| Cap/saddle displacement and director fields | All four; RM5_v2 in RM study | Nitsche full clamping | Doubly curved THB, general rotation, independent continuum loads | [RM paraboloid](../tests/shells/test_paraboloid.py), [gradient paraboloid](../tests/shells/test_gradient_paraboloid.py), [RM6 saddle](../tests/shells/test_rm6_cap_reference.py). The RM6 reference additionally checks moments, vanishing drilling and one moderate trimmed THB/LR case. |
| Moderate trimming and actual local refinement | RM5, RM5_hier, KL | Nitsche | Doubly curved graph, outer cut/inner hole, THB/LR | [Trimmed fields](../tests/shells/test_trimmed_manufactured_shell.py), [local refinement](../tests/shells/test_local_shell_refinement.py). Tensor-product equality and local convergence are separate checks. |
| Mixed displacement and natural moments | RM5, RM5_hier, KL | Nitsche convergence; penalty study and selected bias tests | Doubly curved graph; selected local trimmed THB/LR | [Doubly curved mixed](../tests/shells/test_doubly_curved_mixed_shell.py), [KL mixed](../tests/shells/test_kl_mixed_weak.py), [hierarchical mixed](../tests/shells/test_rm5_hier_mixed_weak.py). Displacement is prescribed on these edges; they are not fully free edges. |
| Moving normal displacement, free tangential displacement | RM5, RM5_hier, KL | Nitsche; complementary natural forces | Double curvature, rotation, selected local trimmed THB/LR | [Normal support tests](../tests/shells/test_normal_shell_boundaries.py). KL direction restrictions remain part of the API. |
| KL corner work and corrected curved force | KL | Full/mixed/plate Nitsche consistency; independent work identity | Rotation, double curvature, moderate trimming, THB/LR | [Corner tests](../tests/shells/test_kl_corner_work.py), [derivation](kl_corner_work.md). No general corner-only support interface is implied. |
| Naturally loaded curved edges and balance | RM5, KL | One weak root clamp, three natural edges; selected penalty comparison | Rotated parabolic graph, slanted trim/local THB/LR patch | [Natural edge tests](../tests/shells/test_natural_shell_edges.py), [conventions](curved_natural_edges.md). Independent force/moment and energy data; naturally loaded does not mean zero traction. RM5 virtual moment recovery is approximate. |
| Conditioning and stabilization | All four in selected cases | Energy bounds, split stabilization, selected solver comparisons | Small cuts/rigid/near-null modes; coverage varies by formulation | [RM energy rank](../tests/shells/test_shell_energy_rank.py), [split stabilization](../tests/shells/test_split_stabilization.py), [cut studies](../tests/shells/test_shell_cut_fraction.py). Extreme-cut failures remain possible, especially KL. No basis removal or hidden regularization. |

## Completed focused pass

### 1. Loaded curved edges with genuinely free traces

Implemented for RM5/KL with independent parabolic continuum fields and raw
force/director-moment edge loads. The three natural edges have no essential
trace constraints. Tests cover convergence/rotation and a slanted, locally
refined trimmed THB/LR patch with nonzero KL corner work. `ShellEdgeLoad` supplies
the derivative/director work unavailable in the coefficient-component LineLoad.

### 2. Close the RM6 loaded doubly curved gap

Implemented independent three-dimensional RM6 stress-flux equilibrium on a saddle.
Body forces and moments include thickness integration. The continuum field has
zero normal spin/drilling; numerical drilling energy decreases under refinement.
Rotation, convergence and one moderate trimmed THB/LR comparison are tested.

### 3. Curved reaction/energy balance and focused regression

Implemented independent continuum force/moment resultants, analytical strain
energy and the discrete weak boundary-work balance for the natural strip. Root
force and virtual-rotation moment comparisons are reported with their approximation
limits. One selected penalty comparison was also run. The eleven targeted new
and neighboring tests pass; full regression status is recorded below.

Measured values and derivations: [curved natural edges](curved_natural_edges.md).
Next: review the public examples together after regression completes. A single
curved reference does not cover all possible free/mixed boundary configurations.

## Outside this pass

Multipatch, nonlinear shells, dynamics/mass matrices, locking/thickness studies,
singular parameterizations, new basis kernels and THB/LR performance comparison.
The hemisphere/Raasch reference differences remain open; no parameter is fitted
to a target displacement to remove them.

## Release validation

The development suite and the smaller structural export suite have different
sizes. Check the release execution record before claiming a completed run.
The export excludes heat-dependent tests and private development history.
