# Curved natural edges, RM6 double curvature and balance checks

## Entry points

- `IGAeasy/examples/shells/verification/boundary_conditions/free_edge_convergence.py`:
  RM5/KL parabolic strip with one Nitsche root clamp and three naturally loaded
  edges. Run the file for convergence plots and one switchable geometry window.
  No export is performed. `domain=trimmed_domain`, `family='LR'` and
  `local_refinement=True` select the moderate slanted/local comparison.
- `IGAeasy/examples/shells/verification/manufactured/rm6_cap_convergence.py`:
  independent RM6 reference on a quadratic graph with two nonzero principal
  curvatures. The default curvature (.4,-.2) gives a saddle. It plots once after
  the convergence study and reports displacement/director/rotation errors and
  drilling energy.

The natural strip edges have unknown displacement/director traces and prescribed
nonzero forces/moments. They are not kinematically fixed and are not claimed to
be traction-free. A zero-traction physical cantilever is a different problem.

## Natural load convention

`ShellEdgeLoad(line, data, formulation='KL'/'RM5')` integrates

    integral [force.u + moment.delta_d] ds.

The callback supplies six Cartesian components per physical edge length. Moment
is conjugate to director change, not RM6 rotation. For KL the director trace
depends on displacement derivatives. Keeping its full tangential work includes
endpoint/corner effects implicitly. The callback accepts raw traction, not an
already condensed effective traction with a second copy of the same corner work.
No new boundary constraint or penalty is introduced on a natural edge.

The existing `LineLoad` addresses coefficient components and cannot express this
KL derivative work or the curved RM5 physical director dual map directly. The new
helper performs only this physical virtual-work integration; it does not compute
the reference forces, stiffness or a Nitsche flux.

The parabolic reference classes now supply their own analytical `edge_data` and
`energy_density`. The KL and RM5 fields remain different and have their own body
loads. Hierarchical references explicitly reject these new KL-only functions
rather than silently inheriting incorrect load or energy data.

## Independence of the RM6 reference

`RM6CapReference` uses X=(u,v,h), h=a*u^2+b*v^2 and
u0=(0,0,A*h^2), omega=0. Since grad_s(h)=(I-nn.T)e_z, the normal spin vanishes
analytically. This field therefore has zero continuum drilling energy.

The three-dimensional virtual-work derivation uses the offset geometry X+z*n,
its dual frame and isotropic shell plane stress. Integrating its stress fluxes
through thickness and taking surface divergence gives all three body forces and
all three body moments. Moments are not set to zero just because omega=0.
Complex-step differentiation acts on continuum expressions, without B matrices
or a desired discrete solution. The thickness rule matches the bulk's three
Gauss points; comparison with eight points changes sampled body loads by at most
5.5e-11 relative at the three checked locations. This is a sampled check, not a
global quadrature bound.

## Measured convergence

Default general spatial rotation, cubic bases, untrimmed THB, 4x4 to 8x8:

| Case | Displacement L2, 4x4 | Displacement L2, 8x8 | Director L2, 8x8 |
| --- | ---: | ---: | ---: |
| RM5 natural strip | 7.576e-7 | 5.112e-8 | 1.032e-8 |
| KL natural strip | 1.522e-6 | 8.767e-8 | 2.597e-6 |
| RM6 saddle | 1.338e-7 | 8.868e-9 | 1.277e-8 |

Displacement rates are approximately four. KL director error, involving a
displacement derivative, has approximately third order here. The RM6 drilling
energy fraction decreases from 1.69e-7 at 2x2 to 6.19e-11 at 8x8.

On the slanted trimmed, locally refined cubic patch, physical L2 errors are
below 1e-9; THB/LR agree. The final contour has a minimum parametric material
fraction of .066 on the tested locally refined mesh; accidental very small
cuts were removed from this reference case. The slanted KL free edges
have nonzero tangential moments and corner work. The existing analytical
four-corner force test separately verifies that `ShellEdgeLoad` retains it.
The RM6 moderate trimmed 4x4 THB/LR displacement error is 1.226e-7 in both cases.
No extreme-cut robustness or asymptotic rate follows from this single mesh.

## Independent balance checks

The strip reports continuous body and edge resultants from analytical data,
including physical moment x cross force + n cross director_moment. This is
separate from discrete reactions f_boundary-K_boundary*u. The test checks
continuous force/moment equilibrium, the weak root force against the analytical
root force, and bulk strain energy against independently integrated energy density.

Weak boundary work is included in 2*U = applied_work + boundary_work. At 8x8
the relative discrete energy balance is below 1e-14 for both strip formulations;
continuous force/moment imbalances are below 5e-14. These absolute values apply
to this problem's force/load scales and are not universal tolerances.

Virtual rigid-rotation work also compares root moments. It is exact in the KL
displacement space. For curved RM5, director coordinates of rigid rotation are
sampled at Greville points and are only approximate; this must not be presented
as a universal exact moment recovery method. On these meshes the errors are
below 5.1e-10 (KL) and 4e-13 (RM5).

A selected 4x4 penalty comparison also satisfies the balances but adds boundary
approximation error: displacement L2 is 7.586e-7 (RM5) and 1.555e-6 (KL), versus
7.576e-7 and 1.522e-6 for Nitsche. No equal-rate guarantee is claimed for fixed
penalty factors.

## Tests and limits

`test_natural_shell_edges.py` checks closed-form flat limits, explicit KL corner
forces, curved rates, rotation, physical balances and the slanted local THB/LR
patch. `test_rm6_cap_reference.py` checks continuum flux divergence, zero normal
spin, loaded convergence, drilling decay, rotation and moderate trimmed THB/LR.
The eleven targeted new/adjacent tests pass. The complete 328-case regression
passed in 917.718 seconds; the final contour/diagnostic adjustment also passed
the four affected tests separately. Both main study entry points ran with a small
mesh sequence and plotting disabled.

Offscreen geometry views of both new examples were rendered and inspected.
The examples do not automatically save these checks. The viewer's existing
checkboxes remain unchanged; this pass does not add moment arrows to the viewer.
No spline kernel or production shell stiffness was modified. These checks do not
resolve the hemisphere/Raasch literature discrepancies, nor do they cover every
free-edge shape, arbitrary mixed masks, nonlinear shells or multipatch coupling.
