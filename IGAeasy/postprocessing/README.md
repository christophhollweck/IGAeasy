# Shell result viewer

`plot_solution(space, deformation_scale=50.)` opens one window. Complete shell
setups save their displacement and director/rotation coefficients with:

```python
update_cps(space, coefficients.ravel(), dof_per_cp=5, formulation='RM5')
```

Supported formulation tags are `RM6`, `RM5`, `RM5_v2`, `RM5_hier` and `KL`.
With a single initial field, checkboxes select displacement, local section stress
or local section strain. Exactly one field stays selected. A separate checkbox
group selects cell means, element interpolation or global L2 projection.
There are no sliders or section-layer controls; section fields in the viewer
refer to the middle surface (z = 0).

Other checkboxes control colors, surface, edges, supports and deformation.
Changing deformation does not change physical scalar values. Displacements
always use their spline representation.

Cell means are weighted by physical quadrature area and preserve interelement
jumps. Element interpolation uses the existing linear RBF approach separately
inside each element. Global L2 uses the shared quadrature-weighted projection
onto the spline space. These last two representations may overshoot; global
projection can smooth genuine jumps. Nothing is silently clipped.
The selected scalar, including von Mises, is reconstructed at the GPs before
interpolation or projection.

## Meaning of section fields

Stress uses the local orthonormal shell frame and the section reconstruction
`sigma = N/t + 12*z*M/t**3`, with average transverse shear `Q/t` included in
von Mises when the formulation supplies Q. Strain 11/22/12 follows isotropic
plane-stress compliance. Strain 12 is tensor shear epsilon_12 = gamma_12/2.
KL supplies no constitutive transverse shear field.

For RM6 this is a linear section reconstruction from integrated resultants.
Curvature-dependent through-thickness variations beyond this reconstruction
are not represented. The viewer does not claim an exact arbitrary-depth 3D stress.

GP results are prepared on the first stress/strain selection. Only requested
representations are cached. No stiffness reassembly or new solve is triggered.
Multiple fixed subplots and explicit custom projected fields remain available
through the existing plotting API; their viewer has no single-field selector.

## ParaView export

Export is explicit and never runs automatically when plotting:

```python
from IGAeasy.postprocessing.result_export import export_shell_results

export_shell_results(space, 'shell.vtm',
                     fields=('stress_von_mises', 'stress_11', 'strain_11'), layer='Top')
```

Open the VTM file in ParaView. The three named blocks keep cell means, element
interpolation and global L2 projection separate. All use reference geometry;
apply **Warp by Vector** with the `displacement` array to deform it. Keep the
VTM file and its generated companion directory together. Stress components use
local shell frames; displacement uses global coordinates. Section depth can be
chosen explicitly at export (`Bottom`, `Middle`, `Top`).

For raw integration-point resultants without spatial averaging, the existing
`export_integration_point_results(space, 'gauss_points.vtp')` remains available
after stress recovery. Its local frame vectors accompany the resultants.
