# Material conventions

`LinearElasticMaterial` implements isotropic **plane stress** and shell resultants.
It does not implement general three-dimensional solid elasticity.

| Method | Strain components | Stress or resultant components |
| --- | --- | --- |
| `get_C_voigt_cart_2D` | epsilon_xx, epsilon_yy, gamma_xy | sigma_xx, sigma_yy, tau_xy |
| `get_C_voigt_cart_3D` | epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_yz, gamma_xz | sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz |
| `compute_C_voigt_curv_2D` | e11, e22, 2 e12; transverse shear gamma1, gamma2 | n11, n22, n12; transverse shear q1, q2 |

Engineering shear is gamma_ij = 2 epsilon_ij. Stress shear components are not doubled.
Consequently the Voigt product stress dot strain equals the tensor contraction.
For shell curvature vectors, the mixed curvature component is doubled in the same way.

The six-component shell law has a zero zz row and column. It is expressed in the
local shell frame; a frame transformation belongs to the element implementation.
`kappa` scales only the transverse shear stiffness, not membrane or bending shear.

Curvilinear routines receive the symmetric positive-definite inverse surface metric.
Strain components are covariant; stress/resultant components are contravariant.
Membrane, bending and transverse-shear stiffnesses include the thickness factors
t, t^3/12 and t respectively. Surface integration still needs the area Jacobian.

`compute_C_tensors_curv_2D` returns fourth-order membrane/bending tensors and a
second-order shear matrix. `tensor4_to_voigt2D` converts the former under the
engineering-shear convention, assuming the tensor minor symmetries.

The Cartesian membrane shear stiffness was previously multiplied by kappa.
Removing that factor changes membrane results when kappa differs from one;
the curvilinear shell laws and six-component shell matrix already used the
uncorrected membrane shear stiffness.

Checks are in `tests/materials/test_linear_elastic.py`: uniaxial and shear responses,
shell component ordering, skew-metric tensor/Voigt equivalence and preservation
of strain energy under a basis change.

The linear structural release does not include third-medium contact materials.
