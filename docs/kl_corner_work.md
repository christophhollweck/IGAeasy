# KL corner forces and effective boundary traction

## Convention and boundary identity

Let n be the surface normal, tau the positively oriented boundary tangent and
mu = tau cross n the outward conormal. The physical director variation satisfies

    theta_t(v) = delta_d(v).tau = -n.dv/ds.

The production operators retain boundary work in the form

    integral [t_raw.v + m_n*theta_n(v) + m_t*theta_t(v)] ds.

Define h = m_t*n. Integration by parts along a smooth edge gives

    integral m_t*theta_t(v) ds
      = integral dh/ds.v ds - [h.v]_start^end.

Consequently the equivalent condensed representation uses

    t_effective = t_raw + d(m_t*n)/ds,
    corner_force = h_outgoing - h_incoming.

At a smooth surface corner, n is continuous and the corner force is normal to
the surface. Reversing moment or boundary orientation conventions changes the
written signs; the work identity fixes the signs used here. Artificial segment
endpoints introduced by element intersections cancel when both adjacent pieces
are included. They are not new physical corners.

For a curved surface the effective traction contains both

    d(m_t)/ds*n + m_t*dn/ds.

The second term is tangential and must not be dropped. The uncondensed traction
already has a bending-curvature contribution. Retaining the full moment work
does not mean this underlying force term can be omitted.

[Benzaken et al., section 3, equations 3.1–3.2 and 3.7, Remark 12](https://arxiv.org/pdf/2007.01279)
discuss both the corner jump and the corrected in-plane bending contribution
to effective traction. Their result is more than a rule to add point forces.

## Mapping to this implementation

- `kl_clamped_weak.py`: `_point_operators` returns raw force and director flux.
  The trace stores t*delta_d and the director flux stores physical moment/t;
  the thickness factors cancel in boundary work. The complete tangential
  director work is retained. No condensed corner force should be added again.
- `kl_mixed_weak.py`: displacement and optional normal rotation are essential
  traces. The complete tangential work is lifted into their discrete trace space.
  In normal-only mode, the identity
  `theta_t = -d(n.v)/ds + (dn/ds).v` requires the existing complementary force
  correction. Its natural traction is therefore not the unmodified raw force.
- `kl_plate_weak.py`: the affine counterpart likewise retains tangential moment
  work and lifts it into the deflection trace rather than discarding its endpoints.
- `kl_symmetry_weak.py`: this is a restricted homogeneous reflection operator.
  It is not a general displacement/traction selector; the plane checks and zero
  complementary data are part of its applicability.

If effective traction is used instead of this retained moment work, the matching
endpoint/corner terms must be used too. An externally prescribed physical corner
load is a separate datum. On a free boundary it can be assembled as a point load;
this does not justify adding a duplicate jump term on an essential edge.
Corner-only displacement conditions and arbitrary mixed component masks do not
have a general automatically condensed Nitsche interface here.

## Targeted checks, 2026-10-16

The prior `test_twist_corner_reactions` already checked four affine reactions.
The new `tests/shells/test_kl_corner_work.py` independently extends the checks:

1. Use w = .01*u*v on affine and doubly curved polynomial graphs. Compare the
   retained tangential work vector with independently integrated effective-line
   work plus explicit endpoint forces. Test THB/LR, general spatial rotation,
   and a moderately trimmed curved rectangle.
2. Require nonzero physical corner forces, cancellation at artificial cell
   intersections, and a nonzero curved normal-variation contribution. Omitting
   corners or the m_t*dn/ds term would fail these checks. The derivative of h
   in the condensed comparison uses centered differences, independently of
   the production flux derivatives.
3. Check the raw tangential traction against analytical membrane and curvature
   contributions for the prescribed graph/field. This targets the in-plane
   bending force as well as the corner jump.
4. Check nonhomogeneous Nitsche residuals for clamped, mixed and affine-plate
   operators on the rotated trimmed twisting patch, without extra corner loads.

For the affine unit plate, E=10920, nu=.3 and t=.1 give D=1. The constant
twisting moment yields alternating corner forces of magnitude
2*D*(1-nu)*.01 = .014 and zero effective line force. The new test checks that
magnitude explicitly rather than comparing two implementations of a flux.

The two new tests and the two existing affine-corner/local-refinement tests pass.
No production kernel or boundary operator was changed: these cases revealed no
missing term. This is finite-case verification, not a proof of stability or
optimal convergence for every boundary configuration. It also does not settle
the hemisphere literature-reference discrepancy; see hemisphere_kl_diagnostic.md.
