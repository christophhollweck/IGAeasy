"""Physical force/director-moment work on KL and RM5 material edges."""
import numpy as np
from IGAeasy.constraints.line_helpers import get_boundary_quadrature
from IGAeasy.splines.batch_evaluation import evaluate_derivatives_batch


class ShellEdgeLoad:
    """Integrate raw traction and moment conjugate to physical director change.

    data(x,y,z,conormal) returns six Cartesian components (force, moment) per
    physical edge length. The moment is not a torque conjugate to RM6 rotation.
    KL retains tangential moment work, including its implicit endpoint forces;
    do not supply condensed effective traction or duplicate corner jumps too.
    Lines follow the material-left orientation, including inner boundaries.
    """
    def __init__(self, line, data, formulation='KL', quadrature_order=None):
        if formulation not in ('KL', 'RM5'):
            raise ValueError('ShellEdgeLoad currently supports KL and RM5')
        self.line, self.data = line, data
        self.formulation, self.quadrature_order = formulation, quadrature_order

    def integrate_contribution(self, cell):
        basis = sorted(cell.supported_b_splines, key=lambda b: b.id)
        dof = 3 if self.formulation == 'KL' else 5
        force = np.zeros(dof*len(basis))
        order = 2*cell.p+1 if self.quadrature_order is None else self.quadrature_order
        points = get_boundary_quadrature(self.line, cell, quadrature_order=order)
        if not points:
            return force
        values = evaluate_derivatives_batch(basis, [(p.x, p.y) for p, _, _ in points],
                                            cell, max_order=1)
        cps = np.asarray([b.cp for b in basis])
        for q, (_, weight, tangent) in enumerate(points):
            N = values[0, 0][q]
            gradient = np.column_stack((values[1, 0][q], values[0, 1][q]))
            tangents = gradient.T@cps
            normal = np.cross(*tangents)
            normal /= np.linalg.norm(normal)
            contra = np.linalg.solve(tangents@tangents.T, tangents)
            length = np.linalg.norm(tangent@tangents)
            conormal = np.cross(tangent@tangents/length, normal)
            load = np.asarray(self.data(*(N@cps), conormal), dtype=float)
            if load.shape != (6,) or not np.isfinite(load).all():
                raise ValueError('Shell edge data must return six finite components')
            if abs(normal@load[3:]) > 1e-12+1e-10*np.linalg.norm(load[3:]):
                raise ValueError('Physical director moment must be tangential')
            local = np.zeros((len(basis), dof))
            local[:, :3] = N[:, None]*load[:3]
            if self.formulation == 'KL':
                local[:, :3] -= ((gradient@contra)@load[3:])[:, None]*normal
            else:
                local[:, 3:] = N[:, None]*(contra@load[3:])
            force += weight*length*local.ravel()
        return force
