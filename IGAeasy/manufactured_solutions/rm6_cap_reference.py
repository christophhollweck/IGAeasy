"""RM6 continuum equilibrium on a regular quadratic graph, without element data."""
import numpy as np


class RM6CapReference:
    """u=(0,0,A*h**2), rotation=0 on X=(u,v,h), h=a*u**2+b*v**2.

    The displacement gradient is parallel to the surface gradient of h, so its
    normal spin and the continuum drilling strain vanish. Body rotations still
    require nonzero distributed moments. Loads follow from 3D plane-stress work
    integrated through thickness; complex-step differentiation takes its surface
    divergence. No finite element B matrix or assembled stiffness is used.
    """
    def __init__(self, curvature=(.4, -.2), E=100., nu=.3, thickness=.1,
                 amplitude=.01, rotation=None):
        self.curvature = curvature
        self.E, self.nu, self.thickness, self.amplitude = E, nu, thickness, amplitude
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        self.nodes, self.weights = np.polynomial.legendre.leggauss(3)

    def field(self, x, y, z):
        local = self.rotation.T@np.array([x, y, z])
        return np.r_[self.rotation@[0., 0., self.amplitude*local[2]**2], np.zeros(3)]

    def director(self, x, y, z):
        return np.zeros(3)

    def fluxes(self, u, v):
        a, b = self.curvature
        h, hu, hv = a*u*u+b*v*v, 2*a*u, 2*b*v
        tangents = np.array([[1., 0., hu], [0., 1., hv]])
        cross = np.array([-hu, -hv, 1.])
        J = np.sqrt(cross@cross)
        normal = cross/J
        cross_gradient = np.array([[-2*a, 0., 0.], [0., -2*b, 0.]])
        normal_gradient = (cross_gradient-(cross_gradient@normal)[:, None]*normal)/J
        displacement_gradient = np.array([[0., 0., 2*self.amplitude*h*hu],
                                          [0., 0., 2*self.amplitude*h*hv], [0., 0., 0.]])
        dtype = np.result_type(u, v, float)
        force, rotation_flux = np.zeros((2, 3), dtype=dtype), np.zeros((2, 3), dtype=dtype)
        rotation_value = np.zeros(3, dtype=dtype)
        G = self.E/(2*(1+self.nu))
        projector = np.eye(3)-np.outer(normal, normal)
        for node, weight in zip(self.nodes, self.weights):
            z = self.thickness*node/2
            jacobian = np.vstack((tangents+z*normal_gradient, normal))
            inverse = np.linalg.inv(jacobian)
            gradient = displacement_gradient.T@inverse.T
            strain = (gradient+gradient.T)/2
            membrane = projector@strain@projector
            shear = projector@strain@normal
            stress = (2*G*membrane+2*G*self.nu/(1-self.nu)*np.trace(membrane)*projector
                      +2*(5/6)*G*(np.outer(shear, normal)+np.outer(normal, shear)))
            measure = weight*self.thickness/2*np.linalg.det(jacobian)/J
            traction = (stress@inverse).T
            force += measure*traction[:2]
            rotation_flux += measure*z*np.cross(normal, traction[:2])
            rotation_value += measure*(z*np.cross(normal_gradient, traction[:2]).sum(axis=0)
                                       +np.cross(normal, traction[2]))
        return J, force, rotation_flux, rotation_value

    def body(self, u, v):
        J, _, _, value = self.fluxes(u, v)
        step = 1e-30
        force_divergence, moment_divergence = np.zeros(3), np.zeros(3)
        for axis, point in enumerate(((u+step*1j, v), (u, v+step*1j))):
            area, force, moment, _ = self.fluxes(*point)
            force_divergence += np.imag(area*force[axis])/step
            moment_divergence += np.imag(area*moment[axis])/step
        return np.r_[self.rotation@(-force_divergence/J), self.rotation@(value-moment_divergence/J)]
