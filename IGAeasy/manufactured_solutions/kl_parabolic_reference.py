"""Manufactured KL solution on X(u,v)=(u,v,c*u**2), with physical body load.

The load is obtained from the one-dimensional energy density
J*(A*e**2/g**2 + D*k**2/g**2)/2, where g=1+4*c**2*u**2,
e=2*c*u*w', k=-(w''-4*c**2*u*w'/g)/sqrt(g). Variations in all
three displacement directions are retained. Expressions were symbolically
differentiated offline; running a study needs no symbolic algebra dependency.
"""
import numpy as np
from itertools import combinations
from math import comb


def monomial_coefficient(inner_knots, power):
    """Polar-form coefficient of u**power in a degree-len(inner_knots) basis."""
    return sum(np.prod(values) for values in combinations(inner_knots, power))/comb(len(inner_knots), power)


class ParabolicKLReference:
    def __init__(self, power=4, curvature=.2, amplitude=.01, E=10920., nu=.3, thickness=.1, rotation=None):
        if power not in (3, 4):
            raise ValueError('Reference supports cubic or quartic displacement')
        self.power, self.curvature, self.amplitude = power, curvature, amplitude
        self.E, self.nu, self.thickness = E, nu, thickness
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)

    def coefficient(self, knots_u, knots_v):
        return self.amplitude*monomial_coefficient(knots_u, self.power)

    def field(self, x, y, z):
        u = (self.rotation.T@np.array([x, y, z]))[0]
        c, W = self.curvature, self.amplitude
        g = 1+4*c*c*u*u
        w, slope = W*u**self.power, self.power*W*u**(self.power-1)
        director = -slope/g**1.5*np.array([1., 0., 2*c*u])
        return np.r_[self.rotation@[0., 0., w], self.rotation@director]

    def body(self, u, v):
        c, W = self.curvature, self.amplitude
        A = self.E*self.thickness/(1-self.nu**2)
        D = self.E*self.thickness**3/(12*(1-self.nu**2))
        g = 1+4*c*c*u*u
        if self.power == 3:
            fx = -6*W*c*(192*A*c**6*u**8+144*A*c**4*u**6+36*A*c**2*u**4+3*A*u**2
                +512*D*c**6*u**6+224*D*c**4*u**4-176*D*c**2*u**2+4*D)/g**6
            fz = -48*W*c*c*u*(64*A*c**8*u**10+112*A*c**6*u**8+60*A*c**4*u**6
                +13*A*c*c*u**4+A*u*u-60*D*c**4*u**4-37*D*c*c*u*u+8*D)/g**6
        else:
            fx = -16*W*c*u*(128*A*c**8*u**10+224*A*c**6*u**8+120*A*c**4*u**6
                +26*A*c*c*u**4+2*A*u*u+192*D*c**6*u**6-136*D*c**4*u**4-118*D*c*c*u*u+9*D)/g**6
            fz = -8*W*(1024*A*c**10*u**12+1408*A*c**8*u**10+672*A*c**6*u**8
                +136*A*c**4*u**6+10*A*c*c*u**4-512*D*c**6*u**6-32*D*c**4*u**4+120*D*c*c*u*u-3*D)/g**6
        return self.rotation@np.array([fx, 0., fz])

    def normal_moment(self, x, y, z, conormal):
        """Independent physical normal moment from the analytic curvature field."""
        u = (self.rotation.T@np.array([x, y, z]))[0]
        mu = self.rotation.T@conormal
        c, W, m = self.curvature, self.amplitude, self.power
        g = 1+4*c*c*u*u
        slope = m*W*u**(m-1)
        second = m*(m-1)*W*u**(m-2)
        bending = -(second-4*c*c*u*slope/g)/np.sqrt(g)
        D = self.E*self.thickness**3/(12*(1-self.nu**2))
        # Orthogonal physical components in a_1/sqrt(g), a_2.
        mu_1 = mu@np.array([1., 0., 2*c*u])/np.sqrt(g)
        return D*bending/g*(mu_1**2+self.nu*mu[1]**2)

    def transverse_traction(self, x, y, z, conormal):
        """Membrane traction along the straight generator a_2, for this field."""
        u = (self.rotation.T@np.array([x,y,z]))[0]
        mu = self.rotation.T@conormal
        slope = self.power*self.amplitude*u**(self.power-1)
        g = 1+4*self.curvature**2*u*u
        A = self.E*self.thickness/(1-self.nu**2)
        return self.rotation@np.array([0.,self.nu*A*2*self.curvature*u*slope/g*mu[1],0.])

    def edge_data(self, x, y, z, conormal):
        """Independent raw force and director moment, before edge condensation."""
        u = (self.rotation.T@np.array([x, y, z]))[0]
        mu = self.rotation.T@conormal
        c, m, W = self.curvature, self.power, self.amplitude
        g = 1+4*c*c*u*u
        connection = 4*c*c*u/g
        connection_u = 4*c*c/g-32*c**4*u*u/g**2
        slope, second = m*W*u**(m-1), m*(m-1)*W*u**(m-2)
        third = m*(m-1)*(m-2)*W*u**(m-3)
        curvature = -(second-connection*slope)/np.sqrt(g)
        curvature_u = -(third-connection_u*slope-connection*second)/np.sqrt(g)-connection*curvature
        A = self.E*self.thickness/(1-self.nu**2)
        D = A*self.thickness**2/12
        N = np.diag([A*2*c*u*slope/g**2, self.nu*A*2*c*u*slope/g])
        M = np.diag([D*curvature/g**2, self.nu*D*curvature/g])
        M11_u = D*(curvature_u-4*connection*curvature)/g**2
        tangents = np.array([[1., 0., 2*c*u], [0., 1., 0.]])
        normal = np.array([-2*c*u, 0., 1.])/np.sqrt(g)
        normal_u = -2*c/g**1.5*tangents[0]
        covariant_mu = tangents@mu
        force = (tangents.T@N@covariant_mu+normal_u*M[0, 0]*covariant_mu[0]
                 +normal*(M11_u+2*connection*M[0, 0])*covariant_mu[0])
        moment = tangents.T@M@covariant_mu
        return np.r_[self.rotation@force, self.rotation@moment]

    def energy_density(self, u, v):
        """Analytical membrane plus bending energy per physical surface area."""
        c, m, W = self.curvature, self.power, self.amplitude
        g = 1+4*c*c*u*u
        slope, second = m*W*u**(m-1), m*(m-1)*W*u**(m-2)
        curvature = -(second-4*c*c*u*slope/g)/np.sqrt(g)
        A = self.E*self.thickness/(1-self.nu**2)
        return .5*A/g**2*((2*c*u*slope)**2+self.thickness**2/12*curvature**2)


class ParabolicRM5HierReference(ParabolicKLReference):
    """Same graph with constant nonzero hierarchical eta^1 and its five loads.

    The director increment adds a_1*eta^1; body[3] is the generalized force
    conjugate to eta^1 per physical area, not a Cartesian moment component.
    """
    def __init__(self, eta=-.001, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta

    def edge_data(self, *args):
        raise NotImplementedError('The KL raw edge reference does not include hierarchical director work')

    def energy_density(self, *args):
        raise NotImplementedError('The KL energy reference does not include hierarchical director work')

    def field(self, x, y, z):
        value = super().field(x, y, z)
        u = (self.rotation.T@np.array([x, y, z]))[0]
        value[3:] += self.rotation@np.array([self.eta, 0., 2*self.curvature*u*self.eta])
        return value

    def moment_vector(self, x, y, z, conormal):
        """Physical edge moment from the analytic hierarchical bending strain."""
        u = (self.rotation.T@np.array([x,y,z]))[0]
        mu = self.rotation.T@conormal
        c, m, W = self.curvature, self.power, self.amplitude
        g = 1+4*c*c*u*u
        slope, second = m*W*u**(m-1), m*(m-1)*W*u**(m-2)
        bending = -(second-4*c*c*u*slope/g)/np.sqrt(g)+4*c*c*u*self.eta
        D = self.E*self.thickness**3/(12*(1-self.nu**2))
        a1 = np.array([1.,0.,2*c*u])
        return self.rotation@(D*bending/g**2*a1*(a1@mu)
                              +self.nu*D*bending/g*np.array([0.,mu[1],0.]))

    def body(self, u, v):
        c, W, eta = self.curvature, self.amplitude, self.eta
        g = 1+4*c*c*u*u
        D = self.E*self.thickness**3/(12*(1-self.nu**2))
        G = (5/6)*self.E*self.thickness/(2*(1+self.nu))
        extra_x = 16*D*c**3*eta*(24*c**4*u**4-26*c*c*u*u+1)/g**4.5
        extra_z = -32*D*c**4*eta*u*(16*c*c*u*u-5)/g**4.5
        force = super().body(u, v)+self.rotation@np.array([extra_x, 0., extra_z])
        slope = self.power*W*u**(self.power-1)
        second = self.power*(self.power-1)*W*u**(self.power-2)
        third = self.power*(self.power-1)*(self.power-2)*W*u**(self.power-3)
        connection = 4*c*c*u/g
        connection_derivative = 4*c*c/g-32*c**4*u*u/g**2
        curvature = -(second-connection*slope)/np.sqrt(g)
        curvature_derivative = -(third-connection_derivative*slope-connection*second)/np.sqrt(g)-curvature*connection
        curvature += 4*c*c*u*eta
        curvature_derivative += 4*c*c*eta
        moment = D*curvature/g**2
        moment_derivative = D*(curvature_derivative-4*connection*curvature)/g**2
        eta_force = -8*c*c*u*moment-g*moment_derivative+G*g*eta
        return np.r_[force, eta_force, 0.]


class PolynomialClampedPlateReference(ParabolicRM5HierReference):
    """w=W*f(u)*f(v), eta=0; displacement/director vanish on unit-square edges.

    f(s)=s**2*(1-s)**2. Distributed generalized director loads are necessary
    to realize the zero-shear solution in RM5_hier; KL uses only the first
    three Cartesian force components. This is a manufactured load case.
    """
    def __init__(self, **kwargs):
        if kwargs.get('curvature', 0.) != 0.:
            raise ValueError('Clamped plate reference requires curvature=0')
        kwargs['curvature'] = 0.
        super().__init__(eta=0., **kwargs)
        if self.power != 4:
            raise ValueError('Clamped plate reference is biquartic')

    @staticmethod
    def polynomial(s):
        return np.array([s*s*(1-s)**2, 2*s-6*s*s+4*s**3,
                         2-12*s+12*s*s, -12+24*s, 24.])

    def coefficient(self, knots_u, knots_v):
        values = [monomial_coefficient(knots, 2)-2*monomial_coefficient(knots, 3)
                  + monomial_coefficient(knots, 4) for knots in (knots_u, knots_v)]
        return self.amplitude*np.prod(values)

    def field(self, x, y, z):
        u, v, _ = self.rotation.T@np.array([x, y, z])
        a, b = self.polynomial(u), self.polynomial(v)
        return np.r_[self.rotation@[0., 0., self.amplitude*a[0]*b[0]],
            self.rotation@[-self.amplitude*a[1]*b[0], -self.amplitude*a[0]*b[1], 0.]]

    def body(self, u, v):
        a, b = self.polynomial(u), self.polynomial(v)
        D = self.E*self.thickness**3/(12*(1-self.nu**2))
        factor = D*self.amplitude
        force = self.rotation@np.array([0., 0., factor*(a[4]*b[0]+2*a[2]*b[2]+a[0]*b[4])])
        return np.r_[force, factor*(a[3]*b[0]+a[1]*b[2]), factor*(a[0]*b[3]+a[2]*b[1])]
