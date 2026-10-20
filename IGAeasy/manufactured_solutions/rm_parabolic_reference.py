"""Axial shear on X(u,v)=(u,v,c*u**2): u0=(0,W*u**m,0), director=(0,Q*u**m,0).

RM5 uses membrane shear stiffness G*t. For RM6 the physical thickness
coordinate z has Jacobian sqrt(g)*(1-z*k), g=1+4*c**2*u**2,
k=2*c/g**1.5. Its axial shear energy per parametric area is
G/(2*sqrt(g))*integral((w'+z*d')**2/(1-z*k), dz)
plus kappa*G*t*sqrt(g)*d**2/2.
The reference uses the same thickness quadrature as the RM6 bulk element.
Loads follow by differentiating the two stress resultants, independently
of the finite element B matrices. A normal spin w'/(2*sqrt(g)) cancels
the artificial RM6 drilling strain without changing the physical director.
"""
import numpy as np


class ParabolicRMShearReference:
    def __init__(self, formulation='RM6', power=4, curvature=.2, amplitude=.01,
                 E=100., nu=.3, thickness=.1, rotation=None, gauss_t=None,
                 director_amplitude=.002, kappa=5/6):
        if formulation not in ('RM6', 'RM5', 'RM5_v2'):
            raise ValueError('Unknown RM formulation')
        if not isinstance(power, int) or power < 2:
            raise ValueError('power must be an integer at least two')
        self.formulation, self.power = formulation, power
        self.curvature, self.amplitude = curvature, amplitude
        self.E, self.nu, self.thickness = E, nu, thickness
        self.director_amplitude, self.kappa = director_amplitude, kappa
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        self.gauss_t = np.polynomial.legendre.leggauss(3) if gauss_t is None else gauss_t

    def displacement(self, x, y, z):
        u = (self.rotation.T@np.array([x, y, z]))[0]
        return self.rotation@np.array([0., self.amplitude*u**self.power, 0.])

    def field(self, x, y, z):
        """Return displacement plus rotation (RM6) or physical director (RM5)."""
        displacement = self.displacement(x, y, z)
        if self.formulation != 'RM6':
            return np.r_[displacement, self.director(x, y, z)]
        u = (self.rotation.T@np.array([x, y, z]))[0]
        q = self.director_amplitude*u**self.power
        slope = self.amplitude*self.power*u**(self.power-1)
        s = 2*self.curvature*u
        spin = slope/(2*(1+s*s))*np.array([-s, 0., 1.])
        rotation = self.rotation@(spin-q/np.sqrt(1+s*s)*np.array([1., 0., s]))
        return np.r_[displacement, rotation]

    def director(self, x, y, z):
        u = (self.rotation.T@np.array([x, y, z]))[0]
        return self.rotation@np.array([0., self.director_amplitude*u**self.power, 0.])

    def edge_data(self, x, y, z, conormal):
        """Independent raw force/director moment for the RM5 shear reference."""
        if self.formulation != 'RM5':
            raise ValueError('Physical edge data currently derived for RM5 only')
        u = (self.rotation.T@np.array([x, y, z]))[0]
        mu = self.rotation.T@conormal
        c, m, t = self.curvature, self.power, self.thickness
        g = 1+4*c*c*u*u
        G = self.E/(2*(1+self.nu))
        w_u = self.amplitude*m*u**(m-1)
        q, q_u = self.director_amplitude*u**m, self.director_amplitude*m*u**(m-1)
        N = G*t*w_u/g*np.array([[0., 1.], [1., 0.]])
        M = G*t**3*q_u/(12*g)*np.array([[0., 1.], [1., 0.]])
        shear = np.array([0., self.kappa*G*t*q])
        tangents = np.array([[1., 0., 2*c*u], [0., 1., 0.]])
        normal = np.array([-2*c*u, 0., 1.])/np.sqrt(g)
        normal_gradient = np.array([-2*c/g**1.5*tangents[0], np.zeros(3)])
        covariant_mu = tangents@mu
        force = tangents.T@N@covariant_mu+normal_gradient.T@M@covariant_mu+normal*(shear@covariant_mu)
        moment = tangents.T@M@covariant_mu
        return np.r_[self.rotation@force, self.rotation@moment]

    def energy_density(self, u, v):
        """Analytical RM5 energy per physical area for the axial shear field."""
        if self.formulation != 'RM5':
            raise ValueError('This surface energy is derived for RM5 only')
        g = 1+4*self.curvature**2*u*u
        m, t = self.power, self.thickness
        slope = self.amplitude*m*u**(m-1)
        q, q_u = self.director_amplitude*u**m, self.director_amplitude*m*u**(m-1)
        G = self.E/(2*(1+self.nu))
        return .5*G*t*((slope*slope+t*t/12*q_u*q_u)/g+self.kappa*q*q)

    def _thickness_moments(self, u):
        if self.formulation != 'RM6':
            return np.array([self.thickness, 0., self.thickness**3/12]), np.zeros(3)
        c = self.curvature
        g = 1+4*c*c*u*u
        curvature = 2*c/g**1.5
        curvature_derivative = -24*c**3*u/g**2.5
        z = self.thickness/2*np.asarray(self.gauss_t[0])
        weights = self.thickness/2*np.asarray(self.gauss_t[1])
        stretch = 1-z*curvature
        moments = np.array([np.sum(weights*z**k/stretch) for k in range(3)])
        derivatives = curvature_derivative*np.array([
            np.sum(weights*z**(k+1)/stretch**2) for k in range(3)])
        return moments, derivatives

    def resultants(self, u):
        """Return displacement/director flux densities per parametric length."""
        g = 1+4*self.curvature**2*u*u
        slope = self.amplitude*self.power*u**(self.power-1)
        director_slope = self.director_amplitude*self.power*u**(self.power-1)
        moments, _ = self._thickness_moments(u)
        return self.E/(2*(1+self.nu))/np.sqrt(g)*(
            moments[:2]*slope+moments[1:]*director_slope)

    def body(self, u, v):
        c, m, W = self.curvature, self.power, self.amplitude
        g = 1+4*c*c*u*u
        slope, second = W*m*u**(m-1), W*m*(m-1)*u**(m-2)
        q = self.director_amplitude*u**m
        q_slope = self.director_amplitude*m*u**(m-1)
        q_second = self.director_amplitude*m*(m-1)*u**(m-2)
        moments, derivatives = self._thickness_moments(u)
        flux = moments[:2]*slope+moments[1:]*q_slope
        flux_derivative = (moments[:2]*second+derivatives[:2]*slope
                           +moments[1:]*q_second+derivatives[1:]*q_slope)
        G = self.E/(2*(1+self.nu))
        loads = -G/g*(flux_derivative-flux*4*c*c*u/g)
        loads[1] += self.kappa*G*self.thickness*q
        force = self.rotation@np.array([0., loads[0], 0.])
        if self.formulation != 'RM6':
            return np.r_[force, 0., loads[1]]
        tangent = np.array([1., 0., 2*c*u])/np.sqrt(g)
        moment = self.rotation@(-loads[1]*tangent)
        return np.r_[force, moment]
