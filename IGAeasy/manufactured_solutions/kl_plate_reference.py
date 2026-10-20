"""Navier and manufactured references for an affine unit-square KL plate."""
from functools import lru_cache
import numpy as np


class KLPlateReference:
    def __init__(self, kind='uniform', rotation=None, n_terms=64):
        if kind not in ('uniform', 'sine', 'twist'):
            raise ValueError('kind must be uniform, sine or twist')
        self.kind = kind
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        self.E, self.nu, self.thickness, self.load = 10920., .3, .1, -1.
        self.rigidity = self.E*self.thickness**3/(12*(1-self.nu**2))
        self.modes = np.arange(1, 2*n_terms, 2)*np.pi
        m, n = self.modes[:, None], self.modes[None, :]
        self.coefficients = 16*self.load/(m*n)/(self.rigidity*(m*m+n*n)**2)

    @lru_cache(maxsize=8192)
    def values(self, u, v):
        """Return deflection, parametric gradient and Hessian."""
        if self.kind == 'twist':
            return .01*u*v, .01*np.array([v, u]), np.array([[0., .01], [.01, 0.]])
        if self.kind == 'sine':
            factor = self.load/(4*self.rigidity*np.pi**4)
            su, sv, cu, cv = np.sin(np.pi*u), np.sin(np.pi*v), np.cos(np.pi*u), np.cos(np.pi*v)
            w = factor*su*sv
            gradient = factor*np.pi*np.array([cu*sv, su*cv])
            mixed = factor*np.pi**2*cu*cv
            return w, gradient, np.array([[-np.pi**2*w, mixed], [mixed, -np.pi**2*w]])
        su, sv = np.sin(self.modes*u), np.sin(self.modes*v)
        du, dv = self.modes*np.cos(self.modes*u), self.modes*np.cos(self.modes*v)
        ddu, ddv = -self.modes**2*su, -self.modes**2*sv
        C = self.coefficients
        w = su@C@sv
        gradient = np.array([du@C@sv, su@C@dv])
        mixed = du@C@dv
        return w, gradient, np.array([[ddu@C@sv, mixed], [mixed, su@C@ddv]])

    def field(self, x, y, z):
        u, v, _ = self.rotation.T@np.array([x, y, z])
        w, gradient, _ = self.values(u, v)
        return np.r_[self.rotation@[0., 0., w], self.rotation@np.r_[-gradient, 0.]]

    def displacement(self, x, y, z):
        return self.field(x, y, z)[:3]

    def body(self, u, v):
        load = (self.load if self.kind == 'uniform' else
                self.load*np.sin(np.pi*u)*np.sin(np.pi*v) if self.kind == 'sine' else 0.)
        return self.rotation@np.array([0., 0., load])

    def normal_moment(self, x, y, z, conormal):
        u, v, _ = self.rotation.T@np.array([x, y, z])
        hessian = self.values(u, v)[2]
        mu = (self.rotation.T@conormal)[:2]
        return -self.rigidity*((1-self.nu)*(mu@hessian@mu)+self.nu*np.trace(hessian))
