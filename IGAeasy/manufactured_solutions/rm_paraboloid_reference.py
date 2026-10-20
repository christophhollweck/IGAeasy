"""Independent RM5 equilibrium loads on an exact doubly curved paraboloid.

The default displacement is (0,0,A*sin(pi*u)*sin(pi*v)), with zero director.
The enriched case adds in-plane displacement and nonzero covariant director
coefficients Q*(1+u, .5+v). Forces and director loads follow from continuum stress
resultants, not the element B matrices. SymPy differentiates the analytic
resultants once when the reference is constructed.
"""
from functools import lru_cache
import numpy as np
import sympy as sp
from IGAeasy.manufactured_solutions.shell_symbolics import (
    paraboloid_geometry, stress_resultant, compile_moment, evaluate_moment, evaluate_traction, evaluate_normal)
from IGAeasy.manufactured_solutions.symbolic_equilibrium import equilibrium_function, boundary_force_function


@lru_cache(maxsize=8)
def _equilibrium(curvature, E, nu, thickness, amplitude, centered, director_amplitude, enriched, mixed):
    coordinates, tangents, metric, inverse, J, normal = paraboloid_geometry(curvature, centered, mixed)
    u, v = coordinates
    E, nu, t, amplitude = map(lambda x: sp.Rational(str(x)), (E, nu, thickness, amplitude))
    dual = [sum((inverse[i, j]*tangents[j] for j in range(2)), sp.zeros(3, 1)).applyfunc(sp.cancel)
            for i in range(2)]
    displacement = sp.Matrix([0, 0, amplitude*sp.sin(sp.pi*u)*sp.sin(sp.pi*v)])
    if enriched:
        displacement[:2, 0] = sp.Matrix([amplitude*u*v/10, amplitude*u*u/5])
    Q = sp.Rational(str(director_amplitude))
    director_coefficients = sp.Matrix([Q*(1+u), Q*(sp.Rational(1, 2)+v)])
    director = sum((dual[i]*director_coefficients[i] for i in range(2)), sp.zeros(3, 1))
    gradients = [displacement.diff(q) for q in coordinates]
    normal_gradients = [normal.diff(q) for q in coordinates]
    strain = sp.Matrix(2, 2, lambda i, j:
        (tangents[i].dot(gradients[j])+tangents[j].dot(gradients[i]))/2)
    bending = sp.Matrix(2, 2, lambda i, j:
        (normal_gradients[i].dot(gradients[j])+normal_gradients[j].dot(gradients[i])
         +tangents[i].dot(director.diff(coordinates[j]))
         +tangents[j].dot(director.diff(coordinates[i])))/2)

    N = stress_resultant(strain, inverse, E, nu, t).applyfunc(sp.factor_terms)
    M = stress_resultant(bending, inverse, E, nu, t**3/12).applyfunc(sp.factor_terms)
    shear = sp.Rational(5, 6)*E*t/(2*(1+nu))*inverse*sp.Matrix([
        normal.dot(gradients[i])+director_coefficients[i] for i in range(2)])
    # Integration by parts gives force per physical area, including metric derivatives.
    flux = [sum((N[i, j]*tangents[j]+M[i, j]*normal_gradients[j]
                 for j in range(2)), sp.zeros(3, 1))+shear[i]*normal for i in range(2)]
    moments = []
    for k in range(2):
        connection = sum(M[i, j]*dual[k].dot(tangents[i].diff(coordinates[j]))
                         for i in range(2) for j in range(2))
        moments.append(shear[k]-connection)
    director_flux = [[M[i, k] for k in range(2)] for i in range(2)]
    return (equilibrium_function(coordinates, J, flux, moments, director_flux),
            compile_moment(coordinates, tangents, M), boundary_force_function(coordinates, J, tangents, flux))


class ParaboloidRMReference:
    def __init__(self, curvature=(.2, .3), E=100., nu=.3, thickness=.1,
                 amplitude=.001, rotation=None, centered=False,
                 director_amplitude=0., enriched=False, mixed=(0., 0.)):
        self.curvature = tuple(curvature)
        self.E, self.nu, self.thickness, self.amplitude = E, nu, thickness, amplitude
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        self.centered = centered
        self.director_amplitude, self.enriched = director_amplitude, enriched
        self.mixed = tuple(mixed)
        self._load, self._moment, self._traction = _equilibrium(self.curvature, E, nu, thickness, amplitude, centered,
                                  director_amplitude, enriched, self.mixed)

    def displacement(self, x, y, z):
        u, v, _ = self.rotation.T@np.array([x, y, z])
        if self.centered:
            u, v = u+.5, v+.5
        value = np.array([0., 0., self.amplitude*np.sin(np.pi*u)*np.sin(np.pi*v)])
        if self.enriched:
            value[:2] = [self.amplitude*u*v/10, self.amplitude*u*u/5]
        return self.rotation@value

    def director(self, x, y, z):
        if self.director_amplitude == 0.:
            return np.zeros(3)
        u, v, _ = self.rotation.T@np.array([x, y, z])
        slope = self._slope(u+.5, v+.5) if self.centered else self._slope(u, v)
        if self.centered:
            u, v = u+.5, v+.5
        tangents = np.array([[1., 0., slope[0]], [0., 1., slope[1]]])
        coefficients = self.director_amplitude*np.array([1+u, .5+v])
        return self.rotation@(np.linalg.solve(tangents@tangents.T, coefficients)@tangents)

    def _slope(self, u, v):
        a, b = self.curvature
        c, d = self.mixed
        x, y, sign = (u-.5, v-.5, -1.) if self.centered else (u, v, 1.)
        return np.array([sign*2*a*x+2*c*x*y+d*y*y, sign*2*b*y+c*x*x+2*d*x*y])

    def field(self, x, y, z):
        return np.r_[self.displacement(x, y, z), self.director(x, y, z)]

    def moment(self, x, y, z, conormal):
        return evaluate_moment(self._moment, self.rotation, (x, y, z), conormal, self.centered)

    def traction(self, x, y, z, conormal):
        return evaluate_traction(self._traction, self.rotation, (x, y, z), conormal, self.centered)

    def normal(self, x, y, z):
        return evaluate_normal(self._moment, self.rotation, (x, y, z), self.centered)

    def body(self, u, v, formulation='RM5'):
        values = np.asarray(self._load(u, v), dtype=float)
        if formulation == 'RM5_v2':
            slope = self._slope(u, v)
            values[3:] = (np.eye(2)+np.outer(slope, slope))@values[3:]
        elif formulation != 'RM5':
            raise ValueError('This reference supports RM5 and RM5_v2')
        return np.r_[self.rotation@values[:3], values[3:]]
