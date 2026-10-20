"""Independent KL and hierarchical RM equilibrium on a regular paraboloid.

Displacement variations enter membrane strain and the covariant Hessian.
The Euler-Lagrange force retains all three Cartesian variations. Hierarchical
director coefficients add their own bending and shear work. No element or
boundary B matrices are used to derive the loads.
"""
from functools import lru_cache
import numpy as np
import sympy as sp
from IGAeasy.manufactured_solutions.shell_symbolics import (
    paraboloid_geometry, stress_resultant, compile_moment, evaluate_moment, evaluate_traction, evaluate_normal)
from IGAeasy.manufactured_solutions.symbolic_equilibrium import equilibrium_function, boundary_force_function


@lru_cache(maxsize=8)
def _reference(curvature, E, nu, thickness, amplitude, eta, mixed):
    coordinates, tangents, metric, inverse, J, normal = paraboloid_geometry(curvature, True, mixed)
    u, v = coordinates
    E, nu, t, A = (sp.Rational(str(value)) for value in (E, nu, thickness, amplitude))
    eta = sp.Matrix([sp.Rational(str(value)) for value in eta])
    dual = [sum((inverse[i, j]*tangents[j] for j in range(2)), sp.zeros(3, 1)).applyfunc(sp.cancel)
            for i in range(2)]
    connection = [[[sp.cancel(dual[k].dot(tangents[i].diff(coordinates[j]))) for k in range(2)]
                   for j in range(2)] for i in range(2)]

    # Nonzero boundary values exercise the prescribed-field consistency terms.
    displacement = sp.Matrix([A*u*v/10, A*u*u/5, A*sp.sin(sp.pi*u)*sp.sin(sp.pi*v)])
    gradients = [displacement.diff(q) for q in coordinates]
    director_kl = -sum((dual[i]*normal.dot(gradients[i]) for i in range(2)), sp.zeros(3, 1))
    extra_director = sum((tangents[i]*eta[i] for i in range(2)), sp.zeros(3, 1))
    strain = sp.Matrix(2, 2, lambda i, j:
        (tangents[i].dot(gradients[j])+tangents[j].dot(gradients[i]))/2)
    bending = sp.Matrix(2, 2, lambda i, j:
        -normal.dot(displacement.diff(coordinates[i], coordinates[j])
                    -sum((connection[i][j][k]*gradients[k] for k in range(2)), sp.zeros(3, 1)))
        +(tangents[j].dot(extra_director.diff(coordinates[i]))
          +tangents[i].dot(extra_director.diff(coordinates[j])))/2)
    N = stress_resultant(strain, inverse, E, nu, t)
    M = stress_resultant(bending, inverse, E, nu, t**3/12)
    N = N.applyfunc(sp.factor_terms)
    M = M.applyfunc(sp.factor_terms)

    # First- and second-gradient work: f = (-div(J*P) + div(div(J*H)))/J.
    P = [sum((N[i, j]*tangents[j] for j in range(2)), sp.zeros(3, 1))
         +normal*sum(M[a, b]*connection[a][b][i] for a in range(2) for b in range(2))
         for i in range(2)]
    director_load = []
    shear_load = sp.Rational(5, 6)*E*t/(2*(1+nu))*metric*eta
    for k in range(2):
        value = sum(M[i, j]*tangents[j].dot(tangents[k].diff(coordinates[i]))
                    for i in range(2) for j in range(2))+shear_load[k]
        director_load.append(value)
    director_flux = [[sum(M[i, j]*metric[j, k] for j in range(2)) for k in range(2)] for i in range(2)]
    second_flux = [[-M[i, j]*normal for j in range(2)] for i in range(2)]
    body = equilibrium_function(coordinates, J, P, director_load, director_flux, second_flux)
    field = sp.lambdify(coordinates, [*displacement, *(director_kl+extra_director)],
                       'numpy', cse=True, docstring_limit=0)
    return (body, field, compile_moment(coordinates, tangents, M),
            boundary_force_function(coordinates, J, tangents, P, second_flux))


class GradientParaboloidReference:
    def __init__(self, formulation='KL', curvature=(.2, .3), E=100., nu=.3,
                 thickness=.1, amplitude=.001, rotation=None, eta=(.0003, -.0002), mixed=(0., 0.)):
        if formulation not in ('KL', 'RM5_hier'):
            raise ValueError('Use KL or RM5_hier')
        self.formulation = formulation
        self.eta = tuple(eta) if formulation == 'RM5_hier' else (0., 0.)
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        self._body, self._field, self._moment, self._traction = _reference(tuple(curvature), E, nu, thickness, amplitude,
                                            self.eta, tuple(mixed))

    def field(self, x, y, z):
        u, v, _ = self.rotation.T@np.array([x, y, z])
        value = np.asarray(self._field(u+.5, v+.5), dtype=float).reshape(2, 3)
        return (value@self.rotation.T).ravel()

    def displacement(self, x, y, z):
        return self.field(x, y, z)[:3]

    def director(self, x, y, z):
        return self.field(x, y, z)[3:]

    def moment(self, x, y, z, conormal):
        return evaluate_moment(self._moment, self.rotation, (x, y, z), conormal)

    def normal(self, x, y, z):
        return evaluate_normal(self._moment, self.rotation, (x, y, z))

    def traction(self, x, y, z, conormal, normal_only=False):
        force = evaluate_traction(self._traction, self.rotation, (x, y, z), conormal)
        if normal_only:
            u, v, _ = self.rotation.T@np.array([x, y, z])
            values = np.asarray(self._moment(u+.5, v+.5), dtype=float)
            tangents = values[:6].reshape(2, 3)
            tangent = np.cross(self.normal(x, y, z), conormal)
            direction = np.linalg.solve(tangents@tangents.T, tangents@(self.rotation.T@tangent))
            normal_s = self.rotation@(direction@values[10:].reshape(2, 3))
            force += normal_s*(tangent@self.moment(x, y, z, conormal))
        return force

    def body(self, u, v):
        value = np.asarray(self._body(u, v), dtype=float)
        force = self.rotation@value[:3]
        return force if self.formulation == 'KL' else np.r_[force, value[3:]]
