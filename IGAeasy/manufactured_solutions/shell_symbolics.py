"""Continuum geometry and isotropic resultants for symbolic shell references."""
import sympy as sp
import numpy as np


def compile_moment(coordinates, tangents, resultant):
    """Compile physical director moment data independently of element operators."""
    normal = tangents[0].cross(tangents[1])
    normal = normal/sp.sqrt(normal.dot(normal))
    return sp.lambdify(coordinates, [*tangents[0], *tangents[1], *resultant,
                                    *normal.diff(coordinates[0]), *normal.diff(coordinates[1])],
                       'numpy', cse=True, docstring_limit=0)


def evaluate_moment(function, rotation, position, conormal, centered=True):
    local = rotation.T@np.asarray(position)
    u, v = local[:2]+(.5 if centered else 0.)
    values = np.asarray(function(u, v), dtype=float)
    tangents, resultant = values[:6].reshape(2, 3), values[6:10].reshape(2, 2)
    covariant_conormal = tangents@(rotation.T@conormal)
    return rotation@(tangents.T@resultant.T@covariant_conormal)


def evaluate_traction(function, rotation, position, conormal, centered=True):
    local = rotation.T@np.asarray(position)
    u, v = local[:2]+(.5 if centered else 0.)
    values = np.asarray(function(u, v), dtype=float)
    tangents, force = values[:6].reshape(2, 3), values[6:].reshape(2, 3)
    return rotation@(force.T@(tangents@(rotation.T@conormal)))


def evaluate_normal(function, rotation, position, centered=True):
    local = rotation.T@np.asarray(position)
    u, v = local[:2]+(.5 if centered else 0.)
    values = np.asarray(function(u, v), dtype=float)
    normal = np.cross(values[:3], values[3:6])
    return rotation@(normal/np.linalg.norm(normal))


def paraboloid_geometry(curvature, centered, mixed=(0., 0.)):
    u, v = sp.symbols('u v', real=True)
    a, b = (sp.Rational(str(value)) for value in curvature)
    c, d = (sp.Rational(str(value)) for value in mixed)
    x, y, sign = u, v, 1
    if centered:
        x, y = u-sp.Rational(1, 2), v-sp.Rational(1, 2)
        sign = -1
    X = sp.Matrix([x, y, sign*(a*x*x+b*y*y)+c*x*x*y+d*x*y*y])
    tangents = [X.diff(q) for q in (u, v)]
    slope = sp.Matrix([tangents[0][2], tangents[1][2]])
    determinant = sp.expand(1+slope.dot(slope))
    metric = sp.eye(2)+slope*slope.T
    # Use one canonical denominator throughout the equilibrium derivatives.
    inverse = (sp.Matrix([[1+slope[1]**2, -slope[0]*slope[1]],
                          [-slope[0]*slope[1], 1+slope[0]**2]])/determinant).applyfunc(sp.cancel)
    J = sp.sqrt(determinant)
    normal = sp.Matrix([-slope[0], -slope[1], 1])/J
    return (u, v), tangents, metric, inverse, J, normal


def stress_resultant(strain, inverse, E, nu, scale):
    """Contravariant plane-stress resultant for a covariant symmetric strain."""
    trace = sum(inverse[i, j]*strain[i, j] for i in range(2) for j in range(2))
    return scale*E/(1+nu)*(inverse*strain*inverse+nu/(1-nu)*trace*inverse)
