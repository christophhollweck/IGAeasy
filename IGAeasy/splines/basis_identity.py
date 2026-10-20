"""Exact identity of polynomial tensor-product basis functions.

Numerical closeness is deliberately separate from dictionary/set identity.
LR scaling weights, control points and solution fields are not part of identity.
"""
def basis_key(degree, knots_u, knots_v):
    return (int(degree), tuple(float(k) for k in knots_u), tuple(float(k) for k in knots_v))
