import sympy as sp
import numpy as np

class ManufacturedSolutionPoisson:
    def __init__(self, dim, u_expr, kappa=1.0):
        self.dim = dim
        self.kappa = kappa
        self.x = sp.symbols(['x', 'y', 'z'][:dim])
        self.u = u_expr(*self.x)
        self.grad_u = [sp.diff(self.u, xi) for xi in self.x]
        self.laplace_u = sum(sp.diff(du, xi) for du, xi in zip(self.grad_u, self.x))

    def rhs_function(self):
        """Returns a sympy expression for the source term f = -div(kappa * grad(u))"""
        return -self.kappa * self.laplace_u

    def neumann_function(self):
        """Returns a callable for Neumann BC: q = kappa * grad(u) · n"""
        return lambda normal: self.kappa * sum(du * ni for du, ni in zip(self.grad_u, normal))

    def u_function(self):
        """Returns a callable for the analytical solution"""
        return self.u

    def to_lambda(self):
        """Returns callables: u(x), f(x), grad_u(x)"""
        u_func = sp.lambdify(self.x, self.u, 'numpy')
        f_func = sp.lambdify(self.x, self.rhs_function(), 'numpy')
        grad_funcs = [sp.lambdify(self.x, gi, 'numpy') for gi in self.grad_u]

        def grad_u_func(*coords):
            return np.array([f(*coords) for f in grad_funcs])

        return u_func, f_func, grad_u_func


