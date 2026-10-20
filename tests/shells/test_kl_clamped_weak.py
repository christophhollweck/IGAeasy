"""Manufactured bending fields check the full-gradient KL Green identity."""
import unittest
import numpy as np
from shapely.geometry import box, LineString
from shapely.geometry.polygon import orient
from IGAeasy.trimming.Brep import BRep
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.elements.KLshell3p import KLShell
from IGAeasy.material.linear_elastic import LinearElasticMaterial
from IGAeasy.numerical_integration.map_gp_to_element import assign_gp_and_properties
from IGAeasy.constraints.kl_clamped_weak import WeakKLAffineClampedBoundary, WeakKLClampedBoundary
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.loads import SurfaceLoad
from IGAeasy.assembler import MatrixAssembler, VectorAssembler


class KLClampedTests(unittest.TestCase):
    def test_rejects_interior_c0_line_away_from_boundary(self):
        ku = [0.]*4+[.125, .25, .375]+[.5]*3+[.625, .75, .875]+[1.]*4
        kv = [0.]*4+[.5]+[1.]*4
        gu = [np.mean(ku[i+1:i+4]) for i in range(len(ku)-4)]
        gv = [np.mean(kv[i+1:i+4]) for i in range(len(kv)-4)]
        rs = HierarchicalSpace(ku, kv, 3, [[u, v, 0.] for v in gv for u in gu])
        boundary = WeakKLClampedBoundary(LineString([(0., 1.), (0., 0.)]), lambda x, y, z: np.zeros(6))
        cell = next(cell for cell in rs.elements if cell.u_min == 0.)
        with self.assertRaisesRegex(ValueError, 'C1'):
            boundary.integrate_contribution(cell)

    def test_polynomial_green_identity(self):
        for degree, trimmed in ((3, False), (2, False), (2, True)):
            with self.subTest(degree=degree, trimmed=trimmed):
                geo = get_geometry.surface(2, 2, 1, 1, 1, 1, 1, degree, distortion=0, n_waves=0)
                rs = HierarchicalSpace(*geo.knot_vectors, degree, geo.control_points)
                angle = .6
                R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
                exact = np.zeros(3*len(rs.basis))
                for b in rs.basis:
                    ku, kv = b.knots_u[1:-1], b.knots_v[1:-1]
                    if degree == 3:
                        value = .01*np.prod(ku)
                    else:
                        value = .01*np.prod(ku)*np.prod(kv)
                    exact[3*b.id:3*b.id+3] = R@[0., 0., value]
                    b.cp = R@np.asarray(b.cp)
                domain = orient(box(.13, .07, .91, .87) if trimmed else box(0., 0., 1., 1.), sign=1.)
                if trimmed:
                    rs.BReps = [BRep(1, polygon=domain, outerloop=True)]
                    trim(rs)
                assign_gp_and_properties(rs, material_model=LinearElasticMaterial(10920., .3, 1.), properties={'t': .1, 'rho': 1.}, full_quad=True)
                def field(x, y, z):
                    u, v, _ = R.T@[x, y, z]
                    if degree == 3:
                        w, dx, dy = .01*u**3, -.03*u**2, 0.
                    else:
                        w, dx, dy = .01*u*u*v*v, -.02*u*v*v, -.02*u*u*v
                    return np.r_[R@[0., 0., w], R@[dx, dy, 0.]]
                boundary = WeakKLAffineClampedBoundary(LineString(domain.exterior.coords), field)
                rigidity = 10920.*.1**3/(12*(1-.3**2))
                q = R@[0., 0., 0. if degree == 3 else .08*rigidity]
                load = SurfaceLoad(3, [lambda x, y, value=value: value for value in q])
                for cell in rs.elements:
                    KLShell().element_routine(cell)
                    force = load.integrate_contribution(cell, 3)
                    kb, fb, gamma = boundary.integrate_contribution(cell)
                    cell.K_el += kb
                    cell.f_el += force+fb
                K = MatrixAssembler(dof_per_cp=3).assemble(rs, 'K_el', use_sparse=True)
                f = VectorAssembler(dof_per_cp=3).assemble(rs, 'f_el')
                np.testing.assert_allclose(K@exact, f, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(K.toarray(), K.toarray().T, atol=1e-10)
                active = np.flatnonzero(K.diagonal() > 0.)
                reduced = K.toarray()[np.ix_(active, active)]
                scale = np.sqrt(np.diag(reduced))
                normalized = reduced/np.outer(scale, scale)
                self.assertGreater(np.linalg.eigvalsh(normalized).min(), 0.)
                solved = np.linalg.solve(normalized, f[active]/scale)/scale
                np.testing.assert_allclose(solved, exact[active], atol=1e-10, rtol=1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
