"""Point loads are assembled once on shared edges and vertices."""
import unittest
import numpy as np
from shapely.geometry import box
from IGAeasy.geometry import get_geometry
from IGAeasy.splines.THBSplines.hierarchical_space import HierarchicalSpace
from IGAeasy.splines.LRSplines.lr_spline import LRSpline
from IGAeasy.trimming.Brep import BRep
from IGAeasy.trimming.trim_elements import trim
from IGAeasy.loads import PointLoad
from IGAeasy.assembler import VectorAssembler


def make_space(family, local=False, domain=None):
    geometry = get_geometry.surface(2,2,1,1,1,1,1,2,distortion=False,n_waves=0)
    space = (HierarchicalSpace(*geometry.knot_vectors,2,geometry.control_points) if family=='THB'
             else LRSpline(2,*geometry.knot_vectors,geometry.control_points))
    if local:
        if family=='THB': space.refine({0:[len(space.basis)//2]})
        else: space.refine_by_structured_mesh([space.basis[len(space.basis)//2]])
    if domain is not None: space.BReps=[BRep(1,polygon=domain,outerloop=True)]
    trim(space)
    return space


def assemble(space, point):
    load=PointLoad(point,[2.,-3.])
    for cell in space.elements:
        cell.bfnums=np.array(sorted(b.id for b in cell.supported_b_splines))
        cell.f_el=load.compute_contribution(cell,2)
    return VectorAssembler(2).assemble(space,'f_el')


class PointLoadTests(unittest.TestCase):
    def test_shared_nodes_and_edges_match_global_basis_evaluation(self):
        for family in ('THB','LR'):
            for local in (False,True):
                space=make_space(family,local)
                for point in ((.5,.5),(.5,.3),(0.,0.),(1.,1.),(.2,.3)):
                    with self.subTest(family=family,local=local,point=point):
                        force=assemble(space,point)
                        expected=np.concatenate([b(*point)*np.array([2.,-3.]) for b in space.basis])
                        np.testing.assert_allclose(force,expected,atol=1e-12)
                        np.testing.assert_allclose(force.reshape(-1,2).sum(axis=0),[2.,-3.],atol=1e-12)

    def test_hole_and_void_cells_receive_no_load(self):
        domain=box(.1,.1,.9,.9).difference(box(.35,.35,.65,.65))
        for family in ('THB','LR'):
            space=make_space(family,domain=domain)
            for point in ((.5,.5),(0.,0.),(1.,1.)):
                np.testing.assert_allclose(assemble(space,point),0.)
            for point in ((.5,.2),(.35,.5)):
                np.testing.assert_allclose(assemble(space,point).reshape(-1,2).sum(axis=0),[2.,-3.],atol=1e-12)


if __name__=='__main__':
    unittest.main(verbosity=2)
