"""Check RM5 strain operators against independently differentiated directors."""
import unittest
import numpy as np
from IGAeasy.elements import RMshell5p, RMshell5p_v2


def geometry(u, v):
    a1 = np.array([1., 0., .4*u+.15*v])
    a2 = np.array([.3, 1.2, .15*u+.6*v])
    a11 = np.array([0., 0., .4])
    a22 = np.array([0., 0., .6])
    a12 = np.array([0., 0., .15])
    cross = np.cross(a1, a2)
    normal = cross / np.linalg.norm(cross)
    def normal_derivative(d1, d2):
        dc = np.cross(d1, a2)+np.cross(a1, d2)
        return (dc-normal*np.dot(normal, dc))/np.linalg.norm(cross)
    contra = np.column_stack((a1, a2)) @ np.linalg.inv(
        np.array([[a1@a1, a1@a2], [a2@a1, a2@a2]]))
    return (a1, a2, normal, contra[:, 0], contra[:, 1],
            a11, a22, a12, a12, normal_derivative(a11, a12),
            normal_derivative(a12, a22))


def shape(u, v):
    return 1+.2*u+.3*v+.1*u*v


class RM5KinematicsTests(unittest.TestCase):
    def test_curved_skew_geometry_matches_director_derivatives(self):
        u, v = .37, .61
        data = geometry(u, v)
        a1, a2, normal = data[:3]
        n1, n2 = .2+.1*v, .3+.1*u
        for module, indices in ((RMshell5p, (3, 4)), (RMshell5p_v2, (0, 1))):
            matrices = module.compute_B_matrix(shape(u, v), n1, n2, *data)
            for dof in range(5):
                values = np.eye(5)[dof]
                displacement1, displacement2 = n1*values[:3], n2*values[:3]
                def director(x, y):
                    bases = geometry(x, y)
                    return shape(x, y)*(values[3]*bases[indices[0]]
                                        + values[4]*bases[indices[1]])
                step = 1e-5
                d = director(u, v)
                d1 = (director(u+step, v)-director(u-step, v))/(2*step)
                d2 = (director(u, v+step)-director(u, v-step))/(2*step)
                expected = (
                    [a1@displacement1, a2@displacement2,
                     a1@displacement2+a2@displacement1],
                    [data[9]@displacement1+a1@d1,
                     data[10]@displacement2+a2@d2,
                     data[10]@displacement1+data[9]@displacement2+a1@d2+a2@d1],
                    [normal@displacement1+a1@d, normal@displacement2+a2@d])
                for matrix, strain in zip(matrices, expected):
                    np.testing.assert_allclose(matrix@values, strain, atol=2e-10, rtol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
