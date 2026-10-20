import numpy as np
from IGAeasy.elements.RMshell5p import RMShell as StandardRMShell

"""
RM5 shell with contravariant director coefficients in the physical basis a1, a2.
"""

class RMShell(StandardRMShell):
    def element_routine(self, cell):
        super().element_routine(cell, kinematics=compute_B_matrix)

def compute_B_matrix(Ni, Ni_1, Ni_2,
                     a1, a2, a3,
                     a1_con, a2_con,
                     a1_1, a2_2, a1_2, a2_1,
                     a3_1, a3_2):
    """
    Construct local B-matrices for membrane, bending, and shear.

    :return: (Bm, Bb, Bs) ∈ ℝ^{(3×5, 3×5, 2×5)}
    """
    # Membrane
    Bm = np.zeros((3, 5))
    Bm[0, :3] = Ni_1 * a1
    Bm[1, :3] = Ni_2 * a2
    Bm[2, :3] = Ni_2 * a1 + Ni_1 * a2

    # Bending
    Bb = np.zeros((3, 5))
    Bb[0, :3] = Ni_1 * a3_1
    Bb[0, 3] = Ni_1*np.dot(a1,a1) + Ni*np.dot(a1_1,a1)
    Bb[0, 4] = Ni_1*np.dot(a2,a1) + Ni*np.dot(a2_1,a1)

    Bb[1, :3] = Ni_2 * a3_2
    Bb[1, 3] = Ni_2*np.dot(a1,a2) + Ni*np.dot(a1_2,a2)
    Bb[1, 4] = Ni_2*np.dot(a2,a2) + Ni*np.dot(a2_2,a2)

    Bb[2, :3] = Ni_1 * a3_2 + Ni_2 * a3_1
    Bb[2, 3] = Ni_1*np.dot(a1,a2) + Ni_2*np.dot(a1,a1) + Ni*np.dot(a1_1,a2) + Ni*np.dot(a1_2,a1)
    Bb[2, 4] = Ni_1*np.dot(a2,a2) + Ni_2*np.dot(a2,a1) + Ni*np.dot(a2_1,a2) + Ni*np.dot(a2_2,a1)

    # Shear
    Bs = np.zeros((2, 5))
    Bs[0, :3] = Ni_1 * a3
    Bs[0, 3] = Ni*np.dot(a1,a1)
    Bs[0, 4] = Ni*np.dot(a2,a1)
    Bs[1, :3] = Ni_2 * a3
    Bs[1, 3] = Ni*np.dot(a1,a2)
    Bs[1, 4] = Ni*np.dot(a2,a2)

    return Bm, Bb, Bs



def get_dofs_from_cps(cps, loc_dofs, dof_per_cp=5):
    from IGAeasy.constraints.prescribed_dofs import control_point_dofs
    return control_point_dofs(cps, loc_dofs, ('x', 'y', 'z', 'r_a1', 'r_a2'), dof_per_cp)

