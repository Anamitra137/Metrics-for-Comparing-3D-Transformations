# Loss functions for comparing two rotations

import numpy as np
from scipy.spatial.transform import Rotation


# Utility functions

def Rot2Quat(R: np.ndarray) -> np.ndarray:
    """
    From a 3x3 rotation matrix get the quaternion representation

    Parameters
    ----------
    R : np.ndarray
        3x3 rotation matrix

    Returns
    -------
    np.ndarray
        Quaternion representation of the rotation matrix
    """
    r = Rotation.from_matrix(R)
    q = r.as_quat()

    # Check if the scalar part is negative
    if q[3] < 0:
        q = -q

    return q


def Rot2Euler(R: np.ndarray) -> np.ndarray:
    """
    From a 3x3 rotation matrix get the euler angles representation

    Parameters
    ----------
    R : np.ndarray
        3x3 rotation matrix

    Returns
    -------
    np.ndarray
        Euler angles representation of the rotation matrix
    """
    r = Rotation.from_matrix(R)
    euler = r.as_euler('xyz', degrees=False)

    return euler




# Losses

def loss1(R1, R2):
    """
    Loss function 1:
        f((a1, b1, c1), (a2, b2, c2)) = sqrt( d(a1, a2)^2 + d(b1, b2)^2 + d(c1, c2)^2 )
        where d(a1, a2) = min( abs(a1 - a2), 2*PI - abs(a1 - a2) )
        and (a, b, c) are Euler angles
    """

    a1, b1, c1 = Rot2Euler(R1)
    a2, b2, c2 = Rot2Euler(R2)

    d_a = min(abs(a1 - a2), 2*np.pi - abs(a1 - a2))
    d_b = min(abs(b1 - b2), 2*np.pi - abs(b1 - b2))
    d_c = min(abs(c1 - c2), 2*np.pi - abs(c1 - c2))

    return np.sqrt(d_a**2 + d_b**2 + d_c**2)


def loss2(R1, R2):
    """
    Loss function 2:
        f(q1, q2) = min( || q1 - q2 ||, || q1 + q2 || )
        where q1 and q2 are quaternions
    """
    q1 = Rot2Quat(R1)
    q2 = Rot2Quat(R2)

    return min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2))

def loss3(R1, R2):
    """
    Loss function 3:
        f(q1, q2) = 1 - | q1 . q2 |
        where q1 and q2 are quaternions
    """
    q1 = Rot2Quat(R1)
    q2 = Rot2Quat(R2)

    return 1 - abs(np.dot(q1, q2))

def loss4(R1, R2):
    """
    Loss function 4:
        f(R1, R2) = || I - R1R2^T||
    """
    return np.linalg.norm(np.eye(3) - np.dot(R1, R2.T))

# def loss5(R1, R2):
#     """
#     Loss function 5:
#         f(R1, R2) = || log(R1R2^T) ||
#     """
#     return np.linalg.norm(np.log(np.dot(R1, R2.T)))



# Test
def main():
    R1 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    R2 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    print(loss1(R1, R2))


if __name__ == "__main__":
    main()