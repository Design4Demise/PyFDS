from typing import Tuple

import numpy as np


def Quaternion2Euler(q: np.ndarray) -> Tuple[float, float, float]:

    """Transform Quaternions to Euler Angles.

    Parameters
    ----------
    q: np.ndarray
        Quaternion to transform.

    Returns
    -------
    phi: float
        Phi Angle.
    theta: float
        Theta Angle.
    psi: float
        Psi Angle.
    """

    phi = np.arctan2(2.0 * (q[0] * q[1] + q[2] * q[3]), q[0] ** 2.0 + q[3] ** 2.0 - q[1] ** 2.0 - q[2] ** 2.0)
    theta = np.arcsin(2.0 * (q[0] * q[2] - q[1] * q[3]))
    psi = np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), q[0] ** 2.0 + q[1] ** 2.0 - q[2] ** 2.0 - q[3] ** 2.0)

    return phi, theta, psi


def Quaternion2Rotation(q: np.ndarray) -> np.ndarray:

    """Transform Quaternion to Rotation Matrix.

    Parameters
    ----------
    q: np.ndarray
        Quaternion to transform to rotation matrix.

    Returns
    -------
    rot: np.ndarray
        Rotation Matrix.
    """

    rot = np.array([[q[1] ** 2.0 + q[0] ** 2.0 - q[2] ** 2.0 - q[3] ** 2.0,
                   2.0 * (q[1] * q[2] - q[3] * q[0]),
                   2.0 * (q[1] * q[3] + q[2] * q[0])],
                  [2.0 * (q[1] * q[2] + q[3] * q[0]),
                   q[2] ** 2.0 + q[0] ** 2.0 - q[1] ** 2.0 - q[3] ** 2.0,
                   2.0 * (q[2] * q[3] - q[1] * q[0])],
                  [2.0 * (q[1] * q[3] - q[2] * q[0]),
                   2.0 * (q[2] * q[3] + q[1] * q[0]),
                   q[3] ** 2.0 + q[0] ** 2.0 - q[1] ** 2.0 - q[2] ** 2.0]])

    rot = rot / np.linalg.det(rot)

    return rot


def Euler2Quaternion(phi: float, theta: float, psi: float) -> np.ndarray:

    """Transform Euler Angles to Quaternion.

    Parameters
    ----------
    phi: float
        Phi Angle.
    theta: float
        Theta Angle.
    psi: float
        Psi Angle.

    Returns
    -------
    np.ndarray
        Quaternion from Euler Angles.
    """

    e0 = (np.cos(psi / 2.0) * np.cos(theta / 2.0) * np.cos(phi / 2.0)
          + np.sin(psi / 2.0) * np.sin(theta / 2.0) * np.sin(phi / 2.0))

    e1 = (np.cos(psi / 2.0) * np.cos(theta/ 2.0) * np.sin(phi / 2.0)
          - np.sin(psi / 2.0) * np.sin(theta / 2.0) * np.cos(phi / 2.0))

    e2 = (np.cos(psi / 2.0) * np.sin(theta / 2.0) * np.cos(phi / 2.0)
          + np.sin(psi / 2.0) * np.cos(theta / 2.0) * np.sin(phi / 2.0))

    e3 = (np.sin(psi / 2.0) * np.cos(theta / 2.0) * np.cos(phi / 2.0)
          - np.cos(psi / 2.0) * np.sin(theta / 2.0) * np.sin(phi / 2.0))

    return np.array([e0, e1, e2, e3])


def Euler2Rotation(phi: float, theta: float, psi: float) -> np.ndarray:

    """Transform Euler Angles to Rotation Matrix.

    Parameters
    ----------
    phi: float
        Phi Angle.
    theta: float
        Theta Angle.
    psi: float
        Psi Angle.

    Returns
    -------
    rot: np.ndarray
        Rotation Matrix.
    """

    r_roll = np.array([[1, 0, 0],
                       [0, np.cos(phi), -np.sin(phi)],
                       [0, np.sin(phi), np.cos(phi)]])

    r_pitch = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])

    r_yaw = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])

    rot = np.matmul(r_yaw, np.matmul(r_pitch, r_roll))

    return rot
