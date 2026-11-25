"""SO(3) / quaternion utilities. Hamilton convention, R = R_WI (body-to-world)."""
import numpy as np


def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def so3_exp(phi):
    angle = np.linalg.norm(phi)
    if angle < 1e-9:
        return np.eye(3) + skew(phi)
    axis = phi / angle
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def so3_log(R):
    cos_angle = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    return angle / (2.0 * np.sin(angle)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def so3_right_jacobian(phi):
    angle = np.linalg.norm(phi)
    if angle < 1e-9:
        return np.eye(3) - 0.5 * skew(phi)
    K = skew(phi / angle)
    return (np.eye(3)
            - (1.0 - np.cos(angle)) / angle * K
            + (1.0 - np.sin(angle) / angle) * (K @ K))


def quat_to_rot(q):
    """Hamilton quaternion (w, x, y, z) -> rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R):
    """Rotation matrix -> Hamilton quaternion (w, x, y, z), w >= 0."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)
