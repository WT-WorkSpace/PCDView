"""SLAM pose parsing and point-cloud transformation helpers."""

import re

import numpy as np
from scipy.spatial.transform import Rotation


def _numeric_tokens(line):
    """Return the numeric columns in a pose line, ignoring an optional frame name."""
    line = line.split("#", 1)[0].strip()
    if not line:
        return []
    tokens = [token for token in re.split(r"[\s,;]+", line) if token]
    values = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            # A filename/frame id is commonly stored in the first column.
            if values:
                raise ValueError("位姿行中包含无法解析的字段: {}".format(token))
    return values


def pose_values_to_matrix(values):
    """Convert a common pose row to a world_T_sensor 4x4 matrix.

    Supported layouts are:
      x y z roll pitch yaw            (radians)
      x y z qx qy qz qw
      timestamp x y z qx qy qz qw
      3x4 matrix (12 values)
      4x4 matrix (16 values)
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 8:
        values = values[1:]

    matrix = np.eye(4, dtype=np.float64)
    if len(values) == 6:
        matrix[:3, :3] = Rotation.from_euler("xyz", values[3:6], degrees=False).as_matrix()
        matrix[:3, 3] = values[:3]
    elif len(values) == 7:
        quaternion = values[3:7]
        norm = np.linalg.norm(quaternion)
        if norm <= np.finfo(float).eps:
            raise ValueError("四元数不能全为 0")
        matrix[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
        matrix[:3, 3] = values[:3]
    elif len(values) == 12:
        matrix[:3, :] = values.reshape(3, 4)
    elif len(values) == 16:
        matrix = values.reshape(4, 4)
    else:
        raise ValueError("每行需要 6、7、8、12 或 16 个数，当前为 {} 个".format(len(values)))

    if not np.all(np.isfinite(matrix)):
        raise ValueError("位姿包含 NaN 或无穷大")
    return matrix


def load_pose_file(path):
    """Load all non-empty pose rows from *path*."""
    poses = []
    with open(path, "r", encoding="utf-8-sig") as pose_file:
        for line_number, line in enumerate(pose_file, 1):
            try:
                values = _numeric_tokens(line)
                if values:
                    poses.append(pose_values_to_matrix(values))
            except ValueError as exc:
                raise ValueError("第 {} 行: {}".format(line_number, exc)) from exc
    if not poses:
        raise ValueError("文件中没有有效位姿")
    return poses


def transform_xyz(points, pose):
    """Transform an Nx3 (or wider) point array using world_T_sensor."""
    xyz = np.asarray(points)[:, :3]
    return xyz.dot(np.asarray(pose)[:3, :3].T) + np.asarray(pose)[:3, 3]
