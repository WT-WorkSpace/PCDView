import numpy as np
from scipy.spatial.transform import Rotation


def xyzrpy2RTmatrix(xyz_rpy,seq="xyz", degrees=False):
    assert len(xyz_rpy) == 6
    dx, dy, dz, roll, pitch, yaw = xyz_rpy
    r = Rotation.from_euler(seq, [roll, pitch, yaw], degrees=degrees)
    rotation_matrix = r.as_matrix()
    translation = np.array([dx, dy, dz])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation
    return matrix

def RTmatrix2xyzrpy(RTmatrix,seq="xyz", degrees=False):
    translation = RTmatrix[:3, 3]
    rotation_matrix = RTmatrix[:3, :3]
    r = Rotation.from_matrix(rotation_matrix)
    rpy = r.as_euler(seq, degrees=degrees)
    dx, dy, dz = translation
    roll, pitch, yaw = rpy
    return np.array([dx, dy, dz, roll, pitch, yaw])

def move_pcd_with_RTmatrix(points, RTmatrix,inv=False):
    if inv:
        RTmatrix = np.linalg.inv(RTmatrix)
    pcd_trans = points.copy()
    pcd_hm = np.pad(points[:, :3], ((0, 0), (0, 1)), 'constant', constant_values=1)  # (N, 4)
    pcd_hm_trans = np.dot(RTmatrix, pcd_hm.T).T
    pcd_trans[:, :3] = pcd_hm_trans[:, :3]
    return pcd_trans

def rotate_pointcloud(points,rpy,seq="xyz",degrees=True):
    r = Rotation.from_euler(seq, rpy, degrees=degrees)
    rotation_matrix = r.as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    new_points = move_pcd_with_RTmatrix(points, matrix)
    return new_points

def translate_pointcloud(points,xyz):
    matrix = np.eye(4)
    matrix[:3, 3] = np.array(xyz)
    new_points = move_pcd_with_RTmatrix(points, matrix)
    return new_points


def move_pcd_with_xyzrpy(points, xyz_rpy,seq="xyz", degrees=False):
    assert len(xyz_rpy) == 6
    RT_matrix = xyzrpy2RTmatrix(xyz_rpy,seq, degrees=degrees)
    new_pcd = move_pcd_with_RTmatrix(points, RT_matrix)
    return new_pcd
