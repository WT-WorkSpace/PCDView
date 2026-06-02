# -*- coding: utf-8 -*-
"""点云 ICP 刚体配准（依赖 numpy / scipy，无 Open3D）。"""
import numpy as np
from scipy.spatial import cKDTree


def voxel_downsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    if xyz is None or len(xyz) == 0:
        return xyz
    if voxel_size <= 0:
        return xyz
    vox = np.floor(xyz / float(voxel_size)).astype(np.int64)
    _, idx = np.unique(vox, axis=0, return_index=True)
    return xyz[idx]


def random_downsample(xyz: np.ndarray, max_points: int) -> np.ndarray:
    if xyz is None or len(xyz) <= max_points:
        return xyz
    step = max(1, len(xyz) // max_points)
    return xyz[::step]


def _best_fit_rigid(source: np.ndarray, target: np.ndarray):
    """求 R,t：target ≈ R @ source + t"""
    c_src = source.mean(axis=0)
    c_tgt = target.mean(axis=0)
    a = source - c_src
    b = target - c_tgt
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = c_tgt - r @ c_src
    return r, t


def _rt4(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 40,
    tolerance: float = 1e-5,
    max_correspondence_distance: float = 2.0,
) -> tuple:
    """
    点到点 ICP，将 source 配准到 target。
    返回 (4x4 变换矩阵 T, 最终均方误差, 是否收敛)。
    应用方式：p_aligned = (T @ [p;1])[:3]
    """
    src = np.asarray(source[:, :3], dtype=np.float64).copy()
    tgt = np.asarray(target[:, :3], dtype=np.float64)
    if len(src) < 10 or len(tgt) < 10:
        raise ValueError("ICP 至少需要每侧 10 个点")

    t_accum = np.eye(4, dtype=np.float64)
    tree = cKDTree(tgt)
    prev_rmse = np.inf

    for _ in range(max_iterations):
        dists, idx = tree.query(src, k=1, distance_upper_bound=max_correspondence_distance)
        valid = np.isfinite(dists)
        if np.count_nonzero(valid) < 10:
            break
        src_m = src[valid]
        tgt_m = tgt[idx[valid]]
        r, t = _best_fit_rigid(src_m, tgt_m)
        src = (r @ src.T).T + t
        t_step = _rt4(r, t)
        t_accum = t_step @ t_accum
        rmse = float(np.sqrt(np.mean(dists[valid] ** 2)))
        if abs(prev_rmse - rmse) < tolerance:
            return t_accum, rmse, True
        prev_rmse = rmse

    dists, _ = tree.query(src, k=1, distance_upper_bound=max_correspondence_distance)
    valid = np.isfinite(dists)
    rmse = float(np.sqrt(np.mean(dists[valid] ** 2))) if np.any(valid) else np.inf
    return t_accum, rmse, False


def icp_fitness(
    source: np.ndarray,
    target: np.ndarray,
    max_correspondence_distance: float = 2.0,
) -> tuple:
    """
    单次评估：将 source 用单位变换对齐到 target 的匹配质量。
    返回 (rmse, inlier_ratio, inlier_count)。
    """
    src = np.asarray(source[:, :3], dtype=np.float64)
    tgt = np.asarray(target[:, :3], dtype=np.float64)
    if len(src) < 10 or len(tgt) < 10:
        return np.inf, 0.0, 0
    tree = cKDTree(tgt)
    dists, _ = tree.query(src, k=1, distance_upper_bound=max_correspondence_distance)
    valid = np.isfinite(dists)
    n_in = int(np.count_nonzero(valid))
    if n_in < 10:
        return np.inf, 0.0, n_in
    rmse = float(np.sqrt(np.mean(dists[valid] ** 2)))
    ratio = n_in / float(len(src))
    return rmse, ratio, n_in


def auto_voxel_size(xyz: np.ndarray, default: float = 0.15) -> float:
    if xyz is None or len(xyz) < 2:
        return default
    ext = np.ptp(xyz[:, :3], axis=0)
    span = float(np.max(ext))
    if span <= 0:
        return default
    return max(default, span / 200.0)
