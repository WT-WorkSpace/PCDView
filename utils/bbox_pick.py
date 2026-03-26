# -*- coding: utf-8 -*-
"""从 3D 视图屏幕坐标拾取与射线与 OBB 相交检测"""
import numpy as np
from PyQt5.QtGui import QVector3D, QVector4D, QMatrix4x4


def ray_from_screen(glwidget, screen_x, screen_y):
    """
    根据 GLViewWidget 的投影与视图矩阵，将屏幕坐标转换为世界空间射线。
    返回 (origin, direction)，均为 numpy 数组 shape (3,)。
    """
    w = glwidget.deviceWidth()
    h = glwidget.deviceHeight()
    if w <= 0 or h <= 0:
        return None
    x_ndc = 2.0 * screen_x / w - 1.0
    y_ndc = 1.0 - 2.0 * screen_y / h
    proj = glwidget.projectionMatrix()
    view = glwidget.viewMatrix()
    combined = proj * view
    inv_result = combined.inverted()
    if isinstance(inv_result, tuple):
        inv, invertible = inv_result
        if not invertible:
            return None
    else:
        inv = inv_result
    near_pt = inv.map(QVector3D(x_ndc, y_ndc, -1.0))
    far_pt = inv.map(QVector3D(x_ndc, y_ndc, 1.0))
    origin = np.array([near_pt.x(), near_pt.y(), near_pt.z()], dtype=np.float64)
    end = np.array([far_pt.x(), far_pt.y(), far_pt.z()], dtype=np.float64)
    direction = end - origin
    n = np.linalg.norm(direction)
    if n < 1e-9:
        return None
    direction = direction / n
    return origin, direction


def world_to_screen(glwidget, world_pt):
    """
    将世界坐标点投影到屏幕坐标。
    world_pt: (3,) 或 (N,3) 数组。
    返回 (sx, sy) 或 (N,2) 数组，屏幕坐标系 (0,0) 在左上角。
    """
    w = glwidget.deviceWidth()
    h = glwidget.deviceHeight()
    if w <= 0 or h <= 0:
        return None
    proj = glwidget.projectionMatrix()
    view = glwidget.viewMatrix()
    combined = proj * view
    pts = np.atleast_2d(np.asarray(world_pt, dtype=np.float64))
    n = pts.shape[0]
    out = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        qpt = QVector4D(float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]), 1.0)
        clip = combined.map(qpt)
        cw = clip.w()
        if abs(cw) < 1e-9:
            out[i] = np.nan, np.nan
        else:
            x_ndc = clip.x() / cw
            y_ndc = clip.y() / cw
            out[i, 0] = (x_ndc + 1.0) * 0.5 * w
            out[i, 1] = (1.0 - y_ndc) * 0.5 * h
    return out


def filter_ground_points(points_xyz, height_threshold=0.7, percentile=20):
    """
    过滤地面点。保留 z > z_min + threshold 的点，或高于下 percentile 分位数的点。
    points_xyz: (N,3) 数组。
    返回过滤后的点数组。
    """
    if points_xyz is None or len(points_xyz) < 4:
        return points_xyz
    if len(points_xyz) < 30:
        return points_xyz
    pts = np.asarray(points_xyz, dtype=np.float64)
    z = pts[:, 2]
    z_min, z_max = np.min(z), np.max(z)
    z_range = z_max - z_min
    # 使用固定阈值与高度范围的比例，取较大者
    thresh = max(height_threshold, 0.07 * z_range)
    cutoff = z_min + thresh
    # 或使用分位数：去掉最低 percentile 的点
    pct_cutoff = np.percentile(z, percentile)
    cutoff = max(cutoff, pct_cutoff)
    mask = z > cutoff
    filtered = pts[mask]
    if len(filtered) < 3:
        return pts  # 过滤后太少则返回原有点
    return filtered


def points_in_screen_rect(glwidget, points_xyz, x_min, x_max, y_min, y_max):
    """
    筛选投影到屏幕矩形 [x_min,x_max] x [y_min,y_max] 内的 3D 点。
    points_xyz: (N,3) 数组。
    返回 mask 布尔数组，True 表示在矩形内。
    """
    if points_xyz is None or len(points_xyz) == 0:
        return np.array([], dtype=bool)
    screen = world_to_screen(glwidget, points_xyz)
    if screen is None:
        return np.zeros(len(points_xyz), dtype=bool)
    if screen.ndim == 1:
        return np.array([True])
    in_rect = (screen[:, 0] >= x_min) & (screen[:, 0] <= x_max) & (screen[:, 1] >= y_min) & (screen[:, 1] <= y_max)
    return in_rect & (~np.isnan(screen[:, 0]))


def fit_obb_xy(points_xy, roll=0, pitch=0):
    """
    对 2D 点拟合最小包围框，roll=0 pitch=0 约束下仅优化 yaw。
    使用 PCA 确定主方向作为 yaw。
    返回 (cx, cy, l, w, yaw)，其中 l 沿主方向，w 垂直。
    """
    if points_xy is None or len(points_xy) < 2:
        return None
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    pts = pts[:, :2]
    center = np.mean(pts, axis=0)
    centered = pts - center
    cov = np.cov(centered.T)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        l = float(np.ptp(pts[:, 0]))
        w = float(np.ptp(pts[:, 1]))
        return center[0], center[1], max(l, 0.1), max(w, 0.1), 0.0
    try:
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigenvals)[::-1]
        main_dir = eigenvecs[:, idx[0]]
        yaw = np.arctan2(main_dir[1], main_dir[0])
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        local_x = cos_yaw * centered[:, 0] - sin_yaw * centered[:, 1]
        local_y = sin_yaw * centered[:, 0] + cos_yaw * centered[:, 1]
        l = float(np.ptp(local_x))
        w = float(np.ptp(local_y))
        l = max(l, 0.1)
        w = max(w, 0.1)
        return center[0], center[1], l, w, yaw
    except Exception:
        l = float(np.ptp(pts[:, 0]))
        w = float(np.ptp(pts[:, 1]))
        return center[0], center[1], max(l, 0.1), max(w, 0.1), 0.0


def ray_plane_z_intersect(ray_origin, ray_dir, z_plane):
    """
    射线与水平面 z=z_plane 相交。返回交点 (x,y,z)，未相交返回 None。
    """
    if abs(ray_dir[2]) < 1e-9:
        return None
    t = (z_plane - ray_origin[2]) / ray_dir[2]
    if t < 0:
        return None
    pt = ray_origin + t * ray_dir
    return pt


def ray_obb_intersect(ray_origin, ray_dir, center, half_extents, yaw):
    """
    射线与 OBB 相交检测。OBB 中心 center=(x,y,z)，半轴 half_extents=(l/2, w/2, h/2)，
    仅绕 Z 轴旋转 yaw（弧度）。返回相交时射线参数 t（t>=0），未相交返回 None。
    """
    cx, cy, cz = center[0], center[1], center[2]
    lx, ly, lz = half_extents[0], half_extents[1], half_extents[2]
    # 与主视图 draw_bbox 一致：朝向 (cos(yaw), sin(yaw)) 对应 OBB 局部 x 轴
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    ox = ray_origin[0] - cx
    oy = ray_origin[1] - cy
    oz = ray_origin[2] - cz
    o_lx = cos_yaw * ox - sin_yaw * oy
    o_ly = sin_yaw * ox + cos_yaw * oy
    o_lz = oz
    d_lx = cos_yaw * ray_dir[0] - sin_yaw * ray_dir[1]
    d_ly = sin_yaw * ray_dir[0] + cos_yaw * ray_dir[1]
    d_lz = ray_dir[2]
    t_min = -np.inf
    t_max = np.inf
    for i, (o, d, lo, hi) in enumerate([
        (o_lx, d_lx, -lx, lx),
        (o_ly, d_ly, -ly, ly),
        (o_lz, d_lz, -lz, lz),
    ]):
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return None
            continue
        t1 = (lo - o) / d
        t2 = (hi - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max or t_max < 0:
            return None
    if t_min >= 0:
        return t_min
    if t_max >= 0:
        return t_max
    return None


def pick_bbox_index(glwidget, mouse_x, mouse_y, bbox_infos):
    """
    bbox_infos: list of dict, 每个 dict 至少包含 'x','y','z','l','w','h','yaw'。
    返回被拾取的 bbox 在列表中的索引；若未拾取到则返回 None。有多个时返回最近的一个。
    """
    ray = ray_from_screen(glwidget, mouse_x, mouse_y)
    if ray is None or not bbox_infos:
        return None
    origin, direction = ray
    best_i = None
    best_t = np.inf
    for i, info in enumerate(bbox_infos):
        try:
            x = float(info["x"]) if info.get("x") is not None else 0.0
            y = float(info["y"]) if info.get("y") is not None else 0.0
            z = float(info["z"]) if info.get("z") is not None else 0.0
            l = float(info["l"]) if info.get("l") is not None else 0.0
            w = float(info["w"]) if info.get("w") is not None else 0.0
            h = float(info["h"]) if info.get("h") is not None else 0.0
            yaw = float(info["yaw"]) if info.get("yaw") is not None else 0.0
        except (TypeError, ValueError, KeyError):
            continue
        t = ray_obb_intersect(
            origin, direction,
            np.array([x, y, z]),
            np.array([l / 2.0, w / 2.0, h / 2.0]),
            yaw,
        )
        if t is not None and 0 <= t < best_t:
            best_t = t
            best_i = i
    return best_i
