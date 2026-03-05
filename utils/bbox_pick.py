# -*- coding: utf-8 -*-
"""从 3D 视图屏幕坐标拾取与射线与 OBB 相交检测"""
import numpy as np
from PyQt5.QtGui import QVector3D, QMatrix4x4


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
