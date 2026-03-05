import sys
import os
import json
from pathlib import Path
from natsort import natsorted
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QLabel, QSizePolicy, QSlider, QMenuBar,
    QAction, QToolBar, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QColorDialog,
)
from PyQt5 import QtCore, QtWidgets, QtGui
import matplotlib.pyplot as plt
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor, QIcon
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLScatterPlotItem
from widget.opengl_widget import PCDViewWidget
from utils.utils import pil2qicon
from utils.load_pcd import get_points_from_pcd_file
import numpy as np
from utils.load_bboxes_json import get_anno_from_tanway_json
from pyqtgraph import Vector
from utils.utils import load_json


def draw_arc_arrow(start_pos, end_pos, color, num_segments=24, arc_height_ratio=0.25):
    """
    绘制从 start_pos 到 end_pos 的弧线箭头（用于 Semitrailer 指向 link_id 目标）。
    使用二次 Bézier 曲线，弧线在竖直方向隆起。
    """
    start = np.array(start_pos, dtype=np.float64)
    end = np.array(end_pos, dtype=np.float64)
    mid = (start + end) / 2
    dist = np.linalg.norm(end - start)
    if dist < 1e-6:
        return None
    # 控制点：中点上方隆起，形成弧线
    curve_height = dist * arc_height_ratio
    control = mid + np.array([0, 0, curve_height])
    # 二次 Bézier: B(t) = (1-t)^2*P0 + 2*(1-t)*t*P1 + t^2*P2
    t = np.linspace(0, 1, num_segments + 1)
    arc_pts = np.column_stack([
        (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0],
        (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1],
        (1 - t) ** 2 * start[2] + 2 * (1 - t) * t * control[2] + t ** 2 * end[2],
    ])
    # 箭头：在末端添加小箭头
    dir_to_end = end - arc_pts[-2]
    dir_len = np.linalg.norm(dir_to_end)
    if dir_len > 1e-6:
        dir_norm = dir_to_end / dir_len
        arrow_len = min(0.3, dist * 0.15)
        arrow_w = arrow_len * 0.4
        if np.abs(dir_norm[2]) < 0.99:
            perp = np.array([-dir_norm[1], dir_norm[0], 0])
        else:
            perp = np.array([1, 0, 0])
        perp = perp / (np.linalg.norm(perp) + 1e-9)
        base = end - arrow_len * dir_norm
        p1 = base + arrow_w * perp
        p2 = base - arrow_w * perp
        points = np.vstack([arc_pts, p1, end, p2])
    else:
        points = arc_pts
    line = gl.GLLinePlotItem(pos=points, color=color, width=2, antialias=True)
    return line


def draw_arc_arrow_missing(start_pos, color, phantom_height=2.5, num_segments=24):
    """
    绘制目标不存在的弧线标记：从 start_pos 向上延伸至 phantom_height 高度，
    表示 link_id 指向的目标框在当前帧中不存在。
    """
    start = np.array(start_pos, dtype=np.float64)
    end = start + np.array([0, 0, phantom_height])
    return draw_arc_arrow(start, end, color, num_segments=num_segments, arc_height_ratio=0.15)


def draw_arrow(start_position, direction, length, color):
    end_point = start_position + length * np.array(direction)
    direction_normalized = np.array(direction) / np.linalg.norm(direction)
    arrowhead_length = 0.3 * length
    arrowhead_width = 0.1 * length
    arrowhead_base = end_point - arrowhead_length * direction_normalized
    if np.allclose(direction_normalized[:2], [0, 0]):
        perp_vector = np.array([1, 0, 0])
    else:
        perp_vector = np.array([-direction_normalized[1], direction_normalized[0], 0])
    perp_vector = perp_vector / np.linalg.norm(perp_vector)
    arrowhead_point1 = arrowhead_base + arrowhead_width * perp_vector
    arrowhead_point2 = arrowhead_base - arrowhead_width * perp_vector
    points = np.vstack([start_position, end_point, arrowhead_point1, end_point, arrowhead_point2])
    arrow = gl.GLLinePlotItem(pos=points, color=color, width=2, antialias=True)
    return arrow


def draw_bbox(x, y, z, l, w, h, yaw, color):
    """绘制 3D 目标框（线框，原始逻辑）"""
    deg_yaw = np.rad2deg(yaw)
    bbox = gl.GLBoxItem(size=QtGui.QVector3D(l, w, h), color=color, glOptions='opaque')
    bbox.translate(-l / 2, -w / 2, -h / 2)
    bbox.rotate(deg_yaw, 0, 0, 1)
    bbox.translate(x, y, z)
    return bbox


def draw_bbox_solid(x, y, z, l, w, h, yaw, color):
    """
    绘制 3D 目标框（全包围半透明实体，用于选中高亮）。
    与图中绿色框一致：填充面、半透明，可透视内部点云。
    """
    from pyqtgraph.opengl import GLMeshItem, MeshData
    hl, hw, hh = l / 2, w / 2, h / 2
    verts = np.array([
        [-hl, -hw, -hh], [hl, -hw, -hh], [-hl, hw, -hh], [hl, hw, -hh],
        [-hl, -hw, hh], [hl, -hw, hh], [-hl, hw, hh], [hl, hw, hh],
    ], dtype=np.float32)
    faces = np.array([
        [0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 6, 7],
        [0, 1, 4], [1, 4, 5], [2, 6, 3], [3, 6, 7],
        [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 5, 7],
    ], dtype=np.uint32)
    r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
    if a <= 0:
        a = 180
    face_colors = np.tile([r / 255, g / 255, b / 255, a / 255], (12, 1)).astype(np.float32)
    md = MeshData(vertexes=verts, faces=faces, faceColors=face_colors)
    mesh = GLMeshItem(meshdata=md, smooth=False, glOptions='translucent', drawEdges=True, edgeColor=(r / 255, g / 255, b / 255, 0.8))
    deg_yaw = np.rad2deg(yaw)
    mesh.rotate(deg_yaw, 0, 0, 1)
    mesh.translate(x, y, z)
    return mesh

