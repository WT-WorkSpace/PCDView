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
    deg_yaw = np.rad2deg(yaw)
    bbox = gl.GLBoxItem(size=QtGui.QVector3D(l, w, h), color=color, glOptions='opaque')
    bbox.translate(-l / 2, -w / 2, -h / 2)
    bbox.rotate(deg_yaw, 0, 0, 1)
    bbox.translate(x, y, z)
    return bbox

