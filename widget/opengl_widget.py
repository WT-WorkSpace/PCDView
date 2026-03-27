import os
import json
import numpy as np
from pathlib import Path
from pyqtgraph import Vector
import pyqtgraph.opengl as gl
from utils.utils import load_json, text_3d
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor, QIcon, QQuaternion, QMouseEvent
from PyQt5.QtWidgets import QFileDialog, QWidget
from PyQt5 import QtGui
import matplotlib.pyplot as plt
from utils.colors import *
from pyqtgraph import Vector
from pyqtgraph.opengl import GLScatterPlotItem, GLTextItem, GLScatterPlotItem
from PyQt5.QtGui import QFont, QColor, QIcon, QQuaternion, QMouseEvent
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QTimer, Qt

from scipy.spatial.transform import Rotation
from PyQt5.QtWidgets import QAction, QToolBar, QWidget, QPushButton, QColorDialog


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

def move_pcd_with_xyzrpy(points, xyz_rpy,seq, degrees=False):
    assert len(xyz_rpy) == 6
    RT_matrix = xyzrpy2RTmatrix(xyz_rpy,seq, degrees=degrees)
    new_pcd = move_pcd_with_RTmatrix(points, RT_matrix)
    return new_pcd



class PCDViewWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.glwidget = gl.GLViewWidget()
        self.glwidget.setWindowTitle('PointCloudViewer')

        self.colors_16 = get_colors_16()
        self.class_map = get_class_map()

        self.axis_visible = False
        self.add_bboxes = False
        self.color_fields = None
        self.bboxes_directory = None
        # 坐标轴由 3 条粗线段构成（比 GLAxisItem 的默认细线更粗）
        self.axis = None

        """调整视角：默认俯视图"""
        self.glwidget.opts['distance'] = 15
        self.glwidget.setCameraPosition(distance=self.glwidget.opts['distance'], elevation=90, azimuth=0)

        """添加点云到视图窗口中"""
        curpath = os.path.dirname(os.path.abspath(__file__))
        text_points = text_3d("Point Cloud Viewer", density=2, font=os.path.join(curpath,'../icons/fengguangming.ttf'), font_size=10)
        text_points = move_pcd_with_xyzrpy(text_points, [0,0,0,0,-90,0],seq="xyz", degrees=True)
        self.raw_points = text_points

        self.Colors = [get_bgyord_bar(), plt.get_cmap('cool'), plt.get_cmap('GnBu'), plt.get_cmap('Greys'), plt.get_cmap('hot')]  # 参考:https://zhuanlan.zhihu.com/p/114420786
        self.colors = self.Colors[0](self.raw_points[:, 0] / 80)

        self.point_size_list = [2, 1, 1.3, 1.7, 2.3, 2.7, 3, 4, 5]
        self.point_size = self.point_size_list[0]
        self.scatter = gl.GLScatterPlotItem(pos=self.raw_points, color=self.colors, size=self.point_size)
        self.glwidget.addItem(self.scatter)


    def create_coordinate(self):
        if self.axis_visible:
            if self.axis:
                # self.axis 可能是 GLScatter/GLLine 的列表
                if isinstance(self.axis, (list, tuple)):
                    for it in self.axis:
                        if it is not None:
                            self.glwidget.removeItem(it)
                else:
                    self.glwidget.removeItem(self.axis)
                self.axis = None
            self.axis_visible = False
        else:
            axis_len = 3
            # width 单位为屏幕像素宽度，可按需调大
            axis_width = 4

            # 三条粗直线：X 红、Y 绿、Z 蓝
            x_line = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [axis_len, 0, 0]], dtype=np.float32),
                color=(1.0, 0.2, 0.2, 1.0),
                width=axis_width,
                antialias=True,
            )
            y_line = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [0, axis_len, 0]], dtype=np.float32),
                color=(0.2, 1.0, 0.2, 1.0),
                width=axis_width,
                antialias=True,
            )
            z_line = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [0, 0, axis_len]], dtype=np.float32),
                color=(0.2, 0.2, 1.0, 1.0),
                width=axis_width,
                antialias=True,
            )
            self.glwidget.addItem(x_line)
            self.glwidget.addItem(y_line)
            self.glwidget.addItem(z_line)
            self.axis = [x_line, y_line, z_line]
            self.axis_visible = True
