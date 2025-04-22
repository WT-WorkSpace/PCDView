import os
import json
import numpy as np
from pathlib import Path
from pyqtgraph import Vector
import pyqtgraph.opengl as gl
from utils.utils import load_json, text_3d


from PyQt5.QtWidgets import QFileDialog, QWidget
from PyQt5 import QtGui
import matplotlib.pyplot as plt
from utils.colors import *


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

        """调整视角"""
        self.glwidget.opts['distance'] = 15
        self.glwidget.setCameraPosition(distance=self.glwidget.opts['distance'], elevation=0, azimuth=0)

        """添加点云到视图窗口中"""
        curpath = os.path.dirname(os.path.abspath(__file__))
        text_points = text_3d("Point Cloud Viewer", density=2, font=os.path.join(curpath,'../icons/fengguangming.ttf'), font_size=10)
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
                self.glwidget.removeItem(self.axis)
                self.axis = None
            self.axis_visible = False
        else:
            self.axis = gl.GLAxisItem()
            self.axis.setSize(x=3, y=3, z=3)
            self.glwidget.addItem(self.axis)
            self.axis_visible = True


    def draw_arrow(self, start_position, direction, length, color):
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

    def draw_bbox(self, x, y, z, l, w, h, yaw, color):

        deg_yaw = np.rad2deg(yaw)
        bbox = gl.GLBoxItem(size=QtGui.QVector3D(l, w, h), color=color, glOptions='opaque')
        bbox.translate(-l / 2, -w / 2, -h / 2)
        bbox.rotate(deg_yaw, 0, 0, 1)
        bbox.translate(x, y, z)
        return bbox
