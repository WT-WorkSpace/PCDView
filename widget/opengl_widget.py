from PyQt5.QtWidgets import QWidget, QFileDialog
import pyqtgraph.opengl as gl
from utils.move_pcd import move_pcd_with_xyzrpy
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageFont, ImageDraw
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def text_3d(text, density=10, font= os.path.join(os.path.dirname(os.path.abspath(__file__)),'../icons/fengguangming.ttf'), font_size=10):
    font_obj = ImageFont.truetype(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../icons/fengguangming.ttf'), int(font_size * density))
    font_dim = font_obj.getbbox(text)
    text_width = int((font_dim[2] - font_dim[0]))  # 计算宽度
    text_height = int((font_dim[3] - font_dim[1])*1.2)  # 计算高度
    img = Image.new('RGB', (text_width, text_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T
    indices = move_pcd_with_xyzrpy(indices,[0,-text_width/2,text_height/2,0,90,0],degrees=True)
    points = indices / 10 / density
    return points

class PCDViewWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.glwidget = gl.GLViewWidget()
        self.glwidget.setWindowTitle('PointCloudViewer')

        """调整视角"""
        self.glwidget.opts['distance'] = 15
        self.glwidget.setCameraPosition(distance=self.glwidget.opts['distance'], elevation=0, azimuth=0)

        """添加点云到视图窗口中"""
        curpath = os.path.dirname(os.path.abspath(__file__))
        text_points = text_3d("Point Cloud Viewer", density=2, font=os.path.join(curpath,'../icons/fengguangming.ttf'), font_size=10)
        self.raw_points = text_points


        colors =  [(0, 'lightblue'), (0.2, 'blue'), (0.8, 'green'), (0.85, 'yellow'), (0.9, 'orange'), (0.95, 'red'),(1, 'darkred')]
        cm = LinearSegmentedColormap.from_list('blue_green_yellow_orange_red_darkred', colors, N=256)
        self.Colors = [cm, plt.get_cmap('cool'), plt.get_cmap('GnBu'), plt.get_cmap('Greys'), plt.get_cmap('hot')]  # 参考:https://zhuanlan.zhihu.com/p/114420786

        self.colors = self.Colors[0](self.raw_points[:, 0] / 80)
        self.point_size_list = [2, 1, 1.3, 1.7, 2.3, 2.7, 3, 4, 5]
        self.point_size = self.point_size_list[0]
        self.scatter = gl.GLScatterPlotItem(pos=self.raw_points, color=self.colors, size=self.point_size)
        self.glwidget.addItem(self.scatter)

        self.colors_16 = [(255, 0, 0),      # 红色
                        (0, 255, 0),      # 绿色
                        (0, 0, 255),      # 蓝色
                        (255, 255, 0),    # 黄色
                        (128, 0, 128),    # 紫色
                        (255, 165, 0),    # 橙色
                        (0, 255, 255),    # 青色
                        (255, 192, 203),  # 粉色
                        (165, 42, 42),    # 棕色
                        (128, 128, 128),  # 灰色
                        (0, 0, 0),        # 黑色
                        (255, 255, 255),  # 白色
                        (255, 0, 127),    # 玫瑰红
                        (127, 255, 0),    # 柠檬绿
                        (0, 191, 255),    # 天蓝色
                        (238, 130, 238)]   # 紫罗兰色

    def save_view(self):
        """
        Save the current camera view (view matrix and projection matrix) to a file.
        """
        # Get the current view matrix and projection matrix
        view_matrix = self.glwidget.viewMatrix()
        projection_matrix = self.glwidget.projectionMatrix()

        # Convert matrices to lists for easy saving (as JSON)
        view_matrix_list = view_matrix.flatten().tolist()
        projection_matrix_list = projection_matrix.flatten().tolist()

        # Save the matrices to a JSON file
        view_data = {
            "view_matrix": view_matrix_list,
            "projection_matrix": projection_matrix_list
        }
        file_name, _ = QFileDialog.getSaveFileName(self, "Save View", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'w') as f:
                json.dump(view_data, f, indent=4)
        print("View saved to:", file_name)

    def load_view(self):
        """
        Load a saved camera view from a file and apply it to the current view.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Open View", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'r') as f:
                view_data = json.load(f)
            view_matrix = np.array(view_data["view_matrix"]).reshape(4, 4)
            projection_matrix = np.array(view_data["projection_matrix"]).reshape(4, 4)
            self.glwidget.setViewMatrix(view_matrix)
            self.glwidget.setProjectionMatrix(projection_matrix)
            print("View loaded from:", file_name)
