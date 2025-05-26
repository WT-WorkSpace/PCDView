import sys
import os
import json
import numpy as np
from pathlib import Path
from natsort import natsorted
import matplotlib.pyplot as plt
import pyqtgraph.opengl as gl
from pyqtgraph import Vector
from pyqtgraph.opengl import GLScatterPlotItem, GLTextItem, GLScatterPlotItem
from PyQt5.QtGui import QFont, QColor, QIcon, QQuaternion, QMouseEvent
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QTimer, Qt
from utils.qt_utils import draw_arrow, draw_bbox
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtWidgets import QLabel, QSizePolicy, QSlider, QMenuBar
from PyQt5.QtWidgets import QAction, QToolBar, QWidget, QPushButton, QColorDialog

from widget.opengl_widget import PCDViewWidget
from utils.utils import pil2qicon
from utils.load_pcd import get_points_from_pcd_file
from utils.load_bboxes_json import get_anno_from_tanway_json

from utils.utils import load_json

class PointCloudViewer(QMainWindow, PCDViewWidget):
    def __init__(self):
        QMainWindow.__init__(self)
        PCDViewWidget.__init__(self)
        self.init_ui()
        self.create_menus()
        self.create_toolbar()
        self.create_controls()
        self.init_state()

    def init_ui(self):
        self.curpath = os.path.dirname(os.path.abspath(__file__))
        self.setWindowTitle("Point Cloud Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

    def init_state(self):
        self.point_cloud_files = []
        self.current_frame_index = -1
        self.scatter_item = GLScatterPlotItem()
        self.playing = False  # Flag for play/pause state
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)  # Timer action for auto-frame transition
        self.colors = QColor(0, 0, 255).getRgbF()
        self.color_fields = None
        self.metadata = None
        self.current_bbox_items = []

        self.right_button_pressed = False
        self.last_mouse_pos = None

    def create_controls(self):
        self.play_button = QPushButton(self)
        self.play_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/play_pcd.png')))
        self.play_button.setIconSize(self.play_button.sizeHint())
        self.play_button.setFlat(True)

        self.prev_button = QPushButton(self)
        self.prev_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/prev_pcd.png')))
        self.prev_button.setIconSize(self.prev_button.sizeHint() * 0.8)
        self.prev_button.setFlat(True)

        self.next_button = QPushButton(self)
        self.next_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/next.png')))
        self.next_button.setIconSize(self.next_button.sizeHint() * 0.8)
        self.next_button.setFlat(True)

        font = QFont("Arial", 10)
        self.frame_info_label = QLabel("Frame: 0 / 0", self)
        self.frame_info_label.setFont(font)
        self.frame_info_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.frame_info_label.setMinimumHeight(int(font.pointSize() * 1.5))

        # Create a slider for selecting frames
        self.frame_slider = QSlider(Qt.Horizontal, self)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.valueChanged.connect(self.on_slider_value_changed)

        # Create a horizontal layout for the controls
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.prev_button)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.next_button)
        control_layout.addWidget(self.frame_info_label)
        control_layout.addWidget(self.frame_slider)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.glwidget)
        main_layout.addLayout(control_layout)  # Add control buttons and slider in the same row
        self.central_widget.setLayout(main_layout)

        self.play_button.clicked.connect(self.toggle_play_pause)
        self.prev_button.clicked.connect(self.previous_frame)
        self.next_button.clicked.connect(self.next_frame)

    def create_toolbar(self):
        self.toolbar = QToolBar(self)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)
        self.toolbar.addAction(self.open_file_action)
        self.toolbar.addAction(self.open_dir_action)
        self.toolbar.addAction(self.open_bboxes_dir_action)
        self.toolbar.addSeparator()

        self.toolbar.addAction(self.increase_pointsize_action)
        self.toolbar.addAction(self.decrease_pointsize_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.points_color)  # Add color button to toolbar
        self.toolbar.addAction(self.coordinate)

        self.toolbar.addAction(self.save_view_action)
        self.toolbar.addAction(self.load_view_action)

        self.toolbar.setStyleSheet("QToolBar { background-color: white; }")

        self.color_sidebar = QToolBar("colors", self)
        self.addToolBar(Qt.RightToolBarArea, self.color_sidebar)
        self.color_sidebar.setVisible(False)  # Initially hidden
        self.color_sidebar.setStyleSheet("QToolBar { background-color: white; }")

    def create_menus(self):
        self.create_file_menus()
        self.create_tools_menus()

    def create_file_menus(self):
        file_menu = self.menu_bar.addMenu("File")
        self.open_file_action = self.create_action("Open File", 'icons/open.png', self.open_file)
        self.open_dir_action = self.create_action("Open Directory", 'icons/open_dir.png', self.open_directory)
        self.open_bboxes_dir_action = self.create_action("Open BBoxes Dir", 'icons/open_boxes_dir.svg',self.open_bboxes_directory)
        file_menu.addAction(self.open_file_action)
        file_menu.addAction(self.open_dir_action)
        file_menu.addAction(self.open_bboxes_dir_action)

    def create_tools_menus(self):
        tool_menu = self.menu_bar.addMenu("Tools")
        tool_pointsize_menu = tool_menu.addMenu("Point Size")
        self.increase_pointsize_action = self.create_action("Point Size +", 'icons/pointsize_increase.png', self.increase_points_size)
        self.decrease_pointsize_action = self.create_action("Point Size -", 'icons/pointsize_decrease.png', self.decrease_points_size)
        self.points_color = self.create_action("Color", 'icons/color.png', self.select_color)
        self.coordinate = self.create_action("Coordinate", 'icons/coordinate.png', self.create_coordinate)
        self.save_view_action = self.create_action("Save View", 'icons/save_view.png', self.save_view)
        self.load_view_action = self.create_action("Load View", 'icons/load_view.svg', self.load_view)

        tool_pointsize_menu.addAction(self.increase_pointsize_action)
        tool_pointsize_menu.addAction(self.decrease_pointsize_action)
        tool_menu.addAction(self.points_color)
        tool_menu.addAction(self.coordinate)
        tool_menu.addAction(self.save_view_action)
        tool_menu.addAction(self.load_view_action)

    def create_action(self, name, icon_path, handler):
        icon = QIcon(os.path.join(self.curpath, icon_path))
        action = QAction(icon, name, self)
        action.triggered.connect(handler)
        return action

    def open_directory(self):
        self.timer.stop()  # Stop the timer to avoid auto-frame transition
        self.playing = False
        self.play_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/play_pcd.png')))

        self.directory = QFileDialog.getExistingDirectory(self, "Select Point Cloud Directory")
        if self.directory:
            self.point_cloud_files = natsorted([
                f for f in os.listdir(self.directory) if f.endswith('.txt') or f.endswith('.pcd')
            ])
            self.current_frame_index = 0
            self.frame_slider.setMaximum(len(self.point_cloud_files) - 1)
            self.bboxes_files = None
            self.load_frame()

    def open_bboxes_directory(self):
        self.timer.stop()  # Stop the timer to avoid auto-frame transition
        self.playing = False
        self.play_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/play_pcd.png')))

        self.bboxes_directory = QFileDialog.getExistingDirectory(self, "Select bboxes json Directory")

        if self.bboxes_directory:
            self.bboxes_files = natsorted([f for f in os.listdir(self.bboxes_directory) if f.endswith('.json')])
            self.load_frame()

    def open_file(self):
        self.timer.stop()
        self.playing = False
        self.play_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/play_pcd.png')))
        self.pcd_file, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Pcd Files (*.pcd)", options=QFileDialog.Options())
        self.colors = QColor(0, 0, 255).getRgbF()
        if self.pcd_file:
            self.directory = None
            self.point_cloud_files = []
            self.current_frame_index = -1
            self.raw_points, self.structured_points, metadata = get_points_from_pcd_file(self.pcd_file)
            self.metadata = metadata
            self.vis_fram(updata_color_bar=True)

    def increase_points_size(self):
        self.point_size = self.point_size + 1
        self.vis_fram()

    def decrease_points_size(self):
        if self.point_size <= 1:
            self.point_size = self.point_size - 0.3
        else:
            self.point_size = self.point_size - 1
        if self.point_size <= 0:
            self.point_size = 0
        self.vis_fram()

    def load_frame(self):
        import time
        start_time = time.time()

        assert self.current_frame_index >= 0 and self.current_frame_index < len(self.point_cloud_files)
        self.pcd_file = os.path.join(self.directory, self.point_cloud_files[self.current_frame_index])
        self.raw_points, self.structured_points, metadata = get_points_from_pcd_file(self.pcd_file)
        if self.color_fields is not None:
            print(self.color_fields)
            self.colors = self.Colors[0](self.min_max_normalization(self.structured_points[self.color_fields]))

        if metadata != self.metadata:
            self.metadata = metadata
            self.vis_fram(updata_color_bar=True)
        else:
            self.vis_fram(updata_color_bar=False)
        end1_time = time.time()
        self.frame_info_label.setText(
            f"Frame: {self.current_frame_index + 1} / {len(self.point_cloud_files)} ({self.point_cloud_files[self.current_frame_index]})")
        self.frame_slider.setValue(self.current_frame_index)

        elapsed1_time = (end1_time - start_time) * 1000  # Calculate time in milliseconds
        print(f"Code execution time: {elapsed1_time:.3f} ms", id)

    def min_max_normalization(self, matrix):
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        return normalized_matrix

    def previous_frame(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.load_frame()

    def next_frame(self):
        if self.current_frame_index < len(self.point_cloud_files) - 1:
            self.current_frame_index += 1
            self.frame_slider.blockSignals(True)  # Block slider signals
            self.frame_slider.setValue(self.current_frame_index)  # Update slider position
            self.frame_slider.blockSignals(False)  # Re-enable signals
            self.load_frame()
        else:
            self.timer.stop()
            self.playing = False
            self.play_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/play_pcd.png')))

    def toggle_play_pause(self):
        if self.playing:
            self.timer.stop()
            self.play_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/play_pcd.png')))
        else:
            self.timer.start(100)
            self.play_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/pause_pcd.png')))
        self.playing = not self.playing

    def on_slider_value_changed(self):
        print("--on_slider_value_changed")
        if self.current_frame_index != self.frame_slider.value() and self.directory is not None:
            self.current_frame_index = self.frame_slider.value()
            self.load_frame()
        else:
            print("single")

    def select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.colors = color.getRgbF()
            self.color_fields = None
            self.vis_fram()

    def update_color_sidebar(self):
        self.color_sidebar.clear()
        for meta in self.metadata:
            action = QAction(pil2qicon(meta[0]), meta, self)
            action.setCheckable(False)
            action.triggered.connect(lambda checked, idx=meta: self.select_dimension(idx))
            self.color_sidebar.addAction(action)
        if not self.color_sidebar.isVisible():
            self.color_sidebar.setVisible(True)

    def select_dimension(self, meta):
        print("select_dimension")
        self.Colors = [plt.get_cmap('gist_ncar'), plt.get_cmap('cool'), plt.get_cmap('GnBu'), plt.get_cmap('Greys'), plt.get_cmap('hot')]  # Reference: https://zhuanlan.zhihu.com/p/114420786
        self.color_fields = meta
        self.vis_fram()

    def save_view(self):
        view_data_ = self.glwidget.cameraParams()
        view_data = {
            "center": [view_data_["center"].x(),view_data_["center"].y(),view_data_["center"].z()],
            "distance": view_data_["distance"],
            "rotation": [
                view_data_["rotation"].scalar(),  # 四元数标量
                view_data_["rotation"].x(),
                view_data_["rotation"].y(),
                view_data_["rotation"].z()
            ],
            "fov": view_data_["fov"],
            "elevation": view_data_["elevation"],
            "azimuth": view_data_["azimuth"],
        }
        file_name, _ = QFileDialog.getSaveFileName(self, "Save View", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'w') as f:
                json.dump(view_data, f, indent=4)
            print("View saved to:", file_name)

    def load_view(self):
        """加载保存的视角参数"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open View", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'r') as f:
                view_data = json.load(f)
            rotation = QQuaternion(
                view_data["rotation"][0],  # 标量
                view_data["rotation"][1],
                view_data["rotation"][2],
                view_data["rotation"][3]
            )
            view_data_ = {}
            view_data_["center"] = Vector(*view_data["center"])
            view_data_["distance"] = view_data["distance"]
            view_data_["rotation"] = rotation
            view_data_["fov"] = view_data["fov"]
            view_data_["elevation"] = view_data['elevation']
            view_data_["azimuth"] = view_data["azimuth"]

            self.glwidget.setCameraPosition(
                pos=view_data_["center"],
                distance=view_data_["distance"],
                elevation=view_data_["elevation"],
                azimuth=view_data_["azimuth"]
            )
            self.vis_fram()

    def vis_fram(self, updata_color_bar=False):
        for item in self.current_bbox_items:
            self.glwidget.removeItem(item)
        self.current_bbox_items = []
        if self.bboxes_directory is not None:
            self.json_path = os.path.join(str(self.bboxes_directory), str(Path(self.pcd_file).stem)+".json")

            if os.path.isfile(self.json_path):
                json_data = get_anno_from_tanway_json(load_json(self.json_path))

                for i, box in enumerate(json_data["bboxes"]):
                    x, y, z, l, w, h, yaw = box

                    class_name = json_data["className"][i].replace("TYPE_", "")
                    if class_name in self.class_map.keys():
                        bbox_color = self.class_map[class_name]
                    else:
                        bbox_color = self.class_map["others"]
                    color = QColor(bbox_color[0], bbox_color[1], bbox_color[2], bbox_color[3])

                    bbox = draw_bbox(x, y, z, l, w, h, yaw, color)
                    class_name_text = GLTextItem(text=class_name, pos=(x, y, z+1), color=color, font=QFont('Helvetica', 10))
                    arrow = draw_arrow(np.array([x, y, z+h/2]), direction = [np.cos(yaw),np.sin(yaw),0],length= l/2 ,color = color)

                    self.glwidget.addItem(bbox)
                    self.glwidget.addItem(arrow)
                    self.glwidget.addItem(class_name_text)
                    self.current_bbox_items.extend([bbox,arrow,class_name_text])

        if self.scatter:
            self.glwidget.removeItem(self.scatter)

        self.points = self.raw_points[:, :3]
        if self.color_fields is not None:
            if max(self.structured_points[self.color_fields]) != 0:
                unique_values = np.unique(self.structured_points[self.color_fields])
                num_unique_values = len(unique_values)
                print(unique_values)
                print(type(unique_values[0]))
                if all(isinstance(x, np.int32) for x in unique_values)  and max(unique_values) < 16 and min(unique_values)>=0:
                    color_map = {}
                    for i, value in enumerate(unique_values):
                        color_map[value] =self.colors_16[value]
                    self.colors = np.array([color_map[val] for val in self.structured_points[self.color_fields]])

                elif num_unique_values <= 16:
                    color_map = {}
                    for i, value in enumerate(unique_values):
                        color_map[value] =self.colors_16[i]
                    self.colors = np.array([color_map[val] for val in self.structured_points[self.color_fields]])
                else:
                    self.colors = self.Colors[0](self.min_max_normalization(self.structured_points[self.color_fields]))
                    
        self.scatter = GLScatterPlotItem(pos=self.points, color=self.colors, size=self.point_size)
        self.glwidget.addItem(self.scatter)
        if updata_color_bar:
            self.update_color_sidebar()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = PointCloudViewer()
    viewer.show()
    sys.exit(app.exec_())