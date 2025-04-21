import sys
import os
from natsort import natsorted
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QLabel, QSizePolicy, QSlider, QMenuBar,
    QAction, QToolBar, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QColorDialog,
)
import matplotlib.pyplot as plt
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor, QIcon
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLScatterPlotItem
from widget.opengl_widget import PCDViewWidget
from utils.utils import pil2qicon
from utils.load_pcd import get_points_from_pcd_file
import numpy as np


class PointCloudViewer(QMainWindow, PCDViewWidget):
    def __init__(self):
        QMainWindow.__init__(self)
        PCDViewWidget.__init__(self)
        self.curpath = os.path.dirname(os.path.abspath(__file__))
        self.setWindowTitle("Point Cloud Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a menu bar
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        file_menu = self.menu_bar.addMenu("File")
        tool_menu = self.menu_bar.addMenu("Tools")
        tool_pointsize_menu = tool_menu.addMenu("Point Size")

        # Create "Open File" action
        open_file_icon = QIcon(QIcon(os.path.join(self.curpath, 'icons/open.png')).pixmap(25, 25))
        self.open_file_action = QAction(open_file_icon, "Open File", self)
        self.open_file_action.triggered.connect(self.open_file)
        file_menu.addAction(self.open_file_action)

        # Create "Open Directory" action
        open_dir_icon = QIcon(QIcon(os.path.join(self.curpath, 'icons/open_dir.png')).pixmap(25, 25))
        self.open_dir_action = QAction(open_dir_icon, "Open Directory", self)
        self.open_dir_action.triggered.connect(self.open_directory)
        file_menu.addAction(self.open_dir_action)

        self.set_pointsize = QAction(QIcon(os.path.join(self.curpath, 'icons/pointsize.png')), "Point Size", self)
        self.set_pointsize.triggered.connect(self.open_file)

        pointsize_increase_icon = QIcon(QIcon(os.path.join(self.curpath, 'icons/pointsize_increase.png')).pixmap(25, 25))
        self.increase_pointsize_action = QAction(pointsize_increase_icon, "Point Size +", self)
        self.increase_pointsize_action.triggered.connect(self.increase_points_size)
        tool_pointsize_menu.addAction(self.increase_pointsize_action)

        pointsize_decrease_icon = QIcon(QIcon(os.path.join(self.curpath, 'icons/pointsize_decrease.png')).pixmap(25, 25))
        self.decrease_pointsize_action = QAction(pointsize_decrease_icon, "Point Size -", self)
        self.decrease_pointsize_action.triggered.connect(self.decrease_points_size)
        tool_pointsize_menu.addAction(self.decrease_pointsize_action)

        # Add "Color" action
        point_color_icon = QIcon(QIcon(os.path.join(self.curpath, 'icons/color.png')).pixmap(25, 25))
        self.points_color = QAction(point_color_icon, "Color", self)
        self.points_color.triggered.connect(self.select_color)  # Connect to color selection
        tool_pointsize_menu.addAction(self.points_color)

        # Add "Coordinate axis" action
        coordinate_icon = QIcon(QIcon(os.path.join(self.curpath, 'icons/coordinate.png')).pixmap(25, 25))
        self.coordinate = QAction(coordinate_icon, "Coordinate", self)
        self.coordinate.triggered.connect(self.create_coordinate)  # Connect to color selection
        tool_pointsize_menu.addAction(self.coordinate)

        # Create a toolbar for quick access to file actions and add it using addToolBar
        self.toolbar = QToolBar(self)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)
        self.toolbar.addAction(self.open_file_action)
        self.toolbar.addAction(self.open_dir_action)
        self.toolbar.addSeparator()

        self.toolbar.addAction(self.increase_pointsize_action)
        self.toolbar.addAction(self.decrease_pointsize_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.points_color)  # Add color button to toolbar
        self.toolbar.addAction(self.coordinate)
        self.toolbar.setStyleSheet("QToolBar { background-color: white; }")



        self.color_sidebar = QToolBar("colors", self)
        self.addToolBar(Qt.RightToolBarArea, self.color_sidebar)
        self.color_sidebar.setVisible(False)  # Initially hidden
        self.color_sidebar.setStyleSheet("QToolBar { background-color: white; }")

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

        self.play_button.clicked.connect(self.toggle_play_pause)
        self.prev_button.clicked.connect(self.previous_frame)
        self.next_button.clicked.connect(self.next_frame)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.glwidget)
        main_layout.addLayout(control_layout)  # Add control buttons and slider in the same row
        central_widget.setLayout(main_layout)

        self.point_cloud_files = []
        self.current_frame_index = -1
        self.scatter_item = GLScatterPlotItem()
        self.playing = False  # Flag for play/pause state
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)  # Timer action for auto-frame transition

        self.colors = QColor(0, 0, 255).getRgbF()
        self.color_fields = None
        self.metadata = None

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
        pcd_path = os.path.join(self.directory, self.point_cloud_files[self.current_frame_index])
        self.raw_points, self.structured_points, metadata = get_points_from_pcd_file(pcd_path)
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

    def vis_fram(self, updata_color_bar=False):
        if self.scatter:
            self.glwidget.removeItem(self.scatter)

        self.points = self.raw_points[:, :3]
        # Apply the selected color

        if self.color_fields is not None:
            if max(self.structured_points[self.color_fields]) != 0:
                unique_values = np.unique(self.structured_points[self.color_fields])
                num_unique_values = len(unique_values)

                if num_unique_values <= 16:
                    color_map = {}
                    for i, value in enumerate(unique_values):
                        color_map[value] =self.colors_16[i]

                    self.colors = np.array([color_map[val] for val in self.structured_points[self.color_fields]])

                    # 为每个点分配颜色
                    self.colors = np.array([color_map[val] for val in self.structured_points[self.color_fields]])
                else:
                    # 使用原来的方法
                    self.colors = self.Colors[0](self.min_max_normalization(self.structured_points[self.color_fields]))
                # print(self.colors)
        self.scatter = gl.GLScatterPlotItem(pos=self.points, color=self.colors, size=self.point_size)
        self.glwidget.addItem(self.scatter)
        if updata_color_bar:
            self.update_color_sidebar()

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
        self.Colors = [plt.get_cmap('gist_ncar'), plt.get_cmap('cool'), plt.get_cmap('GnBu'), plt.get_cmap('Greys'),
                       plt.get_cmap('hot')]  # Reference: https://zhuanlan.zhihu.com/p/114420786
        self.color_fields = meta
        self.vis_fram()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = PointCloudViewer()
    viewer.show()
    sys.exit(app.exec_())
