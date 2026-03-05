import sys
import os
import json

# 在导入 PyQt5 / pyqtgraph 之前，强制使用软件 OpenGL
os.environ.setdefault("QT_XCB_FORCE_SOFTWARE_OPENGL", "1")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
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
from utils.qt_utils import draw_arrow, draw_bbox, draw_bbox_solid, draw_arc_arrow, draw_arc_arrow_missing
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtWidgets import QLabel, QSizePolicy, QSlider, QMenuBar
from PyQt5.QtWidgets import QAction, QToolBar, QWidget, QPushButton, QColorDialog
from PyQt5.QtWidgets import QSplitter, QFrame, QMessageBox
from PyQt5.QtCore import QEvent, QSize

from utils.bbox_pick import pick_bbox_index
from widget.opengl_widget import PCDViewWidget
from widget.bbox_three_views import BboxThreeViewsPanel
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
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        # 主内容：左侧 3D+控制条，右侧为可显隐的三视图面板
        self.central_widget = QWidget()
        main_h = QHBoxLayout(self.central_widget)
        main_h.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        # 左侧容器：用于放 3D 视图和底部控制条（create_controls 里设置 layout）
        self.left_widget = QWidget()
        self.left_widget.setMinimumWidth(400)
        self.splitter.addWidget(self.left_widget)

        self.bbox_three_views_panel = BboxThreeViewsPanel(self)
        self.bbox_three_views_panel.hide()
        self.bbox_three_views_panel.close_btn.clicked.connect(self._on_three_view_closed)
        self.splitter.addWidget(self.bbox_three_views_panel)
        self.splitter.setCollapsible(1, True)
        # 右侧面板隐藏时把空间全给左侧（左侧给足够大，右侧 0）
        self.splitter.setSizes([9999, 0])

        main_h.addWidget(self.splitter)
        self.setCentralWidget(self.central_widget)

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
        self.current_bbox_infos = []  # 每个框的详细信息，用于点击弹窗
        self.current_link_arrows = []  # Semitrailer 指向 link_id 目标的弧线箭头
        self.selected_bbox_index = None  # 当前选中的框索引，用于实体框高亮

        self.right_button_pressed = False
        self.last_mouse_pos = None

    def create_controls(self):
        # 所有左侧控件父对象设为 left_widget，避免布局错乱和主视图空白
        self.play_button = QPushButton(self.left_widget)
        self.play_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/play_pcd.png')))
        self.play_button.setIconSize(self.play_button.sizeHint())
        self.play_button.setFlat(True)

        self.prev_button = QPushButton(self.left_widget)
        self.prev_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/prev_pcd.png')))
        self.prev_button.setIconSize(self.prev_button.sizeHint() * 0.8)
        self.prev_button.setFlat(True)

        self.next_button = QPushButton(self.left_widget)
        self.next_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/next.png')))
        self.next_button.setIconSize(self.next_button.sizeHint() * 0.8)
        self.next_button.setFlat(True)

        font = QFont("Arial", 10)
        self.frame_info_label = QLabel("Frame: 0 / 0", self.left_widget)
        self.frame_info_label.setFont(font)
        self.frame_info_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.frame_info_label.setMinimumHeight(int(font.pointSize() * 1.5))

        self.frame_slider = QSlider(Qt.Horizontal, self.left_widget)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.valueChanged.connect(self.on_slider_value_changed)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.prev_button)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.next_button)
        control_layout.addWidget(self.frame_info_label)
        control_layout.addWidget(self.frame_slider)

        # 将 3D 视图放入左侧容器，保证父子关系正确才能正常显示
        self.glwidget.setParent(self.left_widget)
        main_layout = QVBoxLayout(self.left_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.glwidget, 1)
        main_layout.addLayout(control_layout)

        self.play_button.clicked.connect(self.toggle_play_pause)
        self.prev_button.clicked.connect(self.previous_frame)
        self.next_button.clicked.connect(self.next_frame)

        # 点击目标框弹窗：在 3D 视图上安装鼠标事件过滤
        self.glwidget.installEventFilter(self)

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


        self.color_sidebar = QToolBar("colors", self)
        self.addToolBar(Qt.RightToolBarArea, self.color_sidebar)
        self.color_sidebar.setVisible(False)  # Initially hidden

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

    def eventFilter(self, obj, event):
        """鼠标在 3D 视图上：左键点击显示三视图，右键点击显示目标框信息"""
        if obj is self.glwidget and event.type() == QEvent.MouseButtonRelease and self.current_bbox_infos:
            try:
                ratio = self.glwidget.devicePixelRatioF()
                if ratio <= 0:
                    ratio = 1.0
                mx = event.pos().x() * ratio
                my = event.pos().y() * ratio
                idx = pick_bbox_index(self.glwidget, mx, my, self.current_bbox_infos)
                if idx is not None:
                    if event.button() == Qt.LeftButton:
                        self._show_bbox_info_dialog(idx)
                    elif event.button() == Qt.RightButton:
                        self._show_bbox_info_popup(idx)
            except Exception as e:
                print("bbox pick error:", e)
        return super().eventFilter(obj, event)

    def _show_bbox_info_popup(self, bbox_index):
        """右键单击时弹出目标框信息窗口"""
        info = self.current_bbox_infos[bbox_index]
        if bbox_index < 0 or bbox_index >= len(self.current_bbox_infos):
            return
        lines = []
        if info.get("class_name") is not None:
            lines.append("类别: {}".format(info["class_name"]))
        if info.get("id") is not None:
            lines.append("ID: {}".format(info["id"]))
        x, y, z = info.get("x"), info.get("y"), info.get("z")
        if x is not None and y is not None and z is not None:
            lines.append("中心点: ({:.3f}, {:.3f}, {:.3f})".format(float(x), float(y), float(z)))
        l, w, h = info.get("l"), info.get("w"), info.get("h")
        if l is not None and w is not None and h is not None:
            lines.append("长宽高: L={:.3f} W={:.3f} H={:.3f}".format(float(l), float(w), float(h)))
        yaw = info.get("yaw")
        if yaw is not None:
            lines.append("Yaw: {:.1f}°".format(np.rad2deg(float(yaw))))
        if "link_id" in info:
            lines.append("link_id: {}".format(info["link_id"] if info["link_id"] is not None else "-"))
        text = "\n".join(lines) if lines else "无信息"
        QMessageBox.information(self, "目标框信息", text)

    def _show_bbox_info_dialog(self, bbox_index):
        """仅更新右侧三视图面板，支持在三视图中拖动改框并同步主 3D 视图；选中框变为实体高亮"""
        self.selected_bbox_index = bbox_index
        self._refresh_bbox_selection_style()
        info = self.current_bbox_infos[bbox_index]
        if hasattr(self, "raw_points") and self.raw_points is not None and len(self.raw_points) > 0:
            xyz = self.raw_points[:, :3]
            self.bbox_three_views_panel.update_bbox(
                xyz, info,
                bbox_index=bbox_index,
                on_bbox_edited=self._on_bbox_edited_from_panel,
            )
            self.bbox_three_views_panel.show()
            self.bbox_three_views_panel.setMinimumWidth(320)
            total = self.splitter.width()
            self.splitter.setSizes([max(400, total - 380), 380])

    def _on_bbox_edited_from_panel(self, bbox_index, new_info):
        """三视图中拖动修改框后，同步更新 current_bbox_infos 与主 3D 视图中该框的显示"""
        if bbox_index < 0 or bbox_index >= len(self.current_bbox_infos):
            return
        self.current_bbox_infos[bbox_index] = {**self.current_bbox_infos[bbox_index], **new_info}
        self._refresh_single_bbox_in_main_view(bbox_index)

    def _refresh_bbox_selection_style(self):
        """根据 selected_bbox_index 刷新所有框的显示样式：选中为全包围半透明实体，未选中为线框"""
        if not self.current_bbox_items or not self.current_bbox_infos:
            return
        for i, info in enumerate(self.current_bbox_infos):
            base = i * 3
            if base + 2 >= len(self.current_bbox_items):
                continue
            x, y, z = info["x"], info["y"], info["z"]
            l, w, h = info["l"], info["w"], info["h"]
            yaw = info["yaw"]
            class_name = info.get("class_name", "")
            if class_name in self.class_map.keys():
                bbox_color = self.class_map[class_name]
            else:
                bbox_color = self.class_map["others"]
            is_selected = self.selected_bbox_index == i
            color = QColor(bbox_color[0], bbox_color[1], bbox_color[2], bbox_color[3])
            new_bbox = draw_bbox_solid(x, y, z, l, w, h, yaw, QColor(bbox_color[0], bbox_color[1], bbox_color[2], 80)) if is_selected else draw_bbox(x, y, z, l, w, h, yaw, color)
            old_bbox = self.current_bbox_items[base]
            self.glwidget.removeItem(old_bbox)
            self.current_bbox_items[base] = new_bbox
            self.glwidget.addItem(new_bbox)

    def _refresh_single_bbox_in_main_view(self, bbox_index):
        """仅刷新主 3D 视图中指定索引的 bbox（移除旧 3 项，用 current_bbox_infos 重绘并插入）"""
        if not self.current_bbox_items or bbox_index * 3 + 2 >= len(self.current_bbox_items):
            return
        info = self.current_bbox_infos[bbox_index]
        x, y, z = info["x"], info["y"], info["z"]
        l, w, h = info["l"], info["w"], info["h"]
        yaw = info["yaw"]
        class_name = info.get("class_name", "")
        if class_name in self.class_map.keys():
            bbox_color = self.class_map[class_name]
        else:
            bbox_color = self.class_map["others"]
        color = QColor(bbox_color[0], bbox_color[1], bbox_color[2], bbox_color[3])
        is_selected = self.selected_bbox_index == bbox_index
        new_bbox = draw_bbox_solid(x, y, z, l, w, h, yaw, QColor(bbox_color[0], bbox_color[1], bbox_color[2], 80)) if is_selected else draw_bbox(x, y, z, l, w, h, yaw, color)
        new_arrow = draw_arrow(np.array([x, y, z + h/2]), [np.cos(yaw), np.sin(yaw), 0], l/2, color)
        str_id = str(info.get("id", "")) if info.get("id") is not None else "XX"
        new_text = GLTextItem(text=class_name + "-" + str_id, pos=(x, y, z+1), color=color, font=QFont('Helvetica', 10))
        base = bbox_index * 3
        for i in range(3):
            self.glwidget.removeItem(self.current_bbox_items[base + i])
        self.current_bbox_items[base:base+3] = [new_bbox, new_arrow, new_text]
        self.glwidget.addItem(new_bbox)
        self.glwidget.addItem(new_arrow)
        self.glwidget.addItem(new_text)

    def _on_three_view_closed(self):
        """三视图面板关闭时清除选中高亮"""
        self.selected_bbox_index = None
        self._refresh_bbox_selection_style()

    def _set_topdown_view(self):
        """设置相机为俯视图"""
        dist = self.glwidget.opts.get("distance", 15)
        self.glwidget.setCameraPosition(distance=dist, elevation=90, azimuth=0)

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
            self._set_topdown_view()

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
            self._set_topdown_view()

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

        if elapsed1_time < 100:
            time.sleep((100-elapsed1_time)/1000)



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
        for item in getattr(self, "current_link_arrows", []):
            self.glwidget.removeItem(item)
        self.current_bbox_items = []
        self.current_bbox_infos = []
        self.current_link_arrows = []
        self.selected_bbox_index = None
        if self.bboxes_directory is not None:
            self.json_path = os.path.join(str(self.bboxes_directory), str(Path(self.pcd_file).stem)+".json")

            if os.path.isfile(self.json_path):
                json_data = get_anno_from_tanway_json(load_json(self.json_path))

                for i, box in enumerate(json_data["bboxes"]):
                    x, y, z, l, w, h, yaw = box

                    if "TYPE_" in json_data["className"][i]:
                        class_name = json_data["className"][i].replace("TYPE_", "")
                    else:
                        class_name = json_data["className"][i]
                    if "id" in json_data:
                        str_id = str(json_data["id"][i])
                    else:
                        str_id = "XX"


                    if class_name in self.class_map.keys():
                        bbox_color = self.class_map[class_name]
                    else:
                        bbox_color = self.class_map["others"]
                    color = QColor(bbox_color[0], bbox_color[1], bbox_color[2], bbox_color[3])
                    is_selected = self.selected_bbox_index == i
                    bbox = draw_bbox_solid(x, y, z, l, w, h, yaw, QColor(bbox_color[0], bbox_color[1], bbox_color[2], 80)) if is_selected else draw_bbox(x, y, z, l, w, h, yaw, color)

                    vis_text = GLTextItem(text=class_name + "-" + str_id, pos=(x, y, z+1), color=color, font=QFont('Helvetica', 10))
                    arrow = draw_arrow(np.array([x, y, z+h/2]), direction = [np.cos(yaw),np.sin(yaw),0],length= l/2 ,color = color)

                    self.glwidget.addItem(bbox)
                    self.glwidget.addItem(arrow)
                    self.glwidget.addItem(vis_text)
                    self.current_bbox_items.extend([bbox,arrow,vis_text])

                    # 保存该框信息，供点击拾取时弹窗显示
                    info = {"x": x, "y": y, "z": z, "l": l, "w": w, "h": h, "yaw": yaw, "class_name": class_name}
                    if "confidence" in json_data and i < len(json_data["confidence"]):
                        info["confidence"] = json_data["confidence"][i]
                    if "id" in json_data and i < len(json_data["id"]):
                        info["id"] = json_data["id"][i]
                    if "movement_state" in json_data and i < len(json_data["movement_state"]):
                        info["movement_state"] = json_data["movement_state"][i]
                    if "link_id" in json_data and i < len(json_data["link_id"]):
                        info["link_id"] = json_data["link_id"][i]
                    if "pitch" in json_data and i < len(json_data["pitch"]):
                        info["pitch"] = json_data["pitch"][i]
                    if "numPoints" in json_data and i < len(json_data["numPoints"]):
                        info["numPoints"] = json_data["numPoints"][i]
                    self.current_bbox_infos.append(info)

                # Semitrailer 与 link_id 目标之间的弧线箭头（颜色与框一致）
                id_to_center = {}
                for inf in self.current_bbox_infos:
                    bid = inf.get("id")
                    if bid is not None:
                        id_to_center[str(bid)] = (inf["x"], inf["y"], inf["z"]+inf["h"]/2)
                for inf in self.current_bbox_infos:
                    link_id = inf.get("link_id")
                    if link_id is None:
                        continue
                    class_name = inf.get("class_name", "")
                    if class_name in self.class_map.keys():
                        bbox_color = self.class_map[class_name]
                    else:
                        bbox_color = self.class_map["others"]
                    line_color = QColor(bbox_color[0], bbox_color[1], bbox_color[2], bbox_color[3])
                    target_ids = [link_id] if not isinstance(link_id, (list, tuple)) else list(link_id)
                    src = (inf["x"], inf["y"], inf["z"]+inf["h"]/2)
                    for tid in target_ids:
                        if tid is None:
                            continue
                        tgt = id_to_center.get(str(tid))
                        # tgt = (tgt[0], tgt[1], tgt[2]+inf["h"]/2)
                        if tgt is not None:
                            arc = draw_arc_arrow(src, tgt, line_color)
                            if arc is not None:
                                self.glwidget.addItem(arc)
                                self.current_link_arrows.append(arc)
                        else:
                            # 目标 ID 不存在：绘制向上弧线 + 缺失标签
                            arc = draw_arc_arrow_missing(src, QColor(220, 20, 60, 200) ) # 红色显示
                            if arc is not None:
                                self.glwidget.addItem(arc)
                                self.current_link_arrows.append(arc)
                            label_pos = (src[0], src[1], src[2] + 2.8)
                            missing_text = GLTextItem(
                                text="ID:{} 缺失".format(tid),
                                pos=label_pos,
                                color=QColor(220, 20, 60, 200),
                                font=QFont("Helvetica", 9),
                            )
                            self.glwidget.addItem(missing_text)
                            self.current_link_arrows.append(missing_text)

        # 切换帧后同步三视图：若三视图已打开且 bbox_index 仍有效，用新帧数据刷新
        if (self.bbox_three_views_panel.isVisible() and
                hasattr(self.bbox_three_views_panel, "_bbox_index") and
                self.bbox_three_views_panel._bbox_index is not None):
            idx = self.bbox_three_views_panel._bbox_index
            if idx < len(self.current_bbox_infos) and hasattr(self, "raw_points") and self.raw_points is not None:
                self.bbox_three_views_panel.update_bbox(
                    self.raw_points[:, :3],
                    self.current_bbox_infos[idx],
                    bbox_index=idx,
                    on_bbox_edited=self._on_bbox_edited_from_panel,
                )
            else:
                self.bbox_three_views_panel.hide()

        if self.scatter:
            self.glwidget.removeItem(self.scatter)

        self.points = self.raw_points[:, :3]
        if self.color_fields is not None:
            if max(self.structured_points[self.color_fields]) >= 0:
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


# 全局样式：偏网页化的简洁风格
GLOBAL_STYLESHEET = """
    QMainWindow {
        background-color: #f0f2f5;
    }
    QWidget {
        background-color: #f0f2f5;
        color: #1c1e21;
    }
    QMenuBar {
        background-color: #ffffff;
        color: #1c1e21;
        padding: 6px 8px;
        border-bottom: 1px solid #e4e6eb;
        font-size: 13px;
    }
    QMenuBar::item:selected {
        background-color: #e4e6eb;
        border-radius: 4px;
    }
    QMenu {
        background-color: #ffffff;
        border: 1px solid #e4e6eb;
        border-radius: 8px;
        padding: 6px 0;
    }
    QMenu::item:selected {
        background-color: #e4e6eb;
    }
    QToolBar {
        background-color: #ffffff;
        border: none;
        border-bottom: 1px solid #e4e6eb;
        spacing: 6px;
        padding: 8px 12px;
    }
    QToolBar QToolButton {
        background-color: transparent;
        border: none;
        border-radius: 6px;
        padding: 6px 10px;
    }
    QToolBar QToolButton:hover {
        background-color: #e4e6eb;
    }
    QPushButton {
        background-color: #e4e6eb;
        color: #1c1e21;
        border: none;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 13px;
    }
    QPushButton:hover {
        background-color: #d8dadf;
    }
    QPushButton:pressed {
        background-color: #ccced2;
    }
    QSlider::groove:horizontal {
        border: none;
        height: 6px;
        background: #e4e6eb;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #ffffff;
        width: 16px;
        margin: -5px 0;
        border-radius: 8px;
        border: 1px solid #ccced2;
    }
    QSlider::handle:horizontal:hover {
        background: #f0f2f5;
    }
    QLabel {
        color: #1c1e21;
        font-size: 13px;
    }
    QScrollArea {
        border: none;
        background-color: transparent;
    }
    QScrollBar:vertical {
        background: #e4e6eb;
        width: 10px;
        border-radius: 5px;
        margin: 0;
    }
    QScrollBar::handle:vertical {
        background: #bcc0c4;
        border-radius: 5px;
        min-height: 24px;
    }
    QScrollBar::handle:vertical:hover {
        background: #8a8d91;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }
    QSplitter::handle {
        background: #e4e6eb;
        width: 2px;
    }
"""

if __name__ == "__main__":
    # 在创建 QApplication 之前设置软件 OpenGL 属性
    QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)

    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET)
    viewer = PointCloudViewer()
    viewer.show()
    sys.exit(app.exec_())