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
from PyQt5.QtWidgets import QSplitter, QFrame, QMessageBox, QShortcut, QDockWidget
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QAbstractItemView
from PyQt5.QtWidgets import QHeaderView
from PyQt5.QtCore import QEvent, QSize
from PyQt5.QtGui import QKeySequence, QPainter, QPen

from utils.bbox_pick import (
    pick_bbox_index, points_in_screen_rect, fit_obb_xy,
    filter_ground_points, ray_from_screen, ray_plane_z_intersect,
)
from widget.opengl_widget import PCDViewWidget
from widget.bbox_three_views import BboxThreeViewsPanel
from utils.utils import pil2qicon
from utils.load_pcd import get_points_from_pcd_file
from utils.load_bboxes_json import get_anno_from_tanway_json, save_bboxes_to_tanway_json

from utils.utils import load_json


class BoxSelectOverlay(QWidget):
    """框选时在 glwidget 上绘制拖拽矩形的透明覆盖层"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._start = None
        self._end = None

    def set_rect(self, start, end):
        self._start = start
        self._end = end
        self.update()

    def clear_rect(self):
        self._start = None
        self._end = None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._start is None or self._end is None:
            return
        painter = QPainter(self)
        painter.setPen(QPen(QColor(0, 120, 255), 2, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        x1, y1 = self._start
        x2, y2 = self._end
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        painter.drawRect(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        painter.end()


LIST_POINT_SELECT_CAP = 8000  # 列表展示上限，避免一次框选过多点时界面卡死


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
        # 纯色模式下用于每次 vis_fram 重建颜色；避免框选红色写入 self.colors 后无法恢复
        self._user_solid_rgbf = QColor(0, 0, 255).getRgbF()
        self.color_fields = None
        self.metadata = None
        self.current_bbox_items = []
        self.current_bbox_infos = []  # 每个框的详细信息，用于点击弹窗
        self.current_link_arrows = []  # Semitrailer 指向 link_id 目标的弧线箭头
        self.selected_bbox_index = None  # 当前选中的框索引，用于实体框高亮
        self.box_select_mode = False  # 框选模式：拖拽生成新矩形框
        self.box_select_start = None  # 框选起始屏幕坐标 (x, y)，设备像素
        self.box_select_start_logical = None  # 框选起始逻辑坐标，用于 overlay 绘制
        self.bbox_modified = False  # 框是否被修改过，用于显示 Save 按钮
        self.original_json_agents = None  # 原始 JSON agent 列表，保存时用于保留额外字段

        self.points_rect_select_mode = False  # 点云框选：拖拽矩形，选中点标红并列表展示
        self._points_rect_select_mask = None  # 与当前点云等长的 bool 掩码，或 None
        self._point_select_dock = None

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
        # 框选拖拽矩形预览覆盖层
        self.box_select_overlay = BoxSelectOverlay(self.glwidget)
        self.box_select_overlay.setGeometry(0, 0, self.glwidget.width(), self.glwidget.height())
        self.box_select_overlay.raise_()
        self.box_select_overlay.show()
        # Backspace 删除选中的目标框
        self.delete_shortcut = QShortcut(QKeySequence(Qt.Key_Backspace), self)
        self.delete_shortcut.activated.connect(self._delete_selected_bbox)
        # 主视图右上角 Save 按钮（修改框后显示）
        self.save_bboxes_btn = QPushButton("Save", self.glwidget)
        self.save_bboxes_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; padding: 6px 12px; "
            "border-radius: 4px; font-weight: bold; } "
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self.save_bboxes_btn.clicked.connect(self._save_bboxes_clicked)
        self.save_bboxes_btn.hide()

    def _update_save_button_geometry(self):
        """将 Save 按钮置于主视图右上角"""
        if self.glwidget.width() <= 0 or self.glwidget.height() <= 0:
            return
        m = 12
        bw = max(self.save_bboxes_btn.sizeHint().width(), 70)
        bh = max(self.save_bboxes_btn.sizeHint().height(), 28)
        self.save_bboxes_btn.setGeometry(
            self.glwidget.width() - bw - m, m, bw, bh
        )
        self.save_bboxes_btn.raise_()

    def _show_save_button_if_modified(self):
        """若框已修改且有可保存的 JSON 路径，则显示 Save 按钮"""
        if self.bbox_modified and hasattr(self, "json_path") and self.json_path and hasattr(self, "bboxes_directory") and self.bboxes_directory:
            self._update_save_button_geometry()
            self.save_bboxes_btn.show()
            self.save_bboxes_btn.raise_()

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
        self.toolbar.addSeparator()
        self.box_select_action = self.create_action("标注3D框", "icons/add_bbox.svg", self._toggle_box_select_mode)
        self.box_select_action.setCheckable(True)
        self.toolbar.addAction(self.box_select_action)
        self.points_rect_select_action = self.create_action(
            "点云框选", "icons/box_selection.svg", self._toggle_points_rect_select_mode
        )
        self.points_rect_select_action.setCheckable(True)
        self.toolbar.addAction(self.points_rect_select_action)
        self.cancel_points_rect_select_action = self.create_action(
            "取消框选",
            "icons/cancel_box_selection.svg",
            self._clear_point_rect_selection,
        )
        self.cancel_points_rect_select_action.setCheckable(False)
        self.toolbar.addAction(self.cancel_points_rect_select_action)

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
        self.points_color = self.create_action("Color", 'icons/color.svg', self.select_color)
        self.coordinate = self.create_action("Coordinate", 'icons/coordinate.svg', self.create_coordinate)
        self.save_view_action = self.create_action("Save View", 'icons/save_view.svg', self.save_view)
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
        """鼠标在 3D 视图上：框选模式拖拽生成框；否则左键点击显示三视图，右键显示目标框信息"""
        if obj != self.glwidget:
            return super().eventFilter(obj, event)
        if event.type() == QEvent.Resize:
            self.box_select_overlay.setGeometry(0, 0, self.glwidget.width(), self.glwidget.height())
            self._update_save_button_geometry()

        # 只有鼠标相关事件才有 pos()/button() 等信息；否则（例如 QPaintEvent/QHideEvent）
        # 会导致 event.pos() AttributeError。
        mouse_types = {QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseMove}
        if event.type() not in mouse_types:
            return super().eventFilter(obj, event)
        if not hasattr(event, "pos"):
            return super().eventFilter(obj, event)

        ratio = self.glwidget.devicePixelRatioF()
        if ratio <= 0:
            ratio = 1.0
        mx = event.pos().x() * ratio
        my = event.pos().y() * ratio

        if self.box_select_mode:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.box_select_start = (mx, my)
                self.box_select_start_logical = (event.pos().x(), event.pos().y())
                self.box_select_overlay.set_rect(self.box_select_start_logical, self.box_select_start_logical)
                return True  # 消费事件，阻止 glwidget 旋转
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self.box_select_start:
                dx = abs(mx - self.box_select_start[0])
                dy = abs(my - self.box_select_start[1])
                if dx > 5 and dy > 5:
                    self._add_bbox_from_rect(self.box_select_start[0], self.box_select_start[1], mx, my)
                self.box_select_start = None
                self.box_select_start_logical = None
                self.box_select_overlay.clear_rect()
                return True  # 消费事件
            if event.type() == QEvent.MouseMove and self.box_select_start is not None:
                end_logical = (event.pos().x(), event.pos().y())
                self.box_select_overlay.set_rect(self.box_select_start_logical, end_logical)
                return True  # 拖拽过程中拦截 move，防止 glwidget 旋转
            return super().eventFilter(obj, event)

        if self.points_rect_select_mode:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.box_select_start = (mx, my)
                self.box_select_start_logical = (event.pos().x(), event.pos().y())
                self.box_select_overlay.set_rect(self.box_select_start_logical, self.box_select_start_logical)
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self.box_select_start:
                dx = abs(mx - self.box_select_start[0])
                dy = abs(my - self.box_select_start[1])
                if dx > 5 and dy > 5:
                    self._apply_points_in_rect_from_drag(self.box_select_start[0], self.box_select_start[1], mx, my)
                else:
                    self.frame_info_label.setText("拖拽矩形过小，请点击「点云框选」再试一次")
                self.box_select_start = None
                self.box_select_start_logical = None
                self.box_select_overlay.clear_rect()
                self._finish_points_rect_one_shot()
                return True
            if event.type() == QEvent.MouseMove and self.box_select_start is not None:
                end_logical = (event.pos().x(), event.pos().y())
                self.box_select_overlay.set_rect(self.box_select_start_logical, end_logical)
                return True
            return super().eventFilter(obj, event)

        if event.type() == QEvent.MouseButtonRelease and self.current_bbox_infos:
            try:
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
        self._rebuild_link_arrows()  # 实时更新 link_id 弧线（中心点或尺寸变化会影响弧线起止点）
        self.bbox_modified = True
        self._show_save_button_if_modified()

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

    def _delete_selected_bbox(self):
        """Backspace 删除当前选中的目标框"""
        if self.selected_bbox_index is None or not self.current_bbox_infos:
            return
        idx = self.selected_bbox_index
        if idx < 0 or idx >= len(self.current_bbox_infos):
            return
        base = idx * 3
        if base + 2 >= len(self.current_bbox_items):
            return
        for i in range(3):
            self.glwidget.removeItem(self.current_bbox_items[base + i])
        del self.current_bbox_items[base:base + 3]
        del self.current_bbox_infos[idx]
        self.selected_bbox_index = None
        if self.bbox_three_views_panel.isVisible() and getattr(self.bbox_three_views_panel, "_bbox_index", None) == idx:
            self.bbox_three_views_panel.hide()
        elif getattr(self.bbox_three_views_panel, "_bbox_index", None) is not None and self.bbox_three_views_panel._bbox_index > idx:
            self.bbox_three_views_panel._bbox_index -= 1
        self._rebuild_link_arrows()
        self._update_frame_info_label()
        self.bbox_modified = True
        self._show_save_button_if_modified()

    def _toggle_box_select_mode(self):
        """切换框选模式：开启后拖拽可生成新矩形框"""
        self.box_select_mode = not self.box_select_mode
        self.box_select_action.setChecked(self.box_select_mode)
        if self.box_select_mode:
            self.points_rect_select_mode = False
            if hasattr(self, "points_rect_select_action"):
                self.points_rect_select_action.setChecked(False)
            self.frame_info_label.setText("框选模式：在视图中拖拽绘制矩形区域")
        else:
            self.box_select_start = None
            self.box_select_start_logical = None
            self.box_select_overlay.clear_rect()
            self._update_frame_info_label()

    def _toggle_points_rect_select_mode(self):
        """点云框选：每次点击工具栏仅允许拖拽一次矩形；再次框选需重新点击。"""
        self.points_rect_select_mode = not self.points_rect_select_mode
        self.points_rect_select_action.setChecked(self.points_rect_select_mode)
        if self.points_rect_select_mode:
            # 开启下一轮框选：恢复之前高亮的异色，但不取消左侧弹框（只在取消框选时才隐藏）
            self._restore_points_color_only()
            self.box_select_mode = False
            self.box_select_action.setChecked(False)
            self.point_select_info_label.setText(
                "已就绪：在主视图中按住左键拖拽一次矩形（仅一次机会）。再次框选请先再点工具栏「点云框选」。"
            )
            self.frame_info_label.setText("点云框选：请拖拽一次矩形（仅一次机会）")
            self._update_points_rect_button_style(True)
        else:
            self.box_select_start = None
            self.box_select_start_logical = None
            self.box_select_overlay.clear_rect()
            self._update_frame_info_label()
            self._update_points_rect_button_style(False)

    def _finish_points_rect_one_shot(self):
        """一次拖拽结束后关闭点云框选模式，需再次点击工具栏才能框选。"""
        self.points_rect_select_mode = False
        if hasattr(self, "points_rect_select_action"):
            self.points_rect_select_action.setChecked(False)
        self._update_frame_info_label()
        self._update_points_rect_button_style(False)

    def _restore_points_after_rect_select(self):
        """取消/开始新一轮框选前：去掉红色高亮并清空表格数据区。"""
        if self._points_rect_select_mask is not None:
            self._points_rect_select_mask = None
            self.vis_fram(updata_color_bar=False)
        self._reset_point_select_table_ui()

    def _restore_points_color_only(self):
        """仅恢复点云颜色（去掉红色高亮），不清空/隐藏左侧表格。"""
        if self._points_rect_select_mask is not None:
            self._points_rect_select_mask = None
            self.vis_fram(updata_color_bar=False)

    def _update_points_rect_button_style(self, active: bool):
        """让“点云框选”按钮在生效期间呈现不同颜色。"""
        try:
            if not hasattr(self, "toolbar") or self.points_rect_select_action is None:
                return
            btn = self.toolbar.widgetForAction(self.points_rect_select_action)
            if btn is None:
                return
            if active:
                btn.setStyleSheet(
                    "QToolButton { background-color: #2196F3; color: white; border-radius: 6px; padding: 6px 10px; }"
                )
            else:
                btn.setStyleSheet("QToolButton { background-color: transparent; border-radius: 6px; padding: 6px 10px; }")
        except Exception:
            # 样式更新失败不影响功能
            pass

    def _ensure_point_select_dock(self):
        if self._point_select_dock is not None:
            return
        dock = QDockWidget("框选点云", self)
        dock.setObjectName("PointRectSelectDock")
        w = QWidget()
        lay = QVBoxLayout(w)
        self.point_select_info_label = QLabel(
            "点击「点云框选」后，在主视图拖拽一次矩形；选中点为红色，下方表格查看各点字段。"
        )
        self.point_select_info_label.setWordWrap(True)
        lay.addWidget(self.point_select_info_label)
        self.point_select_table = QTableWidget()
        self.point_select_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.point_select_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.point_select_table.setAlternatingRowColors(True)
        self.point_select_table.horizontalHeader().setStretchLastSection(True)
        self.point_select_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        # 允许用户点击表头对列排序（Qt 会在每次点击时在升/降之间切换）
        self.point_select_table.setSortingEnabled(True)
        lay.addWidget(self.point_select_table, 1)
        dock.setWidget(w)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self._point_select_dock = dock

    def _reset_point_select_table_ui(self):
        if self._point_select_dock is None:
            return
        self.point_select_table.clearContents()
        self.point_select_table.setRowCount(0)
        self.point_select_table.setColumnCount(0)
        self.point_select_table.setHorizontalHeaderLabels([])

    @staticmethod
    def _colors_to_rgba_n4(colors, n):
        """将当前 self.colors（纯色元组或 per-point ndarray）规范为 (N,4) float32 RGBA。"""
        if n <= 0:
            return np.zeros((0, 4), dtype=np.float32)
        if isinstance(colors, tuple):
            r, g, b = float(colors[0]), float(colors[1]), float(colors[2])
            a = float(colors[3]) if len(colors) > 3 else 1.0
            out = np.empty((n, 4), dtype=np.float32)
            out[:, 0] = r
            out[:, 1] = g
            out[:, 2] = b
            out[:, 3] = a
            return out
        arr = np.asarray(colors, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] == n:
            if arr.shape[1] == 4:
                return arr.astype(np.float32)
            if arr.shape[1] == 3:
                out = np.zeros((n, 4), dtype=np.float32)
                out[:, :3] = arr.astype(np.float32)
                out[:, 3] = 1.0
                return out
        if arr.size == 4:
            return np.tile(arr.astype(np.float32).reshape(1, 4), (n, 1))
        return np.ones((n, 4), dtype=np.float32) * 0.7

    def _point_select_column_names(self):
        """表格列名：PCD 字段名；无元数据时用 x,y,z,4,5,6,...（不含点索引列）。"""
        if getattr(self, "structured_points", None) is not None and len(self.structured_points) > 0:
            cols = []
            for name in self.structured_points.dtype.names or ():
                if name is None or str(name).startswith("_"):
                    continue
                cols.append(str(name))
            return cols
        if self.raw_points is None or len(self.raw_points) == 0:
            return []
        n = int(self.raw_points.shape[1])
        meta = getattr(self, "metadata", None) or []
        usable = [str(m) for m in meta if m and m != "_"]
        if len(usable) == n:
            return usable
        base = ["x", "y", "z"]
        if n <= 3:
            return base[:n]
        return base + [str(k) for k in range(4, n + 1)]

    def _point_cell_value(self, idx, col_name):
        if getattr(self, "structured_points", None) is not None and idx < len(self.structured_points):
            names = self.structured_points.dtype.names or ()
            if col_name not in names:
                return ""
            row = self.structured_points[int(idx)]
            try:
                v = row[col_name]
                if isinstance(v, (bytes, np.bytes_)):
                    return v.decode("utf-8", errors="replace")
                if isinstance(v, np.floating) or isinstance(v, float):
                    return "{:.6g}".format(float(v))
                if isinstance(v, (np.integer, int)):
                    return str(int(v))
                return str(v)
            except (TypeError, ValueError):
                return str(row[col_name])
        if self.raw_points is not None and idx < len(self.raw_points):
            row = self.raw_points[idx]
            meta = list(getattr(self, "metadata", None) or [])
            usable = [str(m) for m in meta if m and m != "_"]
            n = len(row)
            if len(usable) == n:
                try:
                    j = usable.index(col_name)
                    return "{:.6g}".format(float(self.raw_points[idx, j]))
                except ValueError:
                    pass
            if col_name == "x" and n >= 1:
                return "{:.6g}".format(float(row[0]))
            if col_name == "y" and n >= 2:
                return "{:.6g}".format(float(row[1]))
            if col_name == "z" and n >= 3:
                return "{:.6g}".format(float(row[2]))
            if col_name.isdigit():
                j = int(col_name) - 1
                if 0 <= j < n:
                    return "{:.6g}".format(float(row[j]))
        return ""

    def _point_cell_sort_value(self, idx, col_name):
        """单元格排序值：写入 QTableWidgetItem 的 EditRole，保证数值按数值排序。"""
        # structured_points：优先用字段值
        if getattr(self, "structured_points", None) is not None and idx < len(self.structured_points):
            names = self.structured_points.dtype.names or ()
            if col_name in names:
                row = self.structured_points[int(idx)]
                try:
                    v = row[col_name]
                    if isinstance(v, (bytes, np.bytes_)):
                        return v.decode("utf-8", errors="replace")
                    if isinstance(v, (np.floating, float)):
                        return float(v)
                    if isinstance(v, (np.integer, int)):
                        return int(v)
                    return str(v)
                except Exception:
                    return None

        # raw_points：兜底（x/y/z 或数字列名）
        if getattr(self, "raw_points", None) is not None and idx < len(self.raw_points):
            row = self.raw_points[int(idx)]
            n = int(self.raw_points.shape[1])
            if col_name == "x" and n >= 1:
                return float(row[0])
            if col_name == "y" and n >= 2:
                return float(row[1])
            if col_name == "z" and n >= 3:
                return float(row[2])
            if str(col_name).isdigit():
                j = int(col_name) - 1
                if 0 <= j < n:
                    return float(row[j])

            # 可能是 metadata 名字：尝试按 metadata 对齐（用于排序）
            meta = list(getattr(self, "metadata", None) or [])
            usable = [str(m) for m in meta if m and m != "_"]
            if col_name in usable:
                j = usable.index(col_name)
                if 0 <= j < n:
                    try:
                        return float(row[j])
                    except Exception:
                        return None
        return None

    def _fill_point_select_table(self, mask):
        self._ensure_point_select_dock()
        cols = self._point_select_column_names()
        indices = np.flatnonzero(np.asarray(mask, dtype=bool))
        total = int(len(indices))
        self.point_select_table.clearContents()
        self.point_select_table.setColumnCount(len(cols))
        self.point_select_table.setHorizontalHeaderLabels(cols)
        if total == 0:
            self.point_select_table.setRowCount(0)
            self.point_select_info_label.setText("框选区域内无点。")
            return
        shown = min(total, LIST_POINT_SELECT_CAP)
        self.point_select_info_label.setText(
            "已选中 {} 个点（表格显示 {} 行{}）。".format(
                total,
                shown,
                "，其余行省略" if total > LIST_POINT_SELECT_CAP else "",
            )
        )
        self.point_select_table.setRowCount(shown)
        for r in range(shown):
            idx = int(indices[r])
            for c, col_name in enumerate(cols):
                text = self._point_cell_value(idx, col_name)
                item = QTableWidgetItem(text)
                sort_val = self._point_cell_sort_value(idx, col_name)
                if sort_val is None:
                    item.setData(Qt.EditRole, text)
                else:
                    item.setData(Qt.EditRole, sort_val)
                self.point_select_table.setItem(r, c, item)
        if total > LIST_POINT_SELECT_CAP:
            self.point_select_info_label.setText(
                self.point_select_info_label.text()
                + " … 另有 {} 点未列出（可缩小框选范围）。".format(total - LIST_POINT_SELECT_CAP)
            )

    def _apply_points_in_rect_from_drag(self, x1, y1, x2, y2):
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        if not hasattr(self, "raw_points") or self.raw_points is None or len(self.raw_points) < 1:
            self.frame_info_label.setText("无点云可框选")
            return
        pts_xyz = np.asarray(self.raw_points[:, :3], dtype=np.float64)
        mask = points_in_screen_rect(self.glwidget, pts_xyz, x_min, x_max, y_min, y_max)
        self._ensure_point_select_dock()
        # 拖拽完成后再弹出表格（点击框选按钮时不弹出）
        if self._point_select_dock is not None:
            self._point_select_dock.show()
            self._point_select_dock.raise_()
        if not np.any(mask):
            self._points_rect_select_mask = None
            self._fill_point_select_table(mask)
            self.frame_info_label.setText("框选区域内无点")
            self.vis_fram(updata_color_bar=False)
            return
        self._points_rect_select_mask = mask
        self._fill_point_select_table(mask)
        self.frame_info_label.setText("已框选 {} 个点".format(int(np.count_nonzero(mask))))
        self.vis_fram(updata_color_bar=False)

    def _clear_point_rect_selection(self):
        """取消框选：恢复点云颜色并清空表格。"""
        # 从工具栏取消框选时，必须保证按钮状态与高亮状态一致
        self.points_rect_select_mode = False
        if hasattr(self, "points_rect_select_action"):
            self.points_rect_select_action.setChecked(False)
        self.box_select_start = None
        self.box_select_start_logical = None
        self.box_select_overlay.clear_rect()
        self._restore_points_after_rect_select()
        if self._point_select_dock is not None:
            # “去掉表格” - 直接隐藏整个 dock（含表格）
            self.point_select_info_label.setText(
                "点击「点云框选」后，在主视图拖拽一次矩形；选中点为红色，下方表格查看各点字段。"
            )
            self._point_select_dock.hide()
        self._update_frame_info_label()
        self._update_points_rect_button_style(False)

    def _add_bbox_from_rect(self, x1, y1, x2, y2):
        """根据屏幕矩形框选的点，拟合 roll=0 pitch=0 的贴合包围框；无点时投影到地面生成框"""
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        pts_xyz = self.raw_points[:, :3] if (hasattr(self, "raw_points") and self.raw_points is not None and len(self.raw_points) > 0) else None
        selected = None
        if pts_xyz is not None:
            mask = points_in_screen_rect(self.glwidget, pts_xyz, x_min, x_max, y_min, y_max)
            selected = pts_xyz[mask]
            selected = filter_ground_points(selected)  # 过滤地面点
        if selected is not None and len(selected) >= 3:
            xy_fit = fit_obb_xy(selected[:, :2])
            if xy_fit is not None:
                x_c, y_c, l, w, yaw = xy_fit
                z_min = float(np.min(selected[:, 2]))
                z_max = float(np.max(selected[:, 2]))
                z_c = (z_min + z_max) / 2
                h = max(z_max - z_min, 0.1)
                self._append_bbox(x_c, y_c, z_c, l, w, h, yaw)
                return
        # 无点或点数不足：将屏幕矩形四角投影到地面，生成轴对齐框
        self._add_bbox_from_rect_empty(x_min, x_max, y_min, y_max)

    def _append_bbox(self, x_c, y_c, z_c, l, w, h, yaw):
        """将拟合好的框追加到列表"""
        class_name = "others"
        if class_name in self.class_map.keys():
            bbox_color = self.class_map[class_name]
        else:
            bbox_color = self.class_map["others"]
        color = QColor(bbox_color[0], bbox_color[1], bbox_color[2], bbox_color[3])
        bbox = draw_bbox(x_c, y_c, z_c, l, w, h, yaw, color)
        arrow = draw_arrow(np.array([x_c, y_c, z_c + h/2]), [np.cos(yaw), np.sin(yaw), 0], l/2, color)
        vis_text = GLTextItem(text=class_name + "-new", pos=(x_c, y_c, z_c+1), color=color, font=QFont('Helvetica', 10))
        self.glwidget.addItem(bbox)
        self.glwidget.addItem(arrow)
        self.glwidget.addItem(vis_text)
        self.current_bbox_items.extend([bbox, arrow, vis_text])
        info = {"x": x_c, "y": y_c, "z": z_c, "l": l, "w": w, "h": h, "yaw": yaw, "roll": 0.0, "pitch": 0.0, "class_name": class_name}
        self.current_bbox_infos.append(info)
        self.box_select_mode = False
        self.box_select_action.setChecked(False)
        self.frame_info_label.setText(f"已添加新框，共 {len(self.current_bbox_infos)} 个")
        self.bbox_modified = True
        self._show_save_button_if_modified()

    def _save_bboxes_clicked(self):
        """将修改后的框保存到原 JSON 文件"""
        if not hasattr(self, "json_path") or not self.json_path:
            self.frame_info_label.setText("无可保存的 JSON 路径")
            return
        if not self.current_bbox_infos:
            self.frame_info_label.setText("无目标框可保存")
            return
        try:
            save_bboxes_to_tanway_json(
                self.json_path,
                self.current_bbox_infos,
                self.original_json_agents,
            )
            self.bbox_modified = False
            self.save_bboxes_btn.hide()
            self.frame_info_label.setText(f"已保存到 {os.path.basename(self.json_path)}")
        except Exception as e:
            self.frame_info_label.setText(f"保存失败: {e}")
            QMessageBox.warning(self, "保存失败", str(e))

    def _add_bbox_from_rect_empty(self, x_min, x_max, y_min, y_max):
        """框选区域无点时，将屏幕矩形四角投影到地面生成轴对齐框"""
        corners_screen = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        pts_3d = []
        z_plane = 0.0
        if hasattr(self, "raw_points") and self.raw_points is not None and len(self.raw_points) > 0:
            z_plane = float(np.median(self.raw_points[:, 2]))
        for sx, sy in corners_screen:
            ray = ray_from_screen(self.glwidget, sx, sy)
            if ray is None:
                self.frame_info_label.setText("投影失败，无法生成框")
                return
            origin, direction = ray
            pt = ray_plane_z_intersect(origin, direction, z_plane)
            if pt is None:
                self.frame_info_label.setText("投影失败，无法生成框")
                return
            pts_3d.append(pt)
        pts_3d = np.array(pts_3d)
        x_c = float(np.mean(pts_3d[:, 0]))
        y_c = float(np.mean(pts_3d[:, 1]))
        z_c = z_plane
        l = max(float(np.ptp(pts_3d[:, 0])), 0.1)
        w = max(float(np.ptp(pts_3d[:, 1])), 0.1)
        h = 2.0
        yaw = 0.0
        self._append_bbox(x_c, y_c, z_c, l, w, h, yaw)

    def _rebuild_link_arrows(self):
        """根据 current_bbox_infos 重建 link 弧线箭头"""
        for item in getattr(self, "current_link_arrows", []):
            self.glwidget.removeItem(item)
        self.current_link_arrows = []
        if not self.current_bbox_infos:
            return
        id_to_center = {}
        for inf in self.current_bbox_infos:
            bid = inf.get("id")
            if bid is not None:
                id_to_center[str(bid)] = (inf["x"], inf["y"], inf["z"])
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
            src = (inf["x"], inf["y"], inf["z"])
            for tid in target_ids:
                if tid is None:
                    continue
                tgt = id_to_center.get(str(tid))
                if tgt is not None:
                    arc = draw_arc_arrow(src, tgt, line_color)
                    if arc is not None:
                        self.glwidget.addItem(arc)
                        self.current_link_arrows.append(arc)
                else:
                    arc = draw_arc_arrow_missing(src, line_color)
                    if arc is not None:
                        self.glwidget.addItem(arc)
                        self.current_link_arrows.append(arc)
                    label_pos = (src[0], src[1], src[2] + 2.8)
                    missing_text = GLTextItem(
                        text="ID:{} 缺失".format(tid),
                        pos=label_pos,
                        color=line_color,
                        font=QFont("Helvetica", 9),
                    )
                    self.glwidget.addItem(missing_text)
                    self.current_link_arrows.append(missing_text)

    def _update_frame_info_label(self):
        """更新底部帧信息标签"""
        if self.point_cloud_files and 0 <= self.current_frame_index < len(self.point_cloud_files):
            self.frame_info_label.setText(
                f"Frame: {self.current_frame_index + 1} / {len(self.point_cloud_files)} ({self.point_cloud_files[self.current_frame_index]})")
        elif self.point_cloud_files:
            self.frame_info_label.setText(f"Frame: {self.current_frame_index + 1} / {len(self.point_cloud_files)}")
        else:
            self.frame_info_label.setText("Point Cloud Viewer")

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
        self._user_solid_rgbf = self.colors
        if self.pcd_file:
            self.directory = None
            self.point_cloud_files = []
            self.current_frame_index = -1
            self.raw_points, self.structured_points, metadata = get_points_from_pcd_file(self.pcd_file)
            self.metadata = metadata
            self._points_rect_select_mask = None
            if self._point_select_dock is not None:
                self._reset_point_select_table_ui()
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
        self._points_rect_select_mask = None
        if self._point_select_dock is not None:
            self._reset_point_select_table_ui()
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
            self._user_solid_rgbf = self.colors
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
        self.bbox_modified = False
        self.save_bboxes_btn.hide()
        if self.bboxes_directory is not None:
            self.json_path = os.path.join(str(self.bboxes_directory), str(Path(self.pcd_file).stem)+".json")
            self.original_json_agents = None
            if os.path.isfile(self.json_path):
                self.original_json_agents = load_json(self.json_path)
                json_data = get_anno_from_tanway_json(self.original_json_agents)

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
        else:
            # 未按字段映射颜色时：必须从「基底」重建，不可沿用上一帧（可能已把框选红色写入 self.colors）
            rgbf = getattr(self, "_user_solid_rgbf", None)
            if rgbf is not None:
                self.colors = rgbf
            elif len(self.points) > 0:
                self.colors = self.Colors[0](self.min_max_normalization(self.points[:, 0]))
                    
        rgba = self._colors_to_rgba_n4(self.colors, len(self.points))
        m = getattr(self, "_points_rect_select_mask", None)
        if m is not None:
            if len(m) != len(self.points):
                self._points_rect_select_mask = None
            elif np.any(m):
                rgba = rgba.copy()
                rgba[np.asarray(m, dtype=bool)] = (1.0, 0.0, 0.0, 1.0)
        self.scatter = GLScatterPlotItem(pos=self.points, color=rgba, size=self.point_size)
        self.colors = rgba
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