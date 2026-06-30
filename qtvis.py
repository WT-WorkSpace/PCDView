import sys
import os
import json

# 在导入 PyQt5 / pyqtgraph 之前，强制使用软件 OpenGL
os.environ.setdefault("QT_XCB_FORCE_SOFTWARE_OPENGL", "1")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
# 本项目使用 PyQt5；若环境中存在残缺的 PySide6（如仅 pyside6_essentials），
# pyqtgraph 会误选 PySide6 并在读取 __version__ 时失败。
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
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
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog, QFormLayout, QDialogButtonBox
from PyQt5.QtWidgets import QLabel, QSizePolicy, QSlider, QMenuBar, QComboBox, QDoubleSpinBox, QLineEdit
from PyQt5.QtWidgets import QAction, QToolBar, QWidget, QPushButton, QColorDialog
from PyQt5.QtWidgets import QSplitter, QFrame, QMessageBox, QShortcut, QDockWidget
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QAbstractItemView
from PyQt5.QtWidgets import QHeaderView, QCheckBox, QSpinBox, QGroupBox, QGridLayout
from PyQt5.QtWidgets import QTextBrowser
from PyQt5.QtCore import QEvent, QSize
from PyQt5.QtGui import QKeySequence, QPainter, QPen, QCursor
from matplotlib.path import Path as MplPath

from utils.bbox_pick import (
    pick_bbox_index, points_in_screen_rect, fit_obb_xy,
    filter_ground_points, ray_from_screen, ray_plane_z_intersect,
)
from widget.opengl_widget import PCDViewWidget
from widget.bbox_three_views import BboxThreeViewsPanel
from widget.bbox_attr_panel import BboxAttributePanel
from utils.utils import pil2qicon
from utils.load_pcd import get_points_from_pcd_file
from utils.load_bboxes_json import get_anno_from_tanway_json, save_bboxes_to_tanway_json

from utils.utils import load_json, app_icon

# 将部分“独立类”拆分到单独文件中，避免 qtvis.py 过长
from ui.box_select_overlay import BoxSelectOverlay
from dialogs.plane_param_dialog import PlaneParamDialog
from dialogs.mask_param_dialog import MaskParamDialog
from dialogs.bbox_attr_settings_dialog import BboxAttrSettingsDialog
from features.point_rect_select_mixin import PointRectSelectMixin
from features.extrinsic_calib_mixin import ExtrinsicCalibMixin
from features.plane_mixin import PlaneMixin
from features.mask_mixin import MaskMixin
from features.obstacle_cluster import ObstacleCluster

LIST_POINT_SELECT_CAP = 8000  # 列表展示上限，避免一次框选过多点时界面卡死

class PointCloudViewer(
    QMainWindow, PCDViewWidget, PointRectSelectMixin, PlaneMixin, MaskMixin, ExtrinsicCalibMixin
):
    def __init__(self):
        QMainWindow.__init__(self)
        PCDViewWidget.__init__(self)
        self.init_ui()
        self.create_menus()
        self.create_toolbar()
        self.create_controls()
        self.init_state()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def init_ui(self):
        # PyInstaller onefile 下资源在 sys._MEIPASS/icons
        if getattr(sys, "frozen", False):
            self.curpath = sys._MEIPASS
        else:
            self.curpath = os.path.dirname(os.path.abspath(__file__))
        self.setWindowTitle("Point Cloud Viewer")
        self.setWindowIcon(app_icon())
        self.setGeometry(100, 100, 850, 600)
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        # 主内容：上方为 3D 视图/三视图 splitter，下方为横跨全宽的播放控制条
        self.central_widget = QWidget()
        self.central_layout = QVBoxLayout(self.central_widget)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(0)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        # 左侧容器只放 3D 视图；底部播放条放在 central_widget 底部横跨全宽。
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

        self.central_layout.addWidget(self.splitter, 1)
        self.setCentralWidget(self.central_widget)

    def init_state(self):
        self.point_cloud_files = []
        self.current_frame_index = -1
        self.scatter_item = GLScatterPlotItem()
        self.playing = False  # Flag for play/pause state
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)  # Timer action for auto-frame transition
        self.key_frame_timer = QTimer(self)
        self.key_frame_timer.timeout.connect(self._step_held_frame_key)
        self._held_frame_key_direction = 0
        self.colors = QColor(0, 0, 255).getRgbF()
        # 纯色模式下用于每次 vis_fram 重建颜色；避免框选红色写入 self.colors 后无法恢复
        self._user_solid_rgbf = QColor(0, 0, 255).getRgbF()
        self.color_fields = None
        self.metadata = None
        self.current_bbox_items = []
        self.current_bbox_infos = []  # 每个框的详细信息，用于点击弹窗
        self.bbox_attr_defs = self._load_bbox_attr_defs()
        self.current_link_arrows = []  # Semitrailer 指向 link_id 目标的弧线箭头
        self.selected_bbox_index = None  # 当前选中的框索引，用于实体框高亮
        self.box_select_mode = False  # 框选模式：拖拽生成新矩形框
        self.box_select_start = None  # 框选起始屏幕坐标 (x, y)，设备像素
        self.box_select_start_logical = None  # 框选起始逻辑坐标，用于 overlay 绘制
        self._bbox_drag_state = None  # 选中目标框后左键拖动移动框
        self.bbox_modified = False  # 框是否被修改过，用于显示 Save 按钮
        self.original_json_agents = None  # 原始 JSON agent 列表，保存时用于保留额外字段
        self.bboxes_directory = None
        self.json_path = None
        self.history_frames_directory = None
        self.history_frame_index = {}
        self.history_points_cache = {}
        self.history_single_points_cache = {}
        self.history_scatter = None
        self.history_shift_down = False
        self.history_display_mode = "overlay"
        self.history_browse_index = 0
        self.history_main_scatter_hidden = False

        self.points_rect_select_mode = False  # 点云框选：拖拽矩形，选中点标红并列表展示
        self._points_rect_select_mask = None  # 与当前点云等长的 bool 掩码，或 None
        self._point_select_dock = None

        # 平面绘制：可切换的“添加平面”
        self._plane_item = None  # GLGridItem 或 GLMeshItem
        # 当前平面参数（用于“添加/取消平面”按钮和“修改平面参数”对话框）
        self._plane_params = {
            "plane_type": "网格",
            "plane_length": 100.0,
            "plane_width": 100.0,
            "grid_spacing": 10.0,
            "center": (0.0, 0.0, -1.7),
            "color_rgb": (180, 180, 180),
            "alpha": 100.0,
        }
        if not hasattr(self, "_add_plane_action"):
            self._add_plane_action = None

        # Mask 绘制
        self._mask_items = []
        self._mask_visible = False
        self._mask_toggle_action = None
        self._mask_params = {
            "json_path": "",
            "point_size": 8.0,
            "line_width": 2.0,
            # 点颜色默认红色
            "point_color": (255, 0, 0),
            # 线颜色默认墨绿色
            "line_color": (0, 100, 0),
            "point_z": 0.0,
            "keep_inside_points": False,
        }
        self._mask_settings_action = None
        self._cluster_bbox_items = []
        self._cluster_bbox_infos = []
        self._selected_cluster_bbox_index = None
        self._cluster_select_mask = None  # 基于 raw_points 长度的 bool 掩码，标记被点击聚类框内的点
        self._cluster_enabled = False
        self._extrinsic_init_state()
        self._obstacle_cluster = ObstacleCluster()
        self._cluster_params = {
            "eps": 0.5,
            "min_points": 5,
            "max_points": 20000,
            "voxel_size": 0.1,
            "use_lshape": False,
            "use_roi": True,
            "roi_x_min": -100.0,
            "roi_x_max": 100.0,
            "roi_y_min": -100.0,
            "roi_y_max": 100.0,
            "roi_z_min": -1.5,
            "roi_z_max": 3.0,
            "use_size_filter": False,
            "l_min": 0.0,
            "l_max": 1000.0,
            "w_min": 0.0,
            "w_max": 1000.0,
            "h_min": 0.0,
            "h_max": 1000.0,
        }

        self.right_button_pressed = False
        self.last_mouse_pos = None

    def create_controls(self):
        # 播放控制条属于 central_widget 底部，横跨 3D 视图和三视图面板。
        self.play_button = QPushButton(self.central_widget)
        self.play_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/play_pcd.png')))
        self.play_button.setIconSize(self.play_button.sizeHint())
        self.play_button.setFlat(True)

        self.prev_button = QPushButton(self.central_widget)
        self.prev_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/prev_pcd.png')))
        self.prev_button.setIconSize(self.prev_button.sizeHint() * 0.8)
        self.prev_button.setFlat(True)

        self.next_button = QPushButton(self.central_widget)
        self.next_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/next.png')))
        self.next_button.setIconSize(self.next_button.sizeHint() * 0.8)
        self.next_button.setFlat(True)

        font = QFont("Arial", 10)
        self.frame_info_display_chars = 35
        self.frame_info_label = QLabel("0 / 0", self.central_widget)
        self.frame_info_label.setFont(font)
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        self.frame_info_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        frame_info_width = self.frame_info_label.fontMetrics().horizontalAdvance(
            "0" * self.frame_info_display_chars
        )
        self.frame_info_label.setFixedWidth(frame_info_width)
        self.frame_info_label.setMinimumHeight(int(font.pointSize() * 1.5))

        self.log_info_edit = QLineEdit(self.central_widget)
        self.log_info_edit.setReadOnly(True)
        self.log_info_edit.setFrame(False)
        self.log_info_edit.setPlaceholderText("信息输出")
        self.log_info_edit.setMinimumWidth(300)
        self.log_info_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.log_info_edit.setMinimumHeight(int(font.pointSize() * 1.8))
        self.log_info_edit.setStyleSheet(
            "QLineEdit { border: none; background: transparent; padding-left: 4px; }"
        )

        self.frame_slider = QSlider(Qt.Horizontal, self.central_widget)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setTickPosition(QSlider.TicksBelow)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.valueChanged.connect(self.on_slider_value_changed)

        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(0)
        control_layout.addWidget(self.prev_button)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.next_button)
        control_layout.addWidget(self.frame_info_label, 1)
        control_layout.addWidget(self.frame_slider, 6)
        control_layout.addWidget(self.log_info_edit, 1)

        # 将 3D 视图放入左侧容器，保证父子关系正确才能正常显示
        self.glwidget.setParent(self.left_widget)
        main_layout = QVBoxLayout(self.left_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.glwidget, 1)
        self.central_layout.addLayout(control_layout)

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
        self.rotate_yaw_shortcut = QShortcut(QKeySequence(Qt.Key_C), self)
        self.rotate_yaw_shortcut.activated.connect(self._rotate_selected_bbox_yaw_90)
        self.next_bbox_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.next_bbox_shortcut.activated.connect(self._show_next_bbox_three_view)
        self.prev_frame_shortcut = QShortcut(QKeySequence(Qt.Key_Z), self)
        self.prev_frame_shortcut.activated.connect(self.previous_frame)
        self.next_frame_shortcut = QShortcut(QKeySequence(Qt.Key_X), self)
        self.next_frame_shortcut.activated.connect(self.next_frame)
        self.prev_history_frame_shortcut = QShortcut(QKeySequence(Qt.SHIFT | Qt.Key_Z), self)
        self.prev_history_frame_shortcut.activated.connect(self.previous_frame)
        self.next_history_frame_shortcut = QShortcut(QKeySequence(Qt.SHIFT | Qt.Key_X), self)
        self.next_history_frame_shortcut.activated.connect(self.next_frame)
        # Z/X 长按播放由 eventFilter 统一处理，避免 QShortcut 和按住定时器重复切帧。
        self.prev_frame_shortcut.setEnabled(False)
        self.next_frame_shortcut.setEnabled(False)
        self.prev_history_frame_shortcut.setEnabled(False)
        self.next_history_frame_shortcut.setEnabled(False)
        # 主视图右上角 Save 按钮（修改框后显示）
        self.copy_prev_bboxes_btn = QPushButton("Copy Prev", self.glwidget)
        self.copy_prev_bboxes_btn.setStyleSheet(
            "QPushButton { background-color: #4B5563; color: white; padding: 6px 12px; "
            "border-radius: 4px; font-weight: bold; } "
            "QPushButton:hover { background-color: #374151; } "
            "QPushButton:disabled { background-color: #9CA3AF; color: #F3F4F6; }"
        )
        self.copy_prev_bboxes_btn.clicked.connect(self._copy_previous_frame_bboxes)
        self.copy_prev_bboxes_btn.hide()

        self.save_bboxes_btn = QPushButton("Save", self.glwidget)
        self.save_bboxes_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; padding: 6px 12px; "
            "border-radius: 4px; font-weight: bold; } "
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self.save_bboxes_btn.clicked.connect(self._save_bboxes_clicked)
        self.save_bboxes_btn.hide()

        self.bbox_attr_panel = BboxAttributePanel(self.glwidget)
        self.bbox_attr_panel.attrSettingsRequested.connect(self._open_annotation_settings)
        self.bbox_attr_panel.hide()

    def _update_save_button_geometry(self):
        """将主视图右上角标注操作按钮定位。"""
        if self.glwidget.width() <= 0 or self.glwidget.height() <= 0:
            return
        m = 12
        gap = 8
        save_w = max(self.save_bboxes_btn.sizeHint().width(), 70)
        copy_w = max(self.copy_prev_bboxes_btn.sizeHint().width(), 100)
        bh = max(self.save_bboxes_btn.sizeHint().height(), self.copy_prev_bboxes_btn.sizeHint().height(), 28)
        self.save_bboxes_btn.setGeometry(
            self.glwidget.width() - save_w - m, m, save_w, bh
        )
        self.copy_prev_bboxes_btn.setGeometry(
            self.glwidget.width() - save_w - copy_w - gap - m, m, copy_w, bh
        )
        self.copy_prev_bboxes_btn.setEnabled(self.current_frame_index > 0 and bool(getattr(self, "point_cloud_files", [])))
        if getattr(self, "pcd_file", None):
            self.copy_prev_bboxes_btn.show()
            self.copy_prev_bboxes_btn.raise_()
        self.save_bboxes_btn.raise_()

    def _update_bbox_attr_panel_geometry(self):
        if not hasattr(self, "bbox_attr_panel") or self.glwidget.width() <= 0 or self.glwidget.height() <= 0:
            return
        margin = 16
        panel_w = min(430, max(360, int(self.glwidget.width() * 0.21)))
        preferred_h = self.bbox_attr_panel.preferred_height() if hasattr(self.bbox_attr_panel, "preferred_height") else self.bbox_attr_panel.sizeHint().height()
        panel_h = max(preferred_h + 8, 170)
        panel_h = min(panel_h, max(170, self.glwidget.height() - margin * 2))
        x = max(margin, self.glwidget.width() - panel_w - margin)
        y = max(margin, self.glwidget.height() - panel_h - margin)
        self.bbox_attr_panel.setGeometry(x, y, panel_w, panel_h)
        self.bbox_attr_panel.raise_()

    def _show_bbox_attr_panel(self, bbox_index, info):
        if not hasattr(self, "bbox_attr_panel"):
            return
        self.bbox_attr_panel.update_bbox(
            info,
            bbox_index=bbox_index,
            on_bbox_edited=self._on_bbox_edited_from_panel,
            class_names=list(self.class_map.keys()),
            attr_defs=self.bbox_attr_defs,
        )
        self._update_bbox_attr_panel_geometry()
        self.bbox_attr_panel.show()
        self.bbox_attr_panel.raise_()

    def _show_save_button_if_modified(self):
        """若框已修改且有可保存的 JSON 路径，则显示 Save 按钮"""
        if self.bbox_modified and getattr(self, "pcd_file", None):
            self._update_save_button_geometry()
            self.save_bboxes_btn.show()
            self.save_bboxes_btn.raise_()

    def create_toolbar(self):
        self.toolbar = QToolBar(self)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)
        self.toolbar.addAction(self.open_file_action)
        self.toolbar.addAction(self.open_dir_action)
        self.toolbar.addAction(self.open_bboxes_dir_action)
        self.toolbar.addAction(self.open_history_frames_action)
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

        # 红圈位置：添加/取消平面
        self._add_plane_action = self.create_action(
            "添加/取消平面",
            "icons/wangge.svg",
            self._toggle_add_plane,
        )
        self._add_plane_action.setCheckable(True)
        self.toolbar.addAction(self._add_plane_action)

        self._mask_toggle_action = self.create_action(
            "显示/关闭Mask",
            "icons/mask.svg",
            self._toggle_mask_visibility,
        )
        self._mask_toggle_action.setCheckable(True)
        self.toolbar.addAction(self._mask_toggle_action)
        self._cluster_action = self.create_action(
            "点云聚类",
            "icons/cluster.svg",
            self._toggle_cluster_from_toolbar,
        )
        self._cluster_action.setCheckable(True)
        self.toolbar.addAction(self._cluster_action)
        self.toolbar.addSeparator()
        self._extrinsic_calib_action = self.create_action(
            "外参标定",
            "icons/calibration.svg",
            self._toggle_extrinsic_calib,
        )
        self._extrinsic_calib_action.setCheckable(True)
        self.toolbar.addAction(self._extrinsic_calib_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.help_action)

        self.color_sidebar = QToolBar("colors", self)
        self.addToolBar(Qt.RightToolBarArea, self.color_sidebar)
        self.color_sidebar.setVisible(False)  # Initially hidden

    def create_menus(self):
        self.create_file_menus()
        self.create_tools_menus()
        self.create_help_menus()

    def create_file_menus(self):
        file_menu = self.menu_bar.addMenu("File")
        self.open_file_action = self.create_action("Open File", 'icons/open.svg', self.open_file)
        self.open_dir_action = self.create_action("Open Directory", 'icons/open_dir.svg', self.open_directory)
        self.open_bboxes_dir_action = self.create_action("Open BBoxes Dir", 'icons/open_boxes_dir.svg',self.open_bboxes_directory)
        self.open_history_frames_action = self.create_action("导入历史帧", 'icons/history.svg', self.open_history_frames_directory)
        file_menu.addAction(self.open_file_action)
        file_menu.addAction(self.open_dir_action)
        file_menu.addAction(self.open_bboxes_dir_action)
        file_menu.addAction(self.open_history_frames_action)

    def create_tools_menus(self):
        tool_menu = self.menu_bar.addMenu("Tools")
        tool_pointsize_menu = tool_menu.addMenu("Point Size")
        self.increase_pointsize_action = self.create_action("Point Size +", 'icons/pointsize_increase.png', self.increase_points_size)
        self.decrease_pointsize_action = self.create_action("Point Size -", 'icons/pointsize_decrease.png', self.decrease_points_size)
        self.points_color = self.create_action("Color", 'icons/color.svg', self.select_color)
        self.coordinate = self.create_action("Coordinate", 'icons/coordinate.svg', self.create_coordinate)
        # Tools 里提供平面参数修改
        self._modify_plane_params_action = self.create_action(
            "修改平面参数", "icons/add_bbox.svg", self._modify_plane_params
        )
        self._mask_settings_action = self.create_action(
            "Mask设置", "icons/mask.svg", self._open_mask_settings
        )
        self._cluster_params_action = self.create_action(
            "当前帧点云聚类", "icons/cluster.svg", self._open_cluster_dialog
        )
        self._extrinsic_calib_menu_action = self.create_action(
            "多雷达外参标定", "icons/calibration.svg", self._toggle_extrinsic_calib
        )
        self._extrinsic_calib_menu_action.setCheckable(True)
        self._bbox_attr_settings_action = self.create_action(
            "自定义标注属性", "icons/add_bbox.svg", self._open_annotation_settings
        )
        self.save_view_action = self.create_action("Save View", 'icons/save_view.svg', self.save_view)
        self.load_view_action = self.create_action("Load View", 'icons/load_view.svg', self.load_view)

        tool_pointsize_menu.addAction(self.increase_pointsize_action)
        tool_pointsize_menu.addAction(self.decrease_pointsize_action)
        tool_menu.addAction(self.points_color)
        tool_menu.addAction(self.coordinate)
        tool_menu.addAction(self._modify_plane_params_action)
        tool_menu.addAction(self._mask_settings_action)
        tool_menu.addAction(self._cluster_params_action)
        tool_menu.addAction(self._extrinsic_calib_menu_action)
        tool_menu.addAction(self._bbox_attr_settings_action)
        tool_menu.addAction(self.save_view_action)
        tool_menu.addAction(self.load_view_action)

    def create_help_menus(self):
        help_menu = self.menu_bar.addMenu("Help")
        self.help_action = self.create_action("功能说明", "icons/help.svg", self.show_help_manual)
        help_menu.addAction(self.help_action)

    def _bbox_attr_config_path(self):
        return Path.home() / ".pcdview_bbox_attrs.json"

    def _default_bbox_attr_defs(self):
        class_options = list(self.class_map.keys()) if hasattr(self, "class_map") else ["others"]
        return [
            {"key": "class_name", "name": "类别", "label": "类别", "type": "select", "options": class_options, "system": True, "allow_empty": False, "default": "others"},
            {"key": "id", "name": "ID", "label": "ID", "type": "text", "options": [], "system": True, "allow_empty": True},
            {"key": "link_id", "name": "关联 ID", "label": "关联 ID", "type": "text", "options": [], "system": True, "allow_empty": True},
            {"key": "confidence", "name": "置信度", "label": "置信度", "type": "check", "options": ["0", "1", "2"], "system": True, "multi": False, "allow_empty": True},
            {"key": "movement_state", "name": "运动状态", "label": "运动状态", "type": "check", "options": ["0", "1"], "system": True, "multi": False, "allow_empty": True},
        ]

    def _load_bbox_attr_defs(self):
        defaults = self._default_bbox_attr_defs()
        path = self._bbox_attr_config_path()
        if not path.is_file():
            return defaults
        try:
            with open(path, "r", encoding="UTF-8") as f:
                data = json.load(f)
        except Exception:
            return defaults
        attr_defs = data.get("attributes", data) if isinstance(data, dict) else data
        if not isinstance(attr_defs, list):
            return defaults
        cleaned = []
        default_by_key = {item["key"]: item for item in defaults}
        for item in attr_defs:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or item.get("name") or "").strip()
            name = str(item.get("name") or item.get("label") or key).strip()
            attr_type = item.get("type") if item.get("type") in ("select", "check", "text") else "text"
            options = [str(v).strip() for v in (item.get("options") or []) if str(v).strip()]
            if key and name:
                cleaned_item = {"key": key, "name": name, "label": name, "type": attr_type, "options": options}
                if key in default_by_key or item.get("system"):
                    cleaned_item["system"] = True
                if not options and key in default_by_key:
                    cleaned_item["options"] = list(default_by_key[key].get("options") or [])
                cleaned_item["allow_empty"] = bool(item.get("allow_empty", default_by_key.get(key, {}).get("allow_empty", True)))
                if "default" in item:
                    cleaned_item["default"] = item.get("default")
                if attr_type == "check" and (item.get("multi") is False or default_by_key.get(key, {}).get("multi") is False):
                    cleaned_item["multi"] = False
                cleaned.append(cleaned_item)
        existing = {item["key"] for item in cleaned}
        missing_defaults = [item for item in defaults if item["key"] not in existing]
        return missing_defaults + cleaned

    def _save_bbox_attr_defs(self):
        path = self._bbox_attr_config_path()
        with open(path, "w", encoding="UTF-8") as f:
            json.dump({"attributes": self.bbox_attr_defs}, f, indent=2, ensure_ascii=False)

    def _open_annotation_settings(self):
        dialog = BboxAttrSettingsDialog(
            self.bbox_attr_defs,
            self.history_display_mode == "browse",
            self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        self.bbox_attr_defs = dialog.attr_defs()
        try:
            self._save_bbox_attr_defs()
        except Exception as exc:
            QMessageBox.warning(self, "自定义标注属性", "保存属性配置失败: {}".format(exc))
        if hasattr(self, "bbox_attr_panel"):
            self._prune_bbox_attr_infos()
            if self.current_bbox_infos:
                self.bbox_modified = True
                self._show_save_button_if_modified()
            self.bbox_attr_panel.set_attr_defs(self.bbox_attr_defs)
            if self.selected_bbox_index is not None and self.selected_bbox_index < len(self.current_bbox_infos):
                self._show_bbox_attr_panel(self.selected_bbox_index, self.current_bbox_infos[self.selected_bbox_index])
        self._set_history_display_mode(dialog.history_browse_enabled())

    def _parse_bbox_attr_default(self, attr_def):
        if "default" not in attr_def:
            if attr_def.get("key") == "class_name":
                return "others"
            if not attr_def.get("allow_empty", True) and attr_def.get("options"):
                value = attr_def["options"][0]
                if attr_def.get("key") in ("confidence", "movement_state"):
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return None
                return value
            return None
        value = attr_def.get("default")
        if value in ("", None):
            return None
        key = attr_def.get("key")
        attr_type = attr_def.get("type")
        if key in ("confidence", "movement_state"):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        if key in ("id", "link_id"):
            text = str(value).strip()
            if not text:
                return None
            if key == "link_id":
                vals = []
                for part in text.replace(";", ",").split(","):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        vals.append(int(part))
                    except ValueError:
                        vals.append(part)
                return vals[0] if len(vals) == 1 else (vals or None)
            try:
                return int(text)
            except ValueError:
                return text
        if attr_type == "check" and isinstance(value, (list, tuple)):
            return list(value) or None
        return str(value)

    def _default_bbox_attr_values(self):
        values = {}
        for attr_def in self.bbox_attr_defs:
            key = attr_def.get("key") or attr_def.get("name")
            if key:
                values[key] = self._parse_bbox_attr_default(attr_def)
        values["class_name"] = values.get("class_name") or "others"
        return values

    def _next_unique_bbox_id(self):
        used = set()
        max_int_id = 0
        for info in self.current_bbox_infos:
            text = self._bbox_id_key(info.get("id"))
            if not text:
                continue
            used.add(text)
            try:
                int_value = int(text)
            except ValueError:
                continue
            max_int_id = max(max_int_id, int_value)
        candidate = max_int_id + 1
        while str(candidate) in used:
            candidate += 1
        return candidate

    @staticmethod
    def _bbox_id_key(value):
        if value in (None, ""):
            return ""
        return str(value).strip()

    def _is_bbox_id_used_by_other(self, bbox_index, value):
        target = self._bbox_id_key(value)
        if not target:
            return False
        for i, info in enumerate(self.current_bbox_infos):
            if i == bbox_index:
                continue
            if self._bbox_id_key(info.get("id")) == target:
                return True
        return False

    def _prune_bbox_attr_infos(self):
        keep = {
            "x", "y", "z", "l", "w", "h", "yaw", "roll", "pitch",
            "class_name", "id", "link_id", "confidence", "movement_state",
        }
        keep.update(attr_def.get("key") or attr_def.get("name") for attr_def in self.bbox_attr_defs)
        keep.discard(None)
        for info in self.current_bbox_infos:
            for key in list(info.keys()):
                if key not in keep:
                    info[key] = None

    def create_action(self, name, icon_path, handler):
        icon = QIcon(os.path.join(self.curpath, icon_path))
        action = QAction(icon, name, self)
        action.triggered.connect(handler)
        return action

    def show_help_manual(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Point Cloud Viewer 功能说明")
        dlg.resize(760, 620)

        browser = QTextBrowser(dlg)
        browser.setOpenExternalLinks(False)
        browser.setHtml("""
        <h2>Point Cloud Viewer 功能说明</h2>

        <h3>文件与数据</h3>
        <ul>
          <li><b>Open File</b>：打开单个 PCD 点云文件。</li>
          <li><b>Open Directory</b>：打开点云目录，支持按帧播放和切换。</li>
          <li><b>Open BBoxes Dir</b>：选择目标框 JSON 标注目录。</li>
          <li><b>导入历史帧</b>：导入历史点云目录，导入成功后按钮蓝底表示已启用。</li>
        </ul>

        <h3>播放与切帧</h3>
        <ul>
          <li>底部播放栏用于上一帧、播放/暂停、下一帧和拖动帧进度。</li>
          <li><b>Z</b>：上一帧；长按 Z 连续向前播放。</li>
          <li><b>X</b>：下一帧；长按 X 连续向后播放。</li>
          <li>底部帧信息固定显示 35 个字符，完整文件名可悬停查看。</li>
        </ul>

        <h3>视图与显示</h3>
        <ul>
          <li><b>Point Size + / -</b>：增大或减小点云点大小，最小点大小为 1。</li>
          <li><b>Color</b>：选择点云颜色或字段映射颜色。</li>
          <li><b>Coordinate</b>：显示或隐藏坐标轴。</li>
          <li><b>Save View / Load View</b>：保存和加载当前视角。</li>
        </ul>

        <h3>标注与编辑</h3>
        <ul>
          <li><b>标注3D框</b>：在主视图拖拽矩形区域生成 3D 目标框。</li>
          <li>点击目标框会打开右侧三视图和目标框属性面板。</li>
          <li><b>Backspace</b>：删除当前选中的目标框。</li>
          <li><b>C</b>：将当前选中目标框 yaw 旋转 90 度。</li>
          <li><b>Space</b>：切换到下一个目标框三视图。</li>
          <li><b>Save</b>：保存当前帧标注；<b>Copy Prev</b>：复制上一帧标注。</li>
          <li><b>自定义标注属性</b>：配置目标框属性字段和历史帧显示模式。</li>
        </ul>

        <h3>点云框选</h3>
        <ul>
          <li><b>点云框选</b>：开启后在主视图拖拽一次矩形，选中点会高亮。</li>
          <li>框选完成后左侧显示点字段表格，不会挤占底部播放条。</li>
          <li><b>取消框选</b>：清除高亮并隐藏框选结果表格。</li>
        </ul>

        <h3>辅助功能</h3>
        <ul>
          <li><b>添加/取消平面</b>：显示或移除参考平面；启用时按钮蓝底。</li>
          <li><b>显示/关闭 Mask</b>：显示 Mask 点线或按 Mask 过滤点云。</li>
          <li><b>点云聚类</b>：对当前帧执行聚类并绘制聚类框；启用时按钮蓝底。</li>
          <li><b>外参标定</b>：打开外参标定面板，对多雷达点云进行位姿调整和应用。</li>
        </ul>
        """)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, dlg)
        buttons.rejected.connect(dlg.reject)

        layout = QVBoxLayout(dlg)
        layout.addWidget(browser, 1)
        layout.addWidget(buttons)
        dlg.exec_()

    def eventFilter(self, obj, event):
        """鼠标在 3D 视图上：框选模式拖拽生成框；否则左键点击显示三视图，右键显示目标框信息"""
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Shift and not event.isAutoRepeat():
            self.history_shift_down = True
            self._refresh_history_frame_visibility()
        elif event.type() == QEvent.KeyRelease and event.key() == Qt.Key_Shift and not event.isAutoRepeat():
            self.history_shift_down = False
            self._hide_history_frames()
        if self._handle_frame_key_event(event):
            return True
        if obj != self.glwidget:
            return super().eventFilter(obj, event)
        if event.type() == QEvent.Resize:
            self.box_select_overlay.setGeometry(0, 0, self.glwidget.width(), self.glwidget.height())
            self._update_save_button_geometry()
            self._update_bbox_attr_panel_geometry()

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

        if (self.history_shift_down and self.history_display_mode == "browse" and
                event.type() == QEvent.MouseButtonPress):
            if event.button() == Qt.LeftButton:
                self._step_history_browse_frame(1)
                return True
            if event.button() == Qt.RightButton:
                self._step_history_browse_frame(-1)
                return True

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
                    self._set_status_message("拖拽矩形过小，请点击「点云框选」再试一次")
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

        if self._bbox_drag_state is not None:
            if event.type() == QEvent.MouseMove and event.buttons() & Qt.LeftButton:
                self._update_bbox_drag(mx, my)
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._finish_bbox_drag()
                return True

        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton and self.current_bbox_infos:
            try:
                idx = pick_bbox_index(self.glwidget, mx, my, self.current_bbox_infos)
                if idx is not None:
                    self._show_bbox_info_dialog(idx)
                    self._start_bbox_drag(idx, mx, my)
                    return True
            except Exception as e:
                print("bbox drag start error:", e)

        if event.type() == QEvent.MouseButtonRelease and self._cluster_bbox_infos:
            try:
                cidx = pick_bbox_index(self.glwidget, mx, my, self._cluster_bbox_infos)
                if cidx is not None and event.button() in (Qt.LeftButton, Qt.RightButton):
                    self._selected_cluster_bbox_index = int(cidx)
                    self._update_cluster_select_mask_from_selected()
                    self.vis_fram(preserve_current_bboxes=True)
                    return True
            except Exception as e:
                print("cluster bbox pick error:", e)
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

    def _start_bbox_drag(self, bbox_index, mx, my):
        if bbox_index < 0 or bbox_index >= len(self.current_bbox_infos):
            self._bbox_drag_state = None
            return
        info = self.current_bbox_infos[bbox_index]
        z_plane = float(info.get("z", 0.0))
        ray = ray_from_screen(self.glwidget, mx, my)
        start_pt = ray_plane_z_intersect(ray[0], ray[1], z_plane) if ray is not None else None
        if start_pt is None:
            self._bbox_drag_state = None
            return
        self._bbox_drag_state = {
            "idx": bbox_index,
            "start_screen": (float(mx), float(my)),
            "start_pt": start_pt,
            "z_plane": z_plane,
            "orig_center": (
                float(info.get("x", 0.0)),
                float(info.get("y", 0.0)),
                float(info.get("z", 0.0)),
            ),
            "moved": False,
        }
        self.glwidget.setCursor(Qt.ClosedHandCursor)

    def _update_bbox_drag(self, mx, my):
        state = self._bbox_drag_state
        if not state:
            return
        idx = state["idx"]
        if idx < 0 or idx >= len(self.current_bbox_infos):
            self._bbox_drag_state = None
            self.glwidget.unsetCursor()
            return
        sx, sy = state["start_screen"]
        if not state["moved"] and abs(mx - sx) < 3 and abs(my - sy) < 3:
            return
        ray = ray_from_screen(self.glwidget, mx, my)
        cur_pt = ray_plane_z_intersect(ray[0], ray[1], state["z_plane"]) if ray is not None else None
        if cur_pt is None:
            return
        delta = cur_pt - state["start_pt"]
        ox, oy, oz = state["orig_center"]
        info = self.current_bbox_infos[idx]
        info["x"] = float(ox + delta[0])
        info["y"] = float(oy + delta[1])
        info["z"] = float(oz)
        state["moved"] = True
        self._refresh_single_bbox_in_main_view(idx)
        self._rebuild_link_arrows()

    def _finish_bbox_drag(self):
        state = self._bbox_drag_state
        self._bbox_drag_state = None
        self.glwidget.unsetCursor()
        if not state or not state.get("moved"):
            return
        idx = state["idx"]
        if idx < 0 or idx >= len(self.current_bbox_infos):
            return
        self.bbox_modified = True
        self._show_save_button_if_modified()
        if self.bbox_three_views_panel.isVisible() and hasattr(self, "raw_points") and self.raw_points is not None:
            self.bbox_three_views_panel.update_bbox(
                self.raw_points[:, :3],
                self.current_bbox_infos[idx],
                bbox_index=idx,
                on_bbox_edited=self._on_bbox_edited_from_panel,
                class_names=list(self.class_map.keys()),
            )
        self._show_bbox_attr_panel(idx, self.current_bbox_infos[idx])

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
        if "confidence" in info:
            lines.append("confidence: {}".format(info["confidence"] if info["confidence"] is not None else "-"))
        if "movement_state" in info:
            lines.append("movement_state: {}".format(info["movement_state"] if info["movement_state"] is not None else "-"))
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
                class_names=list(self.class_map.keys()),
            )
            self.bbox_three_views_panel.show()
            self.bbox_three_views_panel.setMinimumWidth(320)
            self._resize_main_splitter(three_width=380)
            self._show_bbox_attr_panel(bbox_index, info)

    def _on_bbox_edited_from_panel(self, bbox_index, new_info):
        """属性面板或三视图编辑后，同步更新 current_bbox_infos 与主 3D 视图。"""
        if bbox_index < 0 or bbox_index >= len(self.current_bbox_infos):
            return
        old_id = self.current_bbox_infos[bbox_index].get("id")
        new_id = new_info.get("id")
        if ("id" in new_info and
                self._bbox_id_key(new_id) != self._bbox_id_key(old_id) and
                self._is_bbox_id_used_by_other(bbox_index, new_id)):
            QMessageBox.warning(self, "ID重复", "ID {} 已被使用，请输入未占用的 ID。".format(new_info.get("id")))
            self._show_bbox_attr_panel(bbox_index, self.current_bbox_infos[bbox_index])
            return
        self.current_bbox_infos[bbox_index] = {**self.current_bbox_infos[bbox_index], **new_info}
        self._refresh_single_bbox_in_main_view(bbox_index)
        self._rebuild_link_arrows()  # 实时更新 link_id 弧线（中心点或尺寸变化会影响弧线起止点）
        self.bbox_modified = True
        self._show_save_button_if_modified()

    def _rotate_selected_bbox_yaw_90(self):
        if self.selected_bbox_index is None:
            self._set_status_message("请先选中一个目标框")
            return
        idx = self.selected_bbox_index
        if idx < 0 or idx >= len(self.current_bbox_infos):
            return
        info = self.current_bbox_infos[idx]
        info["yaw"] = self._normalize_yaw(float(info.get("yaw", 0.0) or 0.0) + np.pi / 2.0)
        info["l"], info["w"] = info.get("w", 1.0), info.get("l", 1.0)
        info.pop("arrow_yaw", None)
        self._refresh_single_bbox_in_main_view(idx)
        self._show_bbox_attr_panel(idx, info)
        if (self.bbox_three_views_panel.isVisible() and
                getattr(self.bbox_three_views_panel, "_bbox_index", None) == idx and
                hasattr(self, "raw_points") and self.raw_points is not None):
            self.bbox_three_views_panel.update_bbox(
                self.raw_points[:, :3],
                info,
                bbox_index=idx,
                on_bbox_edited=self._on_bbox_edited_from_panel,
                class_names=list(self.class_map.keys()),
            )
        self.bbox_modified = True
        self._show_save_button_if_modified()
        self._set_status_message("已将目标框 yaw 旋转 90 度")

    def _show_next_bbox_three_view(self):
        if self.selected_bbox_index is None:
            return
        if not self.current_bbox_infos or len(self.current_bbox_infos) < 2:
            return
        if not hasattr(self, "raw_points") or self.raw_points is None or len(self.raw_points) == 0:
            return
        current = self.selected_bbox_index
        if current < 0 or current >= len(self.current_bbox_infos):
            return
        next_index = (current + 1) % len(self.current_bbox_infos)
        self._show_bbox_info_dialog(next_index)

    def _normalize_yaw(self, yaw):
        return float((float(yaw) + np.pi) % (2.0 * np.pi) - np.pi)

    def _bbox_arrow_yaw(self, info):
        value = info.get("yaw", 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

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
        arrow_yaw = self._bbox_arrow_yaw(info)
        class_name = info.get("class_name", "")
        if class_name in self.class_map.keys():
            bbox_color = self.class_map[class_name]
        else:
            bbox_color = self.class_map["others"]
        color = QColor(bbox_color[0], bbox_color[1], bbox_color[2], bbox_color[3])
        is_selected = self.selected_bbox_index == bbox_index
        new_bbox = draw_bbox_solid(x, y, z, l, w, h, yaw, QColor(bbox_color[0], bbox_color[1], bbox_color[2], 80)) if is_selected else draw_bbox(x, y, z, l, w, h, yaw, color)
        new_arrow = draw_arrow(np.array([x, y, z + h/2]), [np.cos(arrow_yaw), np.sin(arrow_yaw), 0], l/2, color)
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
        if hasattr(self, "bbox_attr_panel"):
            self.bbox_attr_panel.clear_bbox()
            self.bbox_attr_panel.hide()
        self._resize_main_splitter(three_width=0)

    def _resize_main_splitter(self, point_width=None, three_width=None):
        """调整上方 splitter，不影响底部播放条。"""
        if not hasattr(self, "splitter"):
            return
        count = self.splitter.count()
        total = max(self.splitter.width(), 1)
        sizes = self.splitter.sizes()
        point_panel = getattr(self, "_point_select_dock", None)

        if count == 2:
            right_visible = self.bbox_three_views_panel.isVisible()
            right_w = three_width if three_width is not None else (sizes[1] if right_visible else 0)
            if not right_visible:
                right_w = 0
            self.splitter.setSizes([max(400, total - int(right_w)), int(right_w)])
            return

        if count != 3:
            return

        point_visible = point_panel is not None and point_panel.isVisible()
        right_visible = self.bbox_three_views_panel.isVisible()
        point_w = point_width if point_width is not None else (sizes[0] if point_visible else 0)
        right_w = three_width if three_width is not None else (sizes[2] if right_visible else 0)
        if not point_visible:
            point_w = 0
        if not right_visible:
            right_w = 0
        point_w = int(max(0, point_w))
        right_w = int(max(0, right_w))
        center_w = int(max(400, total - point_w - right_w))
        self.splitter.setSizes([point_w, center_w, right_w])

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
        if hasattr(self, "bbox_attr_panel"):
            self.bbox_attr_panel.clear_bbox()
            self.bbox_attr_panel.hide()
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
            self._set_status_message("框选模式：在视图中拖拽绘制矩形区域")
        else:
            self.box_select_start = None
            self.box_select_start_logical = None
            self.box_select_overlay.clear_rect()
            self._update_frame_info_label()

    # PointRectSelectMixin has moved the point-rectangle selection UI logic here.

    # Plane/Mask 功能已迁移到 PlaneMixin/MaskMixin（features/*_mixin.py）

    def _add_bbox_from_rect(self, x1, y1, x2, y2):
        """根据屏幕矩形框选的点，拟合 roll=0 pitch=0 的贴合包围框；无点时投影到地面生成框"""
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        pts_xyz = self.raw_points[:, :3] if (hasattr(self, "raw_points") and self.raw_points is not None and len(self.raw_points) > 0) else None
        selected = None
        if pts_xyz is not None:
            mask = points_in_screen_rect(self.glwidget, pts_xyz, x_min, x_max, y_min, y_max)
            selected = pts_xyz[mask]
            selected = filter_ground_points(selected)  # 平面拟合过滤地面点
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
        default_attrs = self._default_bbox_attr_values()
        default_attrs["id"] = self._next_unique_bbox_id()
        class_name = default_attrs.get("class_name") or "others"
        bbox_id = default_attrs.get("id")
        str_id = str(bbox_id) if bbox_id is not None else "XX"
        if class_name in self.class_map.keys():
            bbox_color = self.class_map[class_name]
        else:
            bbox_color = self.class_map["others"]
        color = QColor(bbox_color[0], bbox_color[1], bbox_color[2], bbox_color[3])
        bbox = draw_bbox(x_c, y_c, z_c, l, w, h, yaw, color)
        arrow_yaw = self._bbox_arrow_yaw({"yaw": yaw})
        arrow = draw_arrow(np.array([x_c, y_c, z_c + h/2]), [np.cos(arrow_yaw), np.sin(arrow_yaw), 0], l/2, color)
        vis_text = GLTextItem(text=class_name + "-" + str_id, pos=(x_c, y_c, z_c+1), color=color, font=QFont('Helvetica', 10))
        self.glwidget.addItem(bbox)
        self.glwidget.addItem(arrow)
        self.glwidget.addItem(vis_text)
        self.current_bbox_items.extend([bbox, arrow, vis_text])
        info = {
            "x": x_c, "y": y_c, "z": z_c,
            "l": l, "w": w, "h": h,
            "yaw": yaw, "roll": 0.0, "pitch": 0.0,
            "class_name": class_name,
            "confidence": None,
            "movement_state": None,
        }
        info.update(default_attrs)
        self.current_bbox_infos.append(info)
        self.selected_bbox_index = len(self.current_bbox_infos) - 1
        self._refresh_bbox_selection_style()
        self.box_select_mode = False
        self.box_select_action.setChecked(False)
        self._show_bbox_info_dialog(self.selected_bbox_index)
        if (not self.bbox_three_views_panel.isVisible() or
                getattr(self.bbox_three_views_panel, "_bbox_index", None) != self.selected_bbox_index):
            self._show_bbox_attr_panel(self.selected_bbox_index, info)
        self._set_status_message(f"已添加新框，共 {len(self.current_bbox_infos)} 个")
        self.bbox_modified = True
        self._show_save_button_if_modified()

    def _save_bboxes_clicked(self):
        """将修改后的框保存到原 JSON 文件"""
        self._save_current_bboxes(show_message=True)

    def _save_current_bboxes(self, show_message=False):
        """保存当前帧 bbox；返回是否保存成功。"""
        if not self._ensure_bbox_save_path():
            self._set_status_message("已取消选择目标框保存地址")
            return False
        try:
            save_bboxes_to_tanway_json(
                self.json_path,
                self.current_bbox_infos,
                self.original_json_agents,
            )
            self.bbox_modified = False
            self.save_bboxes_btn.hide()
            if hasattr(self, "bbox_attr_panel"):
                self.bbox_attr_panel.raise_()
            self._set_status_message(f"已保存到 {os.path.basename(self.json_path)}")
            return True
        except Exception as e:
            self._set_status_message(f"保存失败: {e}")
            if show_message:
                QMessageBox.warning(self, "保存失败", str(e))
            return False

    def _ensure_bbox_save_path(self):
        if hasattr(self, "json_path") and self.json_path:
            return True
        if not getattr(self, "pcd_file", None):
            QMessageBox.warning(self, "保存失败", "当前没有点云文件，无法生成目标框文件名")
            return False
        directory = QFileDialog.getExistingDirectory(self, "选择目标框保存地址")
        if not directory:
            return False
        self.bboxes_directory = directory
        self.json_path = os.path.join(str(self.bboxes_directory), str(Path(self.pcd_file).stem) + ".json")
        self.original_json_agents = load_json(self.json_path) if os.path.isfile(self.json_path) else None
        self._set_status_message(f"目标框将保存到 {os.path.basename(self.json_path)}")
        return True

    def _ensure_bboxes_directory(self):
        if getattr(self, "bboxes_directory", None):
            return True
        directory = QFileDialog.getExistingDirectory(self, "选择目标框保存地址")
        if not directory:
            return False
        self.bboxes_directory = directory
        if getattr(self, "pcd_file", None):
            self.json_path = os.path.join(str(self.bboxes_directory), str(Path(self.pcd_file).stem) + ".json")
            self.original_json_agents = load_json(self.json_path) if os.path.isfile(self.json_path) else None
        return True

    def _bbox_infos_from_agents(self, agents):
        json_data = get_anno_from_tanway_json(agents)
        infos = []
        for i, box in enumerate(json_data["bboxes"]):
            x, y, z, l, w, h, yaw = box
            class_name = json_data["className"][i]
            class_name = class_name.replace("TYPE_", "") if "TYPE_" in class_name else class_name
            info = {
                "x": x, "y": y, "z": z,
                "l": l, "w": w, "h": h,
                "yaw": yaw,
                "class_name": class_name,
            }
            if "tag" in json_data and i < len(json_data["tag"]) and isinstance(json_data["tag"][i], dict):
                reserved_tag_keys = {"link_id", "link_ID", "confidence", "movement_state", "arrow_yaw"}
                for key, value in json_data["tag"][i].items():
                    if key not in reserved_tag_keys:
                        info[key] = value
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
            infos.append(info)
        return infos

    def _clear_bbox_visual_items(self):
        for item in self.current_bbox_items:
            self.glwidget.removeItem(item)
        for item in getattr(self, "current_link_arrows", []):
            self.glwidget.removeItem(item)
        self.current_bbox_items = []
        self.current_link_arrows = []

    def _redraw_current_bboxes(self, reset_selection=True):
        self._clear_bbox_visual_items()
        selected_index = None if reset_selection else self.selected_bbox_index
        for info in self.current_bbox_infos:
            x, y, z = info["x"], info["y"], info["z"]
            l, w, h = info["l"], info["w"], info["h"]
            yaw = info["yaw"]
            arrow_yaw = self._bbox_arrow_yaw(info)
            class_name = info.get("class_name", "")
            bbox_color = self.class_map[class_name] if class_name in self.class_map.keys() else self.class_map["others"]
            color = QColor(bbox_color[0], bbox_color[1], bbox_color[2], bbox_color[3])
            is_selected = selected_index == len(self.current_bbox_items) // 3
            bbox = draw_bbox_solid(x, y, z, l, w, h, yaw, QColor(bbox_color[0], bbox_color[1], bbox_color[2], 80)) if is_selected else draw_bbox(x, y, z, l, w, h, yaw, color)
            arrow = draw_arrow(np.array([x, y, z+h/2]), direction=[np.cos(arrow_yaw), np.sin(arrow_yaw), 0], length=l/2, color=color)
            str_id = str(info.get("id", "")) if info.get("id") is not None else "XX"
            vis_text = GLTextItem(text=class_name + "-" + str_id, pos=(x, y, z+1), color=color, font=QFont('Helvetica', 10))
            self.glwidget.addItem(bbox)
            self.glwidget.addItem(arrow)
            self.glwidget.addItem(vis_text)
            self.current_bbox_items.extend([bbox, arrow, vis_text])
        if reset_selection:
            self.selected_bbox_index = None
        self._rebuild_link_arrows()
        if reset_selection and hasattr(self, "bbox_attr_panel"):
            self.bbox_attr_panel.clear_bbox()
            self.bbox_attr_panel.hide()

    def _copy_previous_frame_bboxes(self):
        if not getattr(self, "point_cloud_files", None) or self.current_frame_index <= 0:
            self._set_status_message("当前没有上一帧可复制")
            return
        if self.current_bbox_infos:
            ret = QMessageBox.question(
                self,
                "复制上一帧标注",
                "当前帧已有目标框，是否用上一帧标注覆盖当前帧？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return
        if not self._ensure_bboxes_directory():
            self._set_status_message("已取消选择目标框保存地址")
            return
        prev_pcd = self.point_cloud_files[self.current_frame_index - 1]
        prev_json_path = os.path.join(str(self.bboxes_directory), str(Path(prev_pcd).stem) + ".json")
        if not os.path.isfile(prev_json_path):
            QMessageBox.warning(self, "复制失败", f"上一帧标注文件不存在:\n{prev_json_path}")
            return
        try:
            prev_agents = load_json(prev_json_path)
            self.current_bbox_infos = self._bbox_infos_from_agents(prev_agents)
            self.original_json_agents = load_json(self.json_path) if self.json_path and os.path.isfile(self.json_path) else None
            self._redraw_current_bboxes()
            self.bbox_modified = True
            self._show_save_button_if_modified()
            self._set_status_message(f"已复制上一帧标注，共 {len(self.current_bbox_infos)} 个框")
        except Exception as e:
            QMessageBox.warning(self, "复制失败", str(e))

    def _save_current_bboxes_if_modified(self, force_save=False):
        if not getattr(self, "bbox_modified", False) and not force_save:
            return True
        if not self.current_bbox_infos and not getattr(self, "json_path", None) and not getattr(self, "bboxes_directory", None):
            self.bbox_modified = False
            if hasattr(self, "save_bboxes_btn"):
                self.save_bboxes_btn.hide()
            return True
        ok = self._save_current_bboxes(show_message=True)
        if not ok:
            self.timer.stop()
            if hasattr(self, "key_frame_timer"):
                self.key_frame_timer.stop()
                self._held_frame_key_direction = 0
            self.playing = False
            self.play_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/play_pcd.png')))
        return ok

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
                self._set_status_message("投影失败，无法生成框")
                return
            origin, direction = ray
            pt = ray_plane_z_intersect(origin, direction, z_plane)
            if pt is None:
                self._set_status_message("投影失败，无法生成框")
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
            self._set_frame_info_text(
                f"{self.current_frame_index + 1} / {len(self.point_cloud_files)} ({self.point_cloud_files[self.current_frame_index]})")
        elif self.point_cloud_files:
            self._set_frame_info_text(f"{self.current_frame_index + 1} / {len(self.point_cloud_files)}")
        else:
            self._set_frame_info_text("Point Cloud Viewer")

    def _set_frame_info_text(self, text):
        full_text = str(text)
        display_chars = getattr(self, "frame_info_display_chars", 35)
        if len(full_text) > display_chars:
            display_text = full_text[:max(0, display_chars - 3)] + "..."
        else:
            display_text = full_text
        self.frame_info_label.setText(display_text)
        self.frame_info_label.setToolTip(full_text)

    def _set_status_message(self, message):
        """将非帧播放信息写入底部右侧信息框。"""
        if hasattr(self, "log_info_edit"):
            self.log_info_edit.setText(str(message))

    def _set_toolbar_action_active(self, action, active):
        if action is None or not hasattr(self, "toolbar"):
            return
        btn = self.toolbar.widgetForAction(action)
        if btn is None:
            return
        if active:
            btn.setStyleSheet(
                "QToolButton { background-color: #2196F3; color: white; "
                "border-radius: 6px; padding: 6px 10px; }"
            )
        else:
            btn.setStyleSheet(
                "QToolButton { background-color: transparent; "
                "border-radius: 6px; padding: 6px 10px; }"
            )

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
            self._clear_session_extrinsic_offsets()
            self.point_cloud_files = natsorted([
                f for f in os.listdir(self.directory) if f.endswith('.txt') or f.endswith('.pcd')
            ])
            if not self.point_cloud_files:
                self.current_frame_index = -1
                self.frame_slider.blockSignals(True)
                self.frame_slider.setMaximum(0)
                self.frame_slider.setValue(0)
                self.frame_slider.blockSignals(False)
                self._set_status_message("所选目录中没有 .txt 或 .pcd 点云文件")
                QMessageBox.warning(self, "打开目录失败", "所选目录中没有 .txt 或 .pcd 点云文件")
                return
            self.current_frame_index = 0
            self.frame_slider.blockSignals(True)
            self.frame_slider.setMaximum(len(self.point_cloud_files) - 1)
            self.frame_slider.setValue(self.current_frame_index)
            self.frame_slider.blockSignals(False)
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
            if not self.point_cloud_files or self.current_frame_index < 0:
                self._set_status_message("已选择目标框目录，请先打开点云目录或点云文件")
                return
            self.load_frame()

    def open_history_frames_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择历史帧目录", self._default_history_frames_directory())
        if not directory:
            return
        self.history_frames_directory = directory
        self.history_frame_index = self._build_history_frame_index(directory)
        self.history_points_cache = {}
        self.history_single_points_cache = {}
        self.history_browse_index = 0
        self._hide_history_frames()
        count = sum(len(files) for files in self.history_frame_index.values())
        self._set_toolbar_action_active(self.open_history_frames_action, True)
        self._set_status_message("已导入历史帧目录：{}，匹配组 {} 个，文件 {} 个。按住 Shift 显示历史帧。".format(
            os.path.basename(directory), len(self.history_frame_index), count
        ))
        if QApplication.keyboardModifiers() & Qt.ShiftModifier:
            self.history_shift_down = True
            self._refresh_history_frame_visibility()

    def _toggle_history_browse_mode(self, checked=False):
        self._set_history_display_mode(bool(checked))

    def _set_history_display_mode(self, browse_enabled):
        self.history_display_mode = "browse" if browse_enabled else "overlay"
        self.history_browse_index = 0
        self._hide_history_frames()
        mode_text = "播放模式：按住 Shift 后左键下一帧、右键上一帧" if browse_enabled else "叠加模式：按住 Shift 显示所有历史帧"
        self._set_status_message("历史帧{}".format(mode_text))
        if self.history_shift_down:
            self._refresh_history_frame_visibility()

    def _default_history_frames_directory(self):
        candidates = []
        if getattr(self, "directory", None):
            candidates.append(os.path.join(os.path.dirname(self.directory), "multiframe_data"))
            candidates.append(self.directory)
        if getattr(self, "pcd_file", None):
            parent = os.path.dirname(self.pcd_file)
            candidates.append(os.path.join(os.path.dirname(parent), "multiframe_data"))
            candidates.append(parent)
        for path in candidates:
            if path and os.path.isdir(path):
                return path
        return ""

    def _build_history_frame_index(self, directory):
        index = {}
        if not directory or not os.path.isdir(directory):
            return index
        exts = (".pcd", ".txt")
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if os.path.isdir(path):
                files = [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.lower().endswith(exts)
                ]
                if files:
                    index[name] = natsorted(files)
            elif os.path.isfile(path) and name.lower().endswith(exts):
                index.setdefault(Path(name).stem, []).append(path)
        for key, files in list(index.items()):
            index[key] = natsorted(files)
        return index

    def _current_history_key(self):
        if getattr(self, "pcd_file", None):
            return Path(self.pcd_file).stem
        if self.point_cloud_files and 0 <= self.current_frame_index < len(self.point_cloud_files):
            return Path(self.point_cloud_files[self.current_frame_index]).stem
        return None

    def _current_history_files(self):
        key = self._current_history_key()
        if not key or not self.history_frame_index:
            return []
        return list(self.history_frame_index.get(key, []))

    def _load_current_history_points(self):
        key = self._current_history_key()
        if not key:
            return None
        if key in self.history_points_cache:
            return self.history_points_cache[key]
        files = self._current_history_files()
        if not files:
            self.history_points_cache[key] = None
            return None
        chunks = []
        for path in files:
            points = self._load_history_points_file(path)
            if points is not None and len(points) > 0:
                chunks.append(points)
        merged = np.vstack(chunks) if chunks else None
        self.history_points_cache[key] = merged
        return merged

    def _load_history_points_file(self, path):
        if path in self.history_single_points_cache:
            return self.history_single_points_cache[path]
        try:
            points, _, _ = get_points_from_pcd_file(path)
        except Exception as exc:
            print("load history frame failed:", path, exc)
            self.history_single_points_cache[path] = None
            return None
        if points is None or len(points) == 0:
            self.history_single_points_cache[path] = None
            return None
        xyz = np.asarray(points[:, :3], dtype=np.float32)
        self.history_single_points_cache[path] = xyz
        return xyz

    def _hide_main_frame_scatter_for_history(self):
        if self.history_main_scatter_hidden:
            return
        scatter = getattr(self, "scatter", None)
        if scatter is None:
            return
        try:
            items = getattr(self.glwidget, "items", None)
            if items is None or scatter in items:
                self.glwidget.removeItem(scatter)
                self.history_main_scatter_hidden = True
        except ValueError:
            self.history_main_scatter_hidden = True

    def _restore_main_frame_scatter_after_history(self):
        if not self.history_main_scatter_hidden:
            return
        scatter = getattr(self, "scatter", None)
        if scatter is not None:
            items = getattr(self.glwidget, "items", None)
            if items is None or scatter not in items:
                self.glwidget.addItem(scatter)
        self.history_main_scatter_hidden = False

    def _hide_history_frames(self, restore_main=True):
        if self.history_scatter is None:
            if restore_main:
                self._restore_main_frame_scatter_after_history()
            return
        try:
            items = getattr(self.glwidget, "items", None)
            if items is None or self.history_scatter in items:
                self.glwidget.removeItem(self.history_scatter)
        except ValueError:
            pass
        self.history_scatter = None
        if restore_main:
            self._restore_main_frame_scatter_after_history()

    def _refresh_history_frame_visibility(self):
        self._hide_history_frames(restore_main=False)
        if not self.history_shift_down or not self.history_frame_index:
            self._restore_main_frame_scatter_after_history()
            return
        files = self._current_history_files()
        if not files:
            key = self._current_history_key()
            self._restore_main_frame_scatter_after_history()
            self._set_status_message("当前帧无匹配历史帧：{}".format(key or "-"))
            return
        if self.history_display_mode == "browse":
            self._show_history_browse_frame(files)
            return
        points = self._load_current_history_points()
        if points is None or len(points) == 0:
            self._restore_main_frame_scatter_after_history()
            self._set_status_message("当前帧历史帧为空或加载失败")
            return
        rgba = np.tile(np.array([[1.0, 0.62, 0.05, 0.38]], dtype=np.float32), (len(points), 1))
        size = max(float(self.point_size) * 0.75, 0.5)
        self.history_scatter = GLScatterPlotItem(pos=points, color=rgba, size=size)
        self.glwidget.addItem(self.history_scatter)
        self._set_status_message("已叠加显示 {} 帧历史帧，共 {} 个点".format(len(files), len(points)))

    def _show_history_browse_frame(self, files):
        if not files:
            return
        self._hide_main_frame_scatter_for_history()
        self.history_browse_index %= len(files)
        path = files[self.history_browse_index]
        points = self._load_history_points_file(path)
        if points is None or len(points) == 0:
            self._set_status_message("历史帧加载失败：{}".format(os.path.basename(path)))
            return
        rgba = np.tile(np.array([[1.0, 0.62, 0.05, 0.95]], dtype=np.float32), (len(points), 1))
        self.history_scatter = GLScatterPlotItem(pos=points, color=rgba, size=self.point_size)
        self.glwidget.addItem(self.history_scatter)
        self._set_status_message("历史帧浏览 {}/{}：{}，左键下一帧，右键上一帧".format(
            self.history_browse_index + 1, len(files), os.path.basename(path)
        ))

    def _step_history_browse_frame(self, delta):
        files = self._current_history_files()
        if not files:
            return
        self.history_browse_index = (self.history_browse_index + delta) % len(files)
        self._refresh_history_frame_visibility()

    def _is_history_browse_active(self):
        return (
            self.history_shift_down and
            self.history_display_mode == "browse" and
            bool(self._current_history_files())
        )

    def open_file(self):
        self.timer.stop()
        self.playing = False
        self.play_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/play_pcd.png')))
        self.pcd_file, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Pcd Files (*.pcd)", options=QFileDialog.Options())
        self.colors = QColor(0, 0, 255).getRgbF()
        self._user_solid_rgbf = self.colors
        if self.pcd_file:
            self._clear_session_extrinsic_offsets()
            self.directory = None
            self.point_cloud_files = []
            self.current_frame_index = -1
            self.raw_points, self.structured_points, metadata = get_points_from_pcd_file(self.pcd_file)
            self.metadata = metadata
            self._points_rect_select_mask = None
            if self._point_select_dock is not None:
                self._reset_point_select_table_ui()
            self._extrinsic_after_load_frame()
            self.vis_fram(updata_color_bar=True)
            self._set_topdown_view()

    def _point_size_step(self):
        size = float(self.point_size)
        if size < 3.0:
            return 0.25
        return 0.5

    def _set_point_size(self, size):
        self.point_size = round(max(1.0, float(size)), 2)
        self.vis_fram()
        self._set_status_message("点大小: {:.2f}".format(self.point_size))

    def increase_points_size(self):
        self._set_point_size(float(self.point_size) + self._point_size_step())

    def decrease_points_size(self):
        if float(self.point_size) <= 1.0:
            self._set_status_message("点大小已是最小值: 1.00")
            return
        self._set_point_size(float(self.point_size) - self._point_size_step())

    def load_frame(self):
        import time
        start_time = time.time()

        if (not self.point_cloud_files or self.directory is None or
                self.current_frame_index < 0 or
                self.current_frame_index >= len(self.point_cloud_files)):
            self._set_status_message("当前没有可加载的点云帧")
            return False
        self.pcd_file = os.path.join(self.directory, self.point_cloud_files[self.current_frame_index])
        self.raw_points, self.structured_points, metadata = get_points_from_pcd_file(self.pcd_file)
        self._points_rect_select_mask = None
        if self._point_select_dock is not None:
            self._reset_point_select_table_ui()
        metadata_changed = metadata != self.metadata
        if metadata_changed:
            self.metadata = metadata
        self._extrinsic_after_load_frame()
        if self.color_fields is not None:
            print(self.color_fields)
            self.colors = self.Colors[0](self.min_max_normalization(self.structured_points[self.color_fields]))

        self.vis_fram(updata_color_bar=metadata_changed)
        end1_time = time.time()
        self._set_frame_info_text(
            f"{self.current_frame_index + 1} / {len(self.point_cloud_files)} ({self.point_cloud_files[self.current_frame_index]})")
        self.frame_slider.setValue(self.current_frame_index)
        self._update_save_button_geometry()

        elapsed1_time = (end1_time - start_time) * 1000  # Calculate time in milliseconds
        print(f"Code execution time: {elapsed1_time:.3f} ms", id)

        if elapsed1_time < 100:
            time.sleep((100-elapsed1_time)/1000)
        return True



    def min_max_normalization(self, matrix):
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        return normalized_matrix

    def previous_frame(self):
        if self._is_history_browse_active():
            self._step_history_browse_frame(-1)
            return
        if self.current_frame_index > 0:
            self._change_frame(self.current_frame_index - 1, force_save=True)

    def next_frame(self):
        if self._is_history_browse_active():
            self._step_history_browse_frame(1)
            return
        if self.current_frame_index < len(self.point_cloud_files) - 1:
            force_save = self.sender() is self.next_button
            self._change_frame(self.current_frame_index + 1, force_save=force_save)
        else:
            self.timer.stop()
            self.playing = False
            self.play_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/play_pcd.png')))

    def _handle_frame_key_event(self, event):
        if event.type() not in (QEvent.KeyPress, QEvent.KeyRelease):
            return False
        if event.key() not in (Qt.Key_Z, Qt.Key_X):
            return False
        if self._frame_key_should_be_ignored():
            if event.type() == QEvent.KeyRelease:
                self._stop_frame_key_playback()
            return False
        if event.isAutoRepeat():
            return True
        if event.type() == QEvent.KeyPress:
            direction = -1 if event.key() == Qt.Key_Z else 1
            self._start_frame_key_playback(direction)
        else:
            self._stop_frame_key_playback()
        return True

    def _frame_key_should_be_ignored(self):
        focus = QApplication.focusWidget()
        if focus is None or focus is self.glwidget:
            return False
        return isinstance(focus, (QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox))

    def _start_frame_key_playback(self, direction):
        if direction not in (-1, 1):
            return
        self.timer.stop()
        self.playing = False
        self.play_button.setIcon(QIcon(os.path.join(self.curpath, 'icons/play_pcd.png')))
        self._held_frame_key_direction = direction
        self._step_held_frame_key()
        if self._held_frame_key_direction != 0:
            self.key_frame_timer.start(100)

    def _stop_frame_key_playback(self):
        self.key_frame_timer.stop()
        self._held_frame_key_direction = 0

    def _step_held_frame_key(self):
        if self._held_frame_key_direction < 0:
            if self._is_history_browse_active():
                self._step_history_browse_frame(-1)
            elif self.current_frame_index > 0:
                if not self._change_frame(self.current_frame_index - 1, force_save=False):
                    self._stop_frame_key_playback()
            else:
                self._stop_frame_key_playback()
        elif self._held_frame_key_direction > 0:
            if self._is_history_browse_active():
                self._step_history_browse_frame(1)
            elif self.current_frame_index < len(self.point_cloud_files) - 1:
                if not self._change_frame(self.current_frame_index + 1, force_save=False):
                    self._stop_frame_key_playback()
            else:
                self._stop_frame_key_playback()

    def _change_frame(self, target_index, force_save=False):
        if target_index == self.current_frame_index:
            return False
        if target_index < 0 or target_index >= len(self.point_cloud_files):
            return False
        if not self._save_current_bboxes_if_modified(force_save=force_save):
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_index)
            self.frame_slider.blockSignals(False)
            return False
        self.current_frame_index = target_index
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_index)
        self.frame_slider.blockSignals(False)
        self.load_frame()
        return True

    def toggle_play_pause(self):
        if self._is_history_browse_active():
            self._step_history_browse_frame(1)
            return
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
            self._change_frame(self.frame_slider.value())
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

    def vis_fram(self, updata_color_bar=False, preserve_current_bboxes=False):
        # 兜底：程序启动时可能只是用文字点云初始化（此时 structured_points 可能不存在）
        if not hasattr(self, "structured_points"):
            self.structured_points = None
        if not hasattr(self, "metadata"):
            self.metadata = None
        self._hide_history_frames()
        preserved_bbox_infos = [dict(info) for info in self.current_bbox_infos] if preserve_current_bboxes else None
        preserved_selected_bbox_index = self.selected_bbox_index if preserve_current_bboxes else None
        preserved_bbox_modified = self.bbox_modified if preserve_current_bboxes else False
        for item in self.current_bbox_items:
            self.glwidget.removeItem(item)
        for item in getattr(self, "current_link_arrows", []):
            self.glwidget.removeItem(item)
        self.current_bbox_items = []
        self.current_link_arrows = []
        if preserve_current_bboxes:
            self.current_bbox_infos = preserved_bbox_infos
            self.selected_bbox_index = preserved_selected_bbox_index
            self.bbox_modified = preserved_bbox_modified
        else:
            self.current_bbox_infos = []
            self.selected_bbox_index = None
            self.bbox_modified = False
            self.save_bboxes_btn.hide()
            if hasattr(self, "bbox_attr_panel"):
                self.bbox_attr_panel.clear_bbox()
                self.bbox_attr_panel.hide()
        self._update_save_button_geometry()
        self._clear_cluster_bboxes()
        if preserve_current_bboxes:
            self._redraw_current_bboxes(reset_selection=False)
            self._show_save_button_if_modified()
        if not preserve_current_bboxes and self.bboxes_directory is not None:
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
                    if "tag" in json_data and i < len(json_data["tag"]) and isinstance(json_data["tag"][i], dict):
                        reserved_tag_keys = {"link_id", "link_ID", "confidence", "movement_state", "arrow_yaw"}
                        for key, value in json_data["tag"][i].items():
                            if key not in reserved_tag_keys:
                                info[key] = value
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
                    class_names=list(self.class_map.keys()),
                )
                self._show_bbox_attr_panel(idx, self.current_bbox_infos[idx])
            else:
                self.bbox_three_views_panel.hide()
                if hasattr(self, "bbox_attr_panel"):
                    self.bbox_attr_panel.clear_bbox()
                    self.bbox_attr_panel.hide()

        # GLViewWidget.removeItem 在 item 不存在于其 internal list 时会抛 ValueError，
        # 例如 Mask 设置界面实时回调过程中 vis_fram 被多次触发。
        if self.scatter is not None:
            try:
                items = getattr(self.glwidget, "items", None)
                if items is not None:
                    if self.scatter in items:
                        self.glwidget.removeItem(self.scatter)
                else:
                    # 兜底：尝试移除
                    self.glwidget.removeItem(self.scatter)
            except ValueError:
                # 已不在列表中，忽略即可
                pass

        self.points = self.raw_points[:, :3]
        keep_inside_mask = self._mask_keep_inside_points(self.points)
        if len(keep_inside_mask) == len(self.points):
            self.points = self.points[keep_inside_mask]

        structured_for_display = getattr(self, "structured_points", None)
        if structured_for_display is not None and len(keep_inside_mask) == len(structured_for_display):
            structured_for_display = structured_for_display[keep_inside_mask]

        extrinsic_mode = getattr(self, "_extrinsic_calib_mode", False)
        mask_for_extrinsic = None
        if hasattr(self, "raw_points") and self.raw_points is not None:
            full_xyz = self.raw_points[:, :3]
            mask_for_extrinsic = self._mask_keep_inside_points(full_xyz)

        if extrinsic_mode:
            pos, rgba = self._extrinsic_build_pos_rgba(mask_for_extrinsic)
            self.points = pos if pos is not None else np.empty((0, 3))
            if rgba is None:
                rgba = np.empty((0, 4))
            self.scatter = GLScatterPlotItem(pos=self.points, color=rgba, size=self.point_size)
            self.colors = rgba
            self.glwidget.addItem(self.scatter)
            if updata_color_bar:
                self.update_color_sidebar()
            self._refresh_cluster_if_enabled()
            self._refresh_history_frame_visibility()
            return

        if self.color_fields is not None:
            if len(structured_for_display) > 0 and max(structured_for_display[self.color_fields]) >= 0:
                unique_values = np.unique(structured_for_display[self.color_fields])
                num_unique_values = len(unique_values)
                print(unique_values)
                print(type(unique_values[0]))
                if all(isinstance(x, np.int32) for x in unique_values)  and max(unique_values) < 16 and min(unique_values)>=0:
                    color_map = {}
                    for i, value in enumerate(unique_values):
                        color_map[value] =self.colors_16[value]
                    self.colors = np.array([color_map[val] for val in structured_for_display[self.color_fields]])

                elif num_unique_values <= 16:
                    color_map = {}
                    for i, value in enumerate(unique_values):
                        color_map[value] =self.colors_16[i]
                    self.colors = np.array([color_map[val] for val in structured_for_display[self.color_fields]])
                else:
                    self.colors = self.Colors[0](self.min_max_normalization(structured_for_display[self.color_fields]))
            else:
                # 过滤后无点时兜底颜色，避免沿用上一帧颜色长度不匹配
                self.colors = getattr(self, "_user_solid_rgbf", QColor(0, 0, 255).getRgbF())
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
            # 若开启“仅保留圈内点”，self.points 会被 keep_inside_mask 过滤，
            # 此时框选 mask 仍基于 raw_points 全量长度，需要同步过滤后再应用颜色高亮。
            m_use = None
            if len(m) == len(self.points):
                m_use = m
            elif "keep_inside_mask" in locals() and len(m) == len(keep_inside_mask):
                m_use = np.asarray(m, dtype=bool)[keep_inside_mask]
            else:
                m_use = None

            if m_use is not None and np.any(m_use):
                rgba = rgba.copy()
                rgba[np.asarray(m_use, dtype=bool)] = (1.0, 0.0, 0.0, 1.0)
        cm = getattr(self, "_cluster_select_mask", None)
        if cm is not None:
            cm_use = None
            if len(cm) == len(self.points):
                cm_use = cm
            elif "keep_inside_mask" in locals() and len(cm) == len(keep_inside_mask):
                cm_use = np.asarray(cm, dtype=bool)[keep_inside_mask]
            else:
                cm_use = None
            if cm_use is not None and np.any(cm_use):
                if not isinstance(rgba, np.ndarray):
                    rgba = self._colors_to_rgba_n4(rgba, len(self.points))
                rgba = np.asarray(rgba, dtype=np.float32).copy()
                rgba[np.asarray(cm_use, dtype=bool)] = (0.2, 1.0, 0.2, 1.0)
        self.scatter = GLScatterPlotItem(pos=self.points, color=rgba, size=self.point_size)
        self.colors = rgba
        self.glwidget.addItem(self.scatter)
        if updata_color_bar:
            self.update_color_sidebar()
        self._refresh_cluster_if_enabled()
        self._refresh_history_frame_visibility()

    def _open_cluster_dialog(self):
        """打开当前帧点云聚类参数窗口。"""
        if not hasattr(self, "raw_points") or self.raw_points is None or len(self.raw_points) == 0:
            QMessageBox.information(self, "点云聚类", "当前没有可聚类的点云。")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("当前帧点云聚类参数")
        dlg.setMinimumWidth(520)
        form = QFormLayout(dlg)
        form.setLabelAlignment(Qt.AlignRight)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        params = getattr(self, "_cluster_params", {})

        use_lshape_check = QCheckBox("启用L-shape拟合", dlg)
        use_lshape_check.setChecked(bool(params.get("use_lshape", False)))
        use_roi_check = QCheckBox("启用 ROI 过滤", dlg)
        use_roi_check.setChecked(bool(params.get("use_roi", True)))
        use_size_filter_check = QCheckBox("启用框尺寸筛选(l/w/h)", dlg)
        use_size_filter_check.setChecked(bool(params.get("use_size_filter", False)))

        eps_spin = QDoubleSpinBox(dlg)
        eps_spin.setRange(0.01, 20.0)
        eps_spin.setDecimals(2)
        eps_spin.setSingleStep(0.05)
        eps_spin.setValue(float(params.get("eps", 0.5)))

        min_points_spin = QSpinBox(dlg)
        min_points_spin.setRange(3, 100000)
        min_points_spin.setSingleStep(10)
        min_points_spin.setValue(int(params.get("min_points", 5)))

        max_points_spin = QSpinBox(dlg)
        max_points_spin.setRange(10, 200000)
        max_points_spin.setSingleStep(1000)
        max_points_spin.setValue(int(params.get("max_points", 20000)))
        voxel_size_spin = QDoubleSpinBox(dlg)
        voxel_size_spin.setRange(0.0, 5.0)
        voxel_size_spin.setDecimals(2)
        voxel_size_spin.setSingleStep(0.05)
        voxel_size_spin.setValue(float(params.get("voxel_size", 0.1)))

        roi_x_min_spin = QDoubleSpinBox(dlg)
        roi_x_min_spin.setRange(-100000.0, 100000.0)
        roi_x_min_spin.setDecimals(2)
        roi_x_min_spin.setSingleStep(0.5)
        roi_x_min_spin.setValue(float(params.get("roi_x_min", -100.0)))

        roi_x_max_spin = QDoubleSpinBox(dlg)
        roi_x_max_spin.setRange(-100000.0, 100000.0)
        roi_x_max_spin.setDecimals(2)
        roi_x_max_spin.setSingleStep(0.5)
        roi_x_max_spin.setValue(float(params.get("roi_x_max", 100.0)))

        roi_y_min_spin = QDoubleSpinBox(dlg)
        roi_y_min_spin.setRange(-100000.0, 100000.0)
        roi_y_min_spin.setDecimals(2)
        roi_y_min_spin.setSingleStep(0.5)
        roi_y_min_spin.setValue(float(params.get("roi_y_min", -100.0)))

        roi_y_max_spin = QDoubleSpinBox(dlg)
        roi_y_max_spin.setRange(-100000.0, 100000.0)
        roi_y_max_spin.setDecimals(2)
        roi_y_max_spin.setSingleStep(0.5)
        roi_y_max_spin.setValue(float(params.get("roi_y_max", 100.0)))

        roi_z_min_spin = QDoubleSpinBox(dlg)
        roi_z_min_spin.setRange(-100000.0, 100000.0)
        roi_z_min_spin.setDecimals(2)
        roi_z_min_spin.setSingleStep(0.2)
        roi_z_min_spin.setValue(float(params.get("roi_z_min", -1.5)))

        roi_z_max_spin = QDoubleSpinBox(dlg)
        roi_z_max_spin.setRange(-100000.0, 100000.0)
        roi_z_max_spin.setDecimals(2)
        roi_z_max_spin.setSingleStep(0.2)
        roi_z_max_spin.setValue(float(params.get("roi_z_max", 3.0)))

        l_min_spin = QDoubleSpinBox(dlg)
        l_min_spin.setRange(0.0, 10000.0)
        l_min_spin.setDecimals(2)
        l_min_spin.setSingleStep(0.1)
        l_min_spin.setValue(float(params.get("l_min", 0.0)))
        l_max_spin = QDoubleSpinBox(dlg)
        l_max_spin.setRange(0.0, 10000.0)
        l_max_spin.setDecimals(2)
        l_max_spin.setSingleStep(0.1)
        l_max_spin.setValue(float(params.get("l_max", 1000.0)))
        w_min_spin = QDoubleSpinBox(dlg)
        w_min_spin.setRange(0.0, 10000.0)
        w_min_spin.setDecimals(2)
        w_min_spin.setSingleStep(0.1)
        w_min_spin.setValue(float(params.get("w_min", 0.0)))
        w_max_spin = QDoubleSpinBox(dlg)
        w_max_spin.setRange(0.0, 10000.0)
        w_max_spin.setDecimals(2)
        w_max_spin.setSingleStep(0.1)
        w_max_spin.setValue(float(params.get("w_max", 1000.0)))
        h_min_spin = QDoubleSpinBox(dlg)
        h_min_spin.setRange(0.0, 10000.0)
        h_min_spin.setDecimals(2)
        h_min_spin.setSingleStep(0.1)
        h_min_spin.setValue(float(params.get("h_min", 0.0)))
        h_max_spin = QDoubleSpinBox(dlg)
        h_max_spin.setRange(0.0, 10000.0)
        h_max_spin.setDecimals(2)
        h_max_spin.setSingleStep(0.1)
        h_max_spin.setValue(float(params.get("h_max", 1000.0)))

        basic_group = QGroupBox("聚类设置", dlg)
        basic_grid = QGridLayout(basic_group)
        basic_grid.addWidget(use_lshape_check, 0, 0, 1, 2)
        basic_grid.addWidget(QLabel("聚类半径 eps", dlg), 1, 0)
        basic_grid.addWidget(eps_spin, 1, 1)
        basic_grid.addWidget(QLabel("最小聚类点数", dlg), 2, 0)
        basic_grid.addWidget(min_points_spin, 2, 1)
        basic_grid.addWidget(QLabel("最大参与点数", dlg), 3, 0)
        basic_grid.addWidget(max_points_spin, 3, 1)
        basic_grid.addWidget(QLabel("体素下采样大小", dlg), 4, 0)
        basic_grid.addWidget(voxel_size_spin, 4, 1)
        basic_grid.setColumnStretch(0, 2)
        basic_grid.setColumnStretch(1, 3)

        roi_group = QGroupBox("ROI 设置", dlg)
        roi_grid = QGridLayout(roi_group)
        roi_grid.addWidget(use_roi_check, 0, 0, 1, 4)
        roi_grid.addWidget(QLabel("X最小", dlg), 1, 0)
        roi_grid.addWidget(roi_x_min_spin, 1, 1)
        roi_grid.addWidget(QLabel("X最大", dlg), 1, 2)
        roi_grid.addWidget(roi_x_max_spin, 1, 3)
        roi_grid.addWidget(QLabel("Y最小", dlg), 2, 0)
        roi_grid.addWidget(roi_y_min_spin, 2, 1)
        roi_grid.addWidget(QLabel("Y最大", dlg), 2, 2)
        roi_grid.addWidget(roi_y_max_spin, 2, 3)
        roi_grid.addWidget(QLabel("Z最小", dlg), 3, 0)
        roi_grid.addWidget(roi_z_min_spin, 3, 1)
        roi_grid.addWidget(QLabel("Z最大", dlg), 3, 2)
        roi_grid.addWidget(roi_z_max_spin, 3, 3)

        size_group = QGroupBox("框尺寸筛选", dlg)
        size_grid = QGridLayout(size_group)
        size_grid.addWidget(use_size_filter_check, 0, 0, 1, 4)
        size_grid.addWidget(QLabel("L最小", dlg), 1, 0)
        size_grid.addWidget(l_min_spin, 1, 1)
        size_grid.addWidget(QLabel("L最大", dlg), 1, 2)
        size_grid.addWidget(l_max_spin, 1, 3)
        size_grid.addWidget(QLabel("W最小", dlg), 2, 0)
        size_grid.addWidget(w_min_spin, 2, 1)
        size_grid.addWidget(QLabel("W最大", dlg), 2, 2)
        size_grid.addWidget(w_max_spin, 2, 3)
        size_grid.addWidget(QLabel("H最小", dlg), 3, 0)
        size_grid.addWidget(h_min_spin, 3, 1)
        size_grid.addWidget(QLabel("H最大", dlg), 3, 2)
        size_grid.addWidget(h_max_spin, 3, 3)

        form.addRow(basic_group)
        form.addRow(roi_group)
        form.addRow(size_group)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        # 记住上次参数，下次打开沿用
        self._cluster_params = {
            "eps": float(eps_spin.value()),
            "min_points": int(min_points_spin.value()),
            "max_points": int(max_points_spin.value()),
            "voxel_size": float(voxel_size_spin.value()),
            "use_lshape": bool(use_lshape_check.isChecked()),
            "use_roi": bool(use_roi_check.isChecked()),
            "roi_x_min": float(roi_x_min_spin.value()),
            "roi_x_max": float(roi_x_max_spin.value()),
            "roi_y_min": float(roi_y_min_spin.value()),
            "roi_y_max": float(roi_y_max_spin.value()),
            "roi_z_min": float(roi_z_min_spin.value()),
            "roi_z_max": float(roi_z_max_spin.value()),
            "use_size_filter": bool(use_size_filter_check.isChecked()),
            "l_min": float(l_min_spin.value()),
            "l_max": float(l_max_spin.value()),
            "w_min": float(w_min_spin.value()),
            "w_max": float(w_max_spin.value()),
            "h_min": float(h_min_spin.value()),
            "h_max": float(h_max_spin.value()),
        }
        self._cluster_enabled = True

        self._cluster_current_frame()
        if hasattr(self, "_cluster_action"):
            self._cluster_action.setChecked(True)
            self._set_toolbar_action_active(self._cluster_action, True)

    def _disable_cluster(self):
        """关闭聚类效果并清理聚类框/高亮。"""
        self._cluster_enabled = False
        self._selected_cluster_bbox_index = None
        self._cluster_select_mask = None
        self._clear_cluster_bboxes()
        self.vis_fram(preserve_current_bboxes=True)
        self._set_status_message("已关闭点云聚类")
        if hasattr(self, "_cluster_action"):
            self._cluster_action.setChecked(False)
            self._set_toolbar_action_active(self._cluster_action, False)

    def _toggle_cluster_from_toolbar(self, checked=None):
        """工具栏聚类按钮：单击启用聚类，再次单击关闭聚类。"""
        if not hasattr(self, "_cluster_action"):
            return
        checked = self._cluster_action.isChecked() if checked is None else bool(checked)
        if checked:
            # 打开参数框并执行聚类；若取消则回滚按钮状态
            prev_enabled = self._cluster_enabled
            self._open_cluster_dialog()
            if not self._cluster_enabled:
                self._cluster_action.setChecked(prev_enabled)
                self._set_toolbar_action_active(self._cluster_action, prev_enabled)
        else:
            self._disable_cluster()

    def _clear_cluster_bboxes(self):
        """清理聚类绘制的包围框。"""
        for item in getattr(self, "_cluster_bbox_items", []):
            try:
                self.glwidget.removeItem(item)
            except ValueError:
                pass
        self._cluster_bbox_items = []
        self._cluster_bbox_infos = []

    def _cluster_current_frame(self):
        """对当前帧点云做 DBSCAN 聚类并绘制 bbox(x,y,z,l,w,h,yaw)。"""
        if not hasattr(self, "raw_points") or self.raw_points is None or len(self.raw_points) == 0:
            self._set_status_message("当前帧点云为空")
            return
        params = dict(getattr(self, "_cluster_params", {}))
        self._clear_cluster_bboxes()
        self._set_status_message("聚类中...")
        QApplication.processEvents()
        cluster_points = self._points_for_cluster()
        if cluster_points is None or len(cluster_points) == 0:
            self._set_status_message("当前可聚类点云为空")
            self._cluster_select_mask = None
            self._selected_cluster_bbox_index = None
            return
        try:
            boxes, roi_mask = self._obstacle_cluster.cluster(cluster_points, params)
        except Exception as e:
            self._set_status_message(f"聚类失败: {e}")
            self._cluster_select_mask = None
            self._selected_cluster_bbox_index = None
            return

        if len(roi_mask) > 0 and params.get("use_roi", True) and not np.any(roi_mask):
            self._set_status_message("ROI 区域内没有点云，无法聚类")
            self._cluster_select_mask = None
            self._selected_cluster_bbox_index = None
            return
        if not boxes:
            self._set_status_message("未检测到有效聚类")
            self._cluster_select_mask = None
            self._selected_cluster_bbox_index = None
            return

        for box in boxes:
            color = QColor(255, 165, 0, 220)
            bbox_item = draw_bbox(
                float(box["x"]),
                float(box["y"]),
                float(box["z"]),
                float(box["l"]),
                float(box["w"]),
                float(box["h"]),
                float(box["yaw"]),
                color,
            )
            self.glwidget.addItem(bbox_item)
            self._cluster_bbox_items.append(bbox_item)
            self._cluster_bbox_infos.append(dict(box))

        if self._selected_cluster_bbox_index is not None and self._selected_cluster_bbox_index < len(self._cluster_bbox_infos):
            self._update_cluster_select_mask_from_selected()
        else:
            self._selected_cluster_bbox_index = None
            self._cluster_select_mask = None

        stats = getattr(self._obstacle_cluster, "last_stats", {}) or {}
        limited_text = "，已限流" if stats.get("limited") else ""
        self._set_status_message(
            "聚类完成：{} 个包围框；ROI {} 点，下采样 {} 点，参与 {} 点{}".format(
                len(self._cluster_bbox_items),
                stats.get("roi_points", "-"),
                stats.get("downsampled_points", "-"),
                stats.get("cluster_points", "-"),
                limited_text,
            )
        )

    def _points_for_cluster(self):
        pts = np.asarray(self.raw_points[:, :3], dtype=np.float64)
        keep_mask = self._mask_keep_inside_points(pts)
        if len(keep_mask) == len(pts):
            pts = pts[keep_mask]
        return pts

    def _refresh_cluster_if_enabled(self):
        """若已启用聚类，则在当前帧按最新参数自动刷新聚类框。"""
        if not getattr(self, "_cluster_enabled", False):
            return
        if not hasattr(self, "raw_points") or self.raw_points is None or len(self.raw_points) == 0:
            self._clear_cluster_bboxes()
            return
        params = getattr(self, "_cluster_params", None)
        if not params:
            return
        self._cluster_current_frame()

    def _update_cluster_select_mask_from_selected(self):
        """根据当前选中的聚类框更新 raw_points 维度的点掩码。"""
        idx = self._selected_cluster_bbox_index
        if idx is None or idx < 0 or idx >= len(self._cluster_bbox_infos):
            self._cluster_select_mask = None
            return
        if not hasattr(self, "raw_points") or self.raw_points is None or len(self.raw_points) == 0:
            self._cluster_select_mask = None
            return
        info = self._cluster_bbox_infos[idx]
        pts = np.asarray(self.raw_points[:, :3], dtype=np.float64)
        x_c, y_c, z_c = info["x"], info["y"], info["z"]
        l, w, h, yaw = info["l"], info["w"], info["h"], info["yaw"]
        c = np.cos(-yaw)
        s = np.sin(-yaw)
        dx = pts[:, 0] - x_c
        dy = pts[:, 1] - y_c
        dz = pts[:, 2] - z_c
        local_x = c * dx - s * dy
        local_y = s * dx + c * dy
        self._cluster_select_mask = (
            (np.abs(local_x) <= l / 2.0) &
            (np.abs(local_y) <= w / 2.0) &
            (np.abs(dz) <= h / 2.0)
        )


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
    if sys.platform == 'linux':
        QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)

    app = QApplication(sys.argv)
    app.setWindowIcon(app_icon())
    app.setStyleSheet(GLOBAL_STYLESHEET)
    viewer = PointCloudViewer()
    viewer.show()
    sys.exit(app.exec_())
