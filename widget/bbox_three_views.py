# -*- coding: utf-8 -*-
"""目标框三视图面板：BEV、侧视图、后视图，可拖动调整尺寸、yaw 与平移"""
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QSizePolicy,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QCursor
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
import matplotlib.pyplot as plt


# 视图半径 = 框在该视图的最大半尺寸 * 此系数，使框在图中占比与参考图类似（约 2/3）
VIEW_BOX_RATIO = 0.75
# 三视图只显示目标框附近点：在 bbox 外额外保留的上下文范围（米）
VIEW_CONTEXT_MARGIN = 0.5
VIEW_CONTEXT_RATIO = 0.25
VIEW_MIN_RADIUS = 0.8
# 拖动检测：与边/手柄的距离小于该值（米）即视为命中
HIT_THRESHOLD = 0.15
# BEV 旋转手柄长度（米）
YAW_HANDLE_LEN = 0.4
MIN_SIZE = 0.2
MAX_SIZE = 50.0
# 三视图拖动边缘改尺寸的灵敏度；越小越细，1.0 为原始跟手速度。
RESIZE_DRAG_SENSITIVITY = 0.8

# 三视图浅色主题
FIG_FACECOLOR = "#e4e8ec"
AXES_FACECOLOR = "#e4e8ec"
EDGE_COLOR = "#2c3e50"
TEXT_COLOR = "#1c1e21"
TICK_COLOR = "#444"
SCATTER_OUT_COLOR = "#666"
SCATTER_IN_COLOR = "#c71585"
YAW_HANDLE_COLOR = "#1e88e5"
# 按钮旋转 yaw 的步长（度）
YAW_BUTTON_STEP_DEG = 5.0


def _dist_point_to_segment(px, py, x0, y0, x1, y1):
    """点到线段的最短距离"""
    dx, dy = x1 - x0, y1 - y0
    d2 = dx * dx + dy * dy
    if d2 < 1e-12:
        return np.hypot(px - x0, py - y0)
    t = np.clip(((px - x0) * dx + (py - y0) * dy) / d2, 0, 1)
    return np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))

def _safe_float(v, default=0.0):
    """避免 None 或非法值导致 unary - 等报错"""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _normalize_yaw(yaw):
    return float((float(yaw) + np.pi) % (2.0 * np.pi) - np.pi)


def points_in_bbox_local(raw_points_xyz, cx, cy, cz, l, w, h, yaw):
    """
    将世界坐标点变换到 bbox 局部坐标，并返回在 bbox 内的点。
    与主视图一致：主视图朝向为 (cos(yaw), sin(yaw))，对应局部 x 轴正方向。
    局部：x 沿车长/朝向(前为正)，y 沿车宽(左为正)，z 向上。
    返回 (local_xyz, mask)，local_xyz 为 (N,3)，mask 为在框内的布尔数组。
    """
    yaw = _safe_float(yaw)
    xy = raw_points_xyz[:, :2] - np.array([cx, cy])
    # 世界到局部旋转 -yaw：使世界 (cos(yaw), sin(yaw)) -> 局部 (1,0)，与 draw_bbox 的朝向一致
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    local_x = cos_yaw * xy[:, 0] - sin_yaw * xy[:, 1]
    local_y = sin_yaw * xy[:, 0] + cos_yaw * xy[:, 1]
    local_z = raw_points_xyz[:, 2] - cz
    local = np.column_stack([local_x, local_y, local_z])
    half = np.array([l / 2, w / 2, h / 2])
    mask = np.all(np.abs(local) <= half + 1e-6, axis=1)
    return local, mask


def nearby_points_for_views(local_pts, l, w, h):
    """只保留目标框周围的局部点，避免三视图显示远处点。"""
    if local_pts is None or len(local_pts) == 0:
        return np.array([], dtype=bool), VIEW_MIN_RADIUS, VIEW_MIN_RADIUS, VIEW_MIN_RADIUS
    x_limit, y_limit, z_limit = nearby_view_limits(l, w, h)
    near_mask = (
        (np.abs(local_pts[:, 0]) <= x_limit) &
        (np.abs(local_pts[:, 1]) <= y_limit) &
        (np.abs(local_pts[:, 2]) <= z_limit)
    )
    r_bev = max(x_limit, y_limit, VIEW_MIN_RADIUS)
    r_side = max(x_limit, z_limit, VIEW_MIN_RADIUS)
    r_rear = max(y_limit, z_limit, VIEW_MIN_RADIUS)
    return near_mask, r_bev, r_side, r_rear


def nearby_view_limits(l, w, h):
    """目标框附近场景的局部坐标范围。"""
    context = max(VIEW_CONTEXT_MARGIN, max(l, w, h) * VIEW_CONTEXT_RATIO)
    half_l, half_w, half_h = l / 2, w / 2, h / 2
    x_limit = half_l + context
    y_limit = half_w + context
    z_limit = half_h + context
    return x_limit, y_limit, z_limit


class _SquareCanvasWrapper(QWidget):
    """包装 FigureCanvas，使绘图区域保持 1:1 比例，与主 3D 视角一致，避免拉伸导致 L/H 显示颠倒。"""
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self._canvas.setParent(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w, h = event.size().width(), event.size().height()
        s = min(w, h)
        if s > 0:
            x = (w - s) // 2
            y = (h - s) // 2
            self._canvas.setGeometry(x, y, s, s)

    def sizeHint(self):
        return QSize(320, 320)


def inner_margins_cm(local_points, l, w, h):
    """根据框内点的局部坐标计算六面内边距（米转厘米）。"""
    if local_points is None or len(local_points) == 0:
        return None
    mx = np.max(local_points[:, 0])
    mnx = np.min(local_points[:, 0])
    my = np.max(local_points[:, 1])
    mny = np.min(local_points[:, 1])
    mz = np.max(local_points[:, 2])
    mnz = np.min(local_points[:, 2])
    half_l, half_w, half_h = l / 2, w / 2, h / 2
    return {
        "head_cm": (half_l - mx) * 100,
        "tail_cm": (mnx + half_l) * 100,
        "left_cm": (mny + half_w) * 100,
        "right_cm": (half_w - my) * 100,
        "top_cm": (half_h - mz) * 100,
        "bottom_cm": (mnz + half_h) * 100,
    }


class BboxThreeViewsPanel(QWidget):
    """右侧辅助视图：BEV、侧视图、后视图，带尺寸/角度/内边距文字"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.setMaximumWidth(500)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # 标题栏 + 关闭
        header = QHBoxLayout()
        self.title_label = QLabel("Auxiliary View")
        self.title_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.title_label.setStyleSheet("color: #333;")
        header.addWidget(self.title_label)
        header.addStretch()
        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(28, 28)
        self.close_btn.setStyleSheet(
            "font-size: 16px; border: none; border-radius: 4px; background: #e8e8e8;"
            "color: #555; font-weight: bold;"
        )
        self.close_btn.clicked.connect(self._on_close)
        header.addWidget(self.close_btn)
        layout.addLayout(header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setSpacing(8)
        self.scroll.setWidget(content)
        layout.addWidget(self.scroll)

        # 三个图占位，首次 update_bbox 时创建
        self.canvas_bev = self.canvas_side = self.canvas_rear = None
        self.fig_bev = self.fig_side = self.fig_rear = None
        self._scatter_bev_out = self._scatter_bev_in = None
        self._scatter_side_out = self._scatter_side_in = None
        self._scatter_rear_out = self._scatter_rear_in = None
        self._bbox_index = None
        self._on_bbox_edited = None
        self._drag_axis = None
        self._drag_mode = None
        self._drag_start = None
        self._raw_points_xyz = None

    def _on_close(self):
        self.hide()

    def update_bbox(self, raw_points_xyz, bbox_info, bbox_index=None, on_bbox_edited=None, class_names=None):
        """
        根据当前帧点云与选中的 bbox_info 更新三视图，并支持拖动调整尺寸与 BEV 旋转 yaw。
        on_bbox_edited(bbox_index, new_bbox_info) 在编辑时被调用以同步主 3D 视图。
        """
        self._raw_points_xyz = raw_points_xyz
        self._bbox_index = bbox_index
        self._on_bbox_edited = on_bbox_edited
        self._bbox_info = {k: v for k, v in bbox_info.items()}

        x = _safe_float(bbox_info.get("x"))
        y = _safe_float(bbox_info.get("y"))
        z = _safe_float(bbox_info.get("z"))
        l = _safe_float(bbox_info.get("l"))
        w = _safe_float(bbox_info.get("w"))
        h = _safe_float(bbox_info.get("h"))
        yaw = _safe_float(bbox_info.get("yaw"))
        pitch = _safe_float(bbox_info.get("pitch"))
        roll = _safe_float(bbox_info.get("roll"))

        local_pts, mask = points_in_bbox_local(raw_points_xyz, x, y, z, l, w, h, yaw)
        # BEV 恢复原来的俯视范围；侧视图/后视图只显示目标框三维邻域内的点。
        r_bev = max(max(l, w) * VIEW_BOX_RATIO, VIEW_MIN_RADIUS)
        near_mask, _, r_side, r_rear = nearby_points_for_views(local_pts, l, w, h)
        in_bev = (np.abs(local_pts[:, 0]) <= r_bev) & (np.abs(local_pts[:, 1]) <= r_bev)
        local_bev = local_pts[in_bev]
        mask_bev = mask[in_bev]
        inside_bev = local_bev[mask_bev]
        outside_bev = local_bev[~mask_bev]
        local_near = local_pts[near_mask]
        mask_near = mask[near_mask]
        inside = local_near[mask_near]
        outside = local_near[~mask_near]
        margins = inner_margins_cm(inside, l, w, h) if len(inside) > 0 else None
        self._r_bev, self._r_side, self._r_rear = r_bev, r_side, r_rear

        yaw_deg = np.rad2deg(yaw)
        pitch_deg = np.rad2deg(pitch) if abs(pitch) < 10 else pitch
        roll_deg = np.rad2deg(roll) if abs(roll) < 10 else roll

        # 清空旧图：移除所有 layout 项（含 stretch），避免顶部留白累积
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
            del item
        self.canvas_bev = self.canvas_side = self.canvas_rear = None
        self.fig_bev = self.fig_side = self.fig_rear = None

        # BEV 俯视图：与主视图对齐。竖轴 = 目标框可视化朝向（local_x，即主视图箭头方向），横轴 = 垂直朝向（local_y）
        # 显示坐标 display = (-local_y, local_x)，即图的上方为车头、框与主视图时刻一致
        fig_bev = Figure(figsize=(4, 4), facecolor=FIG_FACECOLOR)
        ax_bev = fig_bev.add_subplot(111, facecolor=AXES_FACECOLOR)
        bev_x = -outside_bev[:, 1]
        bev_y = outside_bev[:, 0]
        s_out = ax_bev.scatter(bev_x, bev_y, s=2.5, c=SCATTER_OUT_COLOR, alpha=0.5, rasterized=True)
        bev_xi = -inside_bev[:, 1]
        bev_yi = inside_bev[:, 0]
        s_in = ax_bev.scatter(bev_xi, bev_yi, s=4, c=SCATTER_IN_COLOR, alpha=0.9, rasterized=True)
        # 旋转后：矩形在显示坐标中 xy=(-w/2,-l/2) 宽 w 高 l
        rect_bev = Rectangle((-w/2, -l/2), w, l, fill=False, edgecolor=EDGE_COLOR, linewidth=1.5)
        ax_bev.add_patch(rect_bev)
        # 朝向手柄：局部前方向为 +x，显示为 (0, l/2) -> (0, l/2+len)
        yaw_handle = Line2D([0, 0], [l/2, l/2 + YAW_HANDLE_LEN], color=YAW_HANDLE_COLOR, linewidth=2)
        ax_bev.add_line(yaw_handle)
        ax_bev.set_aspect("equal")
        ax_bev.set_xlim(-r_bev, r_bev)
        ax_bev.set_ylim(-r_bev, r_bev)
        ax_bev.set_facecolor(AXES_FACECOLOR)
        ax_bev.axis("off")
        fig_bev.subplots_adjust(bottom=0.14)
        ax_btn_minus = fig_bev.add_axes([0.18, 0.02, 0.14, 0.06])
        ax_btn_plus = fig_bev.add_axes([0.66, 0.02, 0.14, 0.06])
        self._btn_yaw_minus = Button(ax_btn_minus, "Yaw −")
        self._btn_yaw_plus = Button(ax_btn_plus, "Yaw +")
        self._btn_yaw_minus.on_clicked(lambda _: self._yaw_button_step(-1))
        self._btn_yaw_plus.on_clicked(lambda _: self._yaw_button_step(1))
        canvas_bev = FigureCanvas(fig_bev)
        canvas_bev.mpl_connect("button_press_event", self._bev_press)
        canvas_bev.mpl_connect("motion_notify_event", self._bev_motion)
        canvas_bev.mpl_connect("button_release_event", self._bev_release)
        canvas_bev.mpl_connect("scroll_event", self._scroll_zoom)
        self._rect_bev = rect_bev
        self._yaw_handle_bev = yaw_handle
        self._ax_bev = ax_bev
        self._scatter_bev_out = s_out
        self._scatter_bev_in = s_in
        info_bev = "W {:.3f} m  L {:.3f} m  Yaw {:.1f} deg".format(w, l, yaw_deg)
        if margins:
            info_bev += "  |  Margin head {:.2f} cm  tail {:.2f} cm".format(margins["head_cm"], margins["tail_cm"])
        lbl_bev_title = QLabel("BEV View")
        lbl_bev_title.setStyleSheet("font-weight: bold; color: #333; font-size: 10px;")
        lbl_bev_info = QLabel(info_bev)
        lbl_bev_info.setStyleSheet("color: #555; font-size: 9px; font-family: monospace;")
        lbl_bev_info.setWordWrap(True)
        self.content_layout.addWidget(lbl_bev_title)
        self.content_layout.addWidget(lbl_bev_info)
        self.content_layout.addWidget(_SquareCanvasWrapper(canvas_bev), 1)

        # 侧视图: 局部 XZ，四边可拖改 L/H，框内拖平移（正方形 fig 保证 L 水平、H 竖直与 3D 一致）
        fig_side = Figure(figsize=(4, 4), facecolor=FIG_FACECOLOR)
        ax_side = fig_side.add_subplot(111, facecolor=AXES_FACECOLOR)
        s_side_out = ax_side.scatter(outside[:, 0], outside[:, 2], s=2.5, c=SCATTER_OUT_COLOR, alpha=0.5, rasterized=True)
        s_side_in = ax_side.scatter(inside[:, 0], inside[:, 2], s=4, c=SCATTER_IN_COLOR, alpha=0.9, rasterized=True)
        rect_side = Rectangle((-l/2, -h/2), l, h, fill=False, edgecolor=EDGE_COLOR, linewidth=1.5)
        ax_side.add_patch(rect_side)
        ax_side.set_aspect("equal")
        ax_side.set_xlim(-r_side, r_side)
        ax_side.set_ylim(-r_side, r_side)
        ax_side.set_facecolor(AXES_FACECOLOR)
        ax_side.axis("off")
        fig_side.tight_layout()
        canvas_side = FigureCanvas(fig_side)
        canvas_side.mpl_connect("button_press_event", self._side_press)
        canvas_side.mpl_connect("motion_notify_event", self._side_motion)
        canvas_side.mpl_connect("button_release_event", self._side_release)
        canvas_side.mpl_connect("scroll_event", self._scroll_zoom)
        self._rect_side = rect_side
        self._ax_side = ax_side
        self._scatter_side_out = s_side_out
        self._scatter_side_in = s_side_in
        info_side = "L {:.3f} m  H {:.3f} m  Pitch {:.1f} deg".format(l, h, pitch_deg)
        if margins:
            info_side += "  |  Margin top {:.2f} cm  bottom {:.2f} cm".format(margins["top_cm"], margins["bottom_cm"])
        lbl_side_title = QLabel("Side View")
        lbl_side_title.setStyleSheet("font-weight: bold; color: #333; font-size: 10px;")
        lbl_side_info = QLabel(info_side)
        lbl_side_info.setStyleSheet("color: #555; font-size: 9px; font-family: monospace;")
        lbl_side_info.setWordWrap(True)
        self.content_layout.addWidget(lbl_side_title)
        self.content_layout.addWidget(lbl_side_info)
        self.content_layout.addWidget(_SquareCanvasWrapper(canvas_side), 1)

        # 后视图: 显示 (-local_y, local_z)，使图右=车右（局部 y 左为正，取反后图右=车右）
        fig_rear = Figure(figsize=(4, 4), facecolor=FIG_FACECOLOR)
        ax_rear = fig_rear.add_subplot(111, facecolor=AXES_FACECOLOR)
        s_rear_out = ax_rear.scatter(-outside[:, 1], outside[:, 2], s=2.5, c=SCATTER_OUT_COLOR, alpha=0.5, rasterized=True)
        s_rear_in = ax_rear.scatter(-inside[:, 1], inside[:, 2], s=4, c=SCATTER_IN_COLOR, alpha=0.9, rasterized=True)
        rect_rear = Rectangle((-w/2, -h/2), w, h, fill=False, edgecolor=EDGE_COLOR, linewidth=1.5)
        ax_rear.add_patch(rect_rear)
        ax_rear.set_aspect("equal")
        ax_rear.set_xlim(-r_rear, r_rear)
        ax_rear.set_ylim(-r_rear, r_rear)
        ax_rear.set_facecolor(AXES_FACECOLOR)
        ax_rear.axis("off")
        fig_rear.tight_layout()
        canvas_rear = FigureCanvas(fig_rear)
        canvas_rear.mpl_connect("button_press_event", self._rear_press)
        canvas_rear.mpl_connect("motion_notify_event", self._rear_motion)
        canvas_rear.mpl_connect("button_release_event", self._rear_release)
        canvas_rear.mpl_connect("scroll_event", self._scroll_zoom)
        self._rect_rear = rect_rear
        self._ax_rear = ax_rear
        self._scatter_rear_out = s_rear_out
        self._scatter_rear_in = s_rear_in
        info_rear = "W {:.3f} m  H {:.3f} m  Roll {:.1f} deg".format(w, h, roll_deg)
        if margins:
            info_rear += "  |  Margin left {:.2f} cm  right {:.2f} cm".format(margins["left_cm"], margins["right_cm"])
        lbl_rear_title = QLabel("Rear View")
        lbl_rear_title.setStyleSheet("font-weight: bold; color: #333; font-size: 10px;")
        lbl_rear_info = QLabel(info_rear)
        lbl_rear_info.setStyleSheet("color: #555; font-size: 9px; font-family: monospace;")
        lbl_rear_info.setWordWrap(True)
        self.content_layout.addWidget(lbl_rear_title)
        self.content_layout.addWidget(lbl_rear_info)
        self.content_layout.addWidget(_SquareCanvasWrapper(canvas_rear), 1)

        self.fig_bev, self.fig_side, self.fig_rear = fig_bev, fig_side, fig_rear
        self.canvas_bev, self.canvas_side, self.canvas_rear = canvas_bev, canvas_side, canvas_rear

    def _emit_bbox_edited(self):
        if self._on_bbox_edited and self._bbox_index is not None:
            geometry_keys = ("x", "y", "z", "l", "w", "h", "yaw")
            self._on_bbox_edited(
                self._bbox_index,
                {key: self._bbox_info[key] for key in geometry_keys if key in self._bbox_info},
            )

    def _yaw_button_step(self, direction):
        """BEV 中 Yaw −/＋：直接修改七值框的 yaw。"""
        step_rad = np.deg2rad(YAW_BUTTON_STEP_DEG) * direction
        self._bbox_info["yaw"] = _normalize_yaw(_safe_float(self._bbox_info.get("yaw")) + step_rad)
        self._sync_rects()
        self._emit_bbox_edited()

    def _sync_rects(self, reset_limits=True):
        l = _safe_float(self._bbox_info.get("l"))
        w = _safe_float(self._bbox_info.get("w"))
        h = _safe_float(self._bbox_info.get("h"))
        if hasattr(self, "_rect_bev") and self._rect_bev:
            self._rect_bev.set_xy((-w/2, -l/2))
            self._rect_bev.set_width(w)
            self._rect_bev.set_height(l)
        if hasattr(self, "_yaw_handle_bev") and self._yaw_handle_bev:
            self._yaw_handle_bev.set_ydata([l/2, l/2 + YAW_HANDLE_LEN])
        if hasattr(self, "_rect_side") and self._rect_side:
            self._rect_side.set_xy((-l/2, -h/2))
            self._rect_side.set_width(l)
            self._rect_side.set_height(h)
        if hasattr(self, "_rect_rear") and self._rect_rear:
            self._rect_rear.set_xy((-w/2, -h/2))
            self._rect_rear.set_width(w)
            self._rect_rear.set_height(h)
        self._update_scatter_points(reset_limits=reset_limits)
        for c in (self.canvas_bev, self.canvas_side, self.canvas_rear):
            if c:
                c.draw_idle()

    def _update_scatter_points(self, reset_limits=True):
        """根据当前 _bbox_info 的 center/yaw 重算局部点并更新三视图 scatter"""
        if not hasattr(self, "_raw_points_xyz") or self._raw_points_xyz is None or len(self._raw_points_xyz) == 0:
            return
        x = _safe_float(self._bbox_info.get("x"))
        y = _safe_float(self._bbox_info.get("y"))
        z = _safe_float(self._bbox_info.get("z"))
        l = _safe_float(self._bbox_info.get("l"))
        w = _safe_float(self._bbox_info.get("w"))
        h = _safe_float(self._bbox_info.get("h"))
        yaw = _safe_float(self._bbox_info.get("yaw"))
        local_pts, mask = points_in_bbox_local(self._raw_points_xyz, x, y, z, l, w, h, yaw)
        r_bev = max(max(l, w) * VIEW_BOX_RATIO, VIEW_MIN_RADIUS)
        _, _, r_side, r_rear = nearby_points_for_views(local_pts, l, w, h)
        self._r_bev, self._r_side, self._r_rear = r_bev, r_side, r_rear
        if reset_limits and hasattr(self, "_ax_bev"):
            self._ax_bev.set_xlim(-r_bev, r_bev)
            self._ax_bev.set_ylim(-r_bev, r_bev)
        if reset_limits and hasattr(self, "_ax_side"):
            self._ax_side.set_xlim(-r_side, r_side)
            self._ax_side.set_ylim(-r_side, r_side)
        if reset_limits and hasattr(self, "_ax_rear"):
            self._ax_rear.set_xlim(-r_rear, r_rear)
            self._ax_rear.set_ylim(-r_rear, r_rear)

        bev_xlim = self._ax_bev.get_xlim() if hasattr(self, "_ax_bev") else (-r_bev, r_bev)
        bev_ylim = self._ax_bev.get_ylim() if hasattr(self, "_ax_bev") else (-r_bev, r_bev)
        side_xlim = self._ax_side.get_xlim() if hasattr(self, "_ax_side") else (-r_side, r_side)
        side_ylim = self._ax_side.get_ylim() if hasattr(self, "_ax_side") else (-r_side, r_side)
        rear_xlim = self._ax_rear.get_xlim() if hasattr(self, "_ax_rear") else (-r_rear, r_rear)
        rear_ylim = self._ax_rear.get_ylim() if hasattr(self, "_ax_rear") else (-r_rear, r_rear)

        bev_disp_x = -local_pts[:, 1]
        bev_disp_y = local_pts[:, 0]
        side_disp_x = local_pts[:, 0]
        side_disp_y = local_pts[:, 2]
        rear_disp_x = -local_pts[:, 1]
        rear_disp_y = local_pts[:, 2]
        x_limit, y_limit, _ = nearby_view_limits(l, w, h)
        half_l, half_w = l / 2, w / 2
        side_view_w = abs(side_xlim[1] - side_xlim[0])
        side_view_h = abs(side_ylim[1] - side_ylim[0])
        rear_view_w = abs(rear_xlim[1] - rear_xlim[0])
        rear_view_h = abs(rear_ylim[1] - rear_ylim[0])
        side_scale = max(
            side_view_w / max(2 * r_side, 1e-9),
            side_view_h / max(2 * r_side, 1e-9),
        )
        rear_scale = max(
            rear_view_w / max(2 * r_rear, 1e-9),
            rear_view_h / max(2 * r_rear, 1e-9),
        )
        side_y_limit = half_w + max(0.0, y_limit - half_w) * side_scale
        rear_x_limit = half_l + max(0.0, x_limit - half_l) * rear_scale

        in_bev = (
            (bev_disp_x >= min(bev_xlim)) & (bev_disp_x <= max(bev_xlim)) &
            (bev_disp_y >= min(bev_ylim)) & (bev_disp_y <= max(bev_ylim))
        )
        in_side = (
            (side_disp_x >= min(side_xlim)) & (side_disp_x <= max(side_xlim)) &
            (side_disp_y >= min(side_ylim)) & (side_disp_y <= max(side_ylim)) &
            (np.abs(local_pts[:, 1]) <= side_y_limit)
        )
        in_rear = (
            (rear_disp_x >= min(rear_xlim)) & (rear_disp_x <= max(rear_xlim)) &
            (rear_disp_y >= min(rear_ylim)) & (rear_disp_y <= max(rear_ylim)) &
            (np.abs(local_pts[:, 0]) <= rear_x_limit)
        )
        local_bev = local_pts[in_bev]
        mask_bev = mask[in_bev]
        inside_bev = local_bev[mask_bev]
        outside_bev = local_bev[~mask_bev]
        local_side = local_pts[in_side]
        mask_side = mask[in_side]
        inside_side = local_side[mask_side]
        outside_side = local_side[~mask_side]
        local_rear = local_pts[in_rear]
        mask_rear = mask[in_rear]
        inside_rear = local_rear[mask_rear]
        outside_rear = local_rear[~mask_rear]
        # BEV: 竖轴=local_x(朝向)，横轴=-local_y，与主视图框对齐
        if self._scatter_bev_out is not None:
            off = np.column_stack([-outside_bev[:, 1], outside_bev[:, 0]]) if len(outside_bev) > 0 else np.empty((0, 2))
            self._scatter_bev_out.set_offsets(off)
        if self._scatter_bev_in is not None:
            off = np.column_stack([-inside_bev[:, 1], inside_bev[:, 0]]) if len(inside_bev) > 0 else np.empty((0, 2))
            self._scatter_bev_in.set_offsets(off)
        if self._scatter_side_out is not None:
            self._scatter_side_out.set_offsets(np.column_stack([outside_side[:, 0], outside_side[:, 2]]) if len(outside_side) > 0 else np.empty((0, 2)))
        if self._scatter_side_in is not None:
            self._scatter_side_in.set_offsets(np.column_stack([inside_side[:, 0], inside_side[:, 2]]) if len(inside_side) > 0 else np.empty((0, 2)))
        if self._scatter_rear_out is not None:
            off = np.column_stack([-outside_rear[:, 1], outside_rear[:, 2]]) if len(outside_rear) > 0 else np.empty((0, 2))
            self._scatter_rear_out.set_offsets(off)
        if self._scatter_rear_in is not None:
            off = np.column_stack([-inside_rear[:, 1], inside_rear[:, 2]]) if len(inside_rear) > 0 else np.empty((0, 2))
            self._scatter_rear_in.set_offsets(off)

    def _scroll_zoom(self, event):
        """鼠标滚轮缩放当前三视图坐标轴。"""
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        base = 0.85 if event.button == "up" else 1.18
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        new_w = (x1 - x0) * base
        new_h = (y1 - y0) * base
        relx = (cx - x0) / (x1 - x0) if abs(x1 - x0) > 1e-9 else 0.5
        rely = (cy - y0) / (y1 - y0) if abs(y1 - y0) > 1e-9 else 0.5
        ax.set_xlim(cx - new_w * relx, cx + new_w * (1 - relx))
        ax.set_ylim(cy - new_h * rely, cy + new_h * (1 - rely))
        self._update_scatter_points(reset_limits=False)
        if event.canvas:
            event.canvas.draw_idle()

    def _bev_hit(self, xd, yd):
        """命中检测：俯视图已逆时针旋转 90°，按到各边/手柄的最近距离判定，避免角点误判"""
        l = _safe_float(self._bbox_info.get("l"))
        w = _safe_float(self._bbox_info.get("w"))
        if xd is None or yd is None:
            return None
        if -w/2 < xd < w/2 and -l/2 < yd < l/2:
            return "move"
        # 到各边/手柄的距离（仅考虑落在边延长线上的最近点）
        d_left = abs(xd - (-w/2)) if -l/2 - HIT_THRESHOLD <= yd <= l/2 + HIT_THRESHOLD else np.inf
        d_right = abs(xd - w/2) if -l/2 - HIT_THRESHOLD <= yd <= l/2 + HIT_THRESHOLD else np.inf
        d_bottom = abs(yd - (-l/2)) if -w/2 - HIT_THRESHOLD <= xd <= w/2 + HIT_THRESHOLD else np.inf
        d_top = abs(yd - l/2) if -w/2 - HIT_THRESHOLD <= xd <= w/2 + HIT_THRESHOLD else np.inf
        d_handle = _dist_point_to_segment(xd, yd, 0, l/2, 0, l/2 + YAW_HANDLE_LEN)
        best = [
            (d_left, "resize_w_left"),
            (d_right, "resize_w"),
            (d_bottom, "resize_l_bottom"),
            (d_top, "resize_l"),
            (d_handle, "rotate"),
        ]
        best.sort(key=lambda t: t[0])
        if best[0][0] <= HIT_THRESHOLD:
            return best[0][1]
        return None

    def _bev_press(self, event):
        if event.inaxes != self._ax_bev or event.button != 1:
            return
        self._drag_mode = self._bev_hit(event.xdata, event.ydata)
        if self._drag_mode:
            self._drag_axis = "bev"
            self._drag_start = (
                event.xdata, event.ydata,
                _safe_float(self._bbox_info.get("x")), _safe_float(self._bbox_info.get("y")),
                _safe_float(self._bbox_info.get("l")), _safe_float(self._bbox_info.get("w")),
                _safe_float(self._bbox_info.get("yaw")),
            )

    def _bev_motion(self, event):
        if event.inaxes != self._ax_bev:
            if self.canvas_bev:
                self.canvas_bev.setCursor(QCursor(Qt.ArrowCursor))
            return
        # 未拖动时悬停：根据命中设置光标
        if self._drag_axis != "bev":
            hit = self._bev_hit(event.xdata, event.ydata)
            if hit in ("resize_w", "resize_w_left"):
                self.canvas_bev.setCursor(QCursor(Qt.SizeHorCursor))
            elif hit in ("resize_l", "resize_l_bottom"):
                self.canvas_bev.setCursor(QCursor(Qt.SizeVerCursor))
            elif hit in ("rotate", "move"):
                self.canvas_bev.setCursor(QCursor(Qt.SizeAllCursor))
            else:
                self.canvas_bev.setCursor(QCursor(Qt.ArrowCursor))
            return
        if self._drag_mode is None:
            return
        l = _safe_float(self._bbox_info.get("l"))
        w = _safe_float(self._bbox_info.get("w"))
        yaw = _safe_float(self._bbox_info.get("yaw"))
        cx, cy = _safe_float(self._bbox_info.get("x")), _safe_float(self._bbox_info.get("y"))
        sx, sy, scx, scy = (self._drag_start[:4] if self._drag_start is not None and len(self._drag_start) >= 4 else (event.xdata, event.ydata, cx, cy))
        sl = self._drag_start[4] if self._drag_start is not None and len(self._drag_start) >= 6 else l
        sw = self._drag_start[5] if self._drag_start is not None and len(self._drag_start) >= 6 else w
        syaw = self._drag_start[6] if self._drag_start is not None and len(self._drag_start) >= 7 else yaw
        # 俯视图已旋转 90°：显示中 x 对应车宽，y 对应车长；左/右拖改 w，下/上拖改 l
        if self._drag_mode == "resize_w":
            # 拖显示右边缘 (x=w/2)
            dx = ((event.xdata if event.xdata is not None else sx) - sx) * RESIZE_DRAG_SENSITIVITY
            new_w = np.clip(sw + dx, MIN_SIZE, MAX_SIZE)
            disp_cx = (new_w - sw) / 2
            self._bbox_info["w"] = new_w
            self._bbox_info["x"] = scx + np.sin(yaw) * disp_cx
            self._bbox_info["y"] = scy - np.cos(yaw) * disp_cx
        elif self._drag_mode == "resize_w_left":
            dx = ((event.xdata if event.xdata is not None else sx) - sx) * RESIZE_DRAG_SENSITIVITY
            new_w = np.clip(sw - dx, MIN_SIZE, MAX_SIZE)
            disp_cx = (sw - new_w) / 2
            self._bbox_info["w"] = new_w
            self._bbox_info["x"] = scx + np.sin(yaw) * disp_cx
            self._bbox_info["y"] = scy - np.cos(yaw) * disp_cx
        elif self._drag_mode == "resize_l":
            # 拖显示上边缘 (y=l/2)，即车长
            dy = ((event.ydata if event.ydata is not None else sy) - sy) * RESIZE_DRAG_SENSITIVITY
            new_l = np.clip(sl + dy, MIN_SIZE, MAX_SIZE)
            disp_cy = (new_l - sl) / 2
            self._bbox_info["l"] = new_l
            self._bbox_info["x"] = scx + np.cos(yaw) * disp_cy
            self._bbox_info["y"] = scy + np.sin(yaw) * disp_cy
        elif self._drag_mode == "resize_l_bottom":
            dy = ((event.ydata if event.ydata is not None else sy) - sy) * RESIZE_DRAG_SENSITIVITY
            new_l = np.clip(sl - dy, MIN_SIZE, MAX_SIZE)
            disp_cy = (sl - new_l) / 2
            self._bbox_info["l"] = new_l
            self._bbox_info["x"] = scx + np.cos(yaw) * disp_cy
            self._bbox_info["y"] = scy + np.sin(yaw) * disp_cy
        elif self._drag_mode == "rotate":
            if event.xdata is not None and event.ydata is not None:
                delta_yaw = np.arctan2(-event.xdata, event.ydata)
                self._bbox_info["yaw"] = _normalize_yaw(syaw + delta_yaw)
        elif self._drag_mode == "move" and self._drag_start is not None and len(self._drag_start) >= 4:
            sx, sy, scx, scy = self._drag_start[0], self._drag_start[1], self._drag_start[2], self._drag_start[3]
            dx = (event.xdata or sx) - sx
            dy = (event.ydata or sy) - sy
            # 显示 (dx,dy) -> 局部 (dy, -dx) -> 世界
            self._bbox_info["x"] = scx + np.cos(yaw) * dy + np.sin(yaw) * dx
            self._bbox_info["y"] = scy + np.sin(yaw) * dy - np.cos(yaw) * dx
        self._sync_rects(reset_limits=False)
        self._emit_bbox_edited()

    def _bev_release(self, event):
        if self._drag_axis == "bev":
            self._drag_axis = None
            self._drag_mode = None
            self._drag_start = None

    def _side_hit(self, xd, yd):
        """按到各边最近距离判定，避免角点误判"""
        l = _safe_float(self._bbox_info.get("l"))
        h = _safe_float(self._bbox_info.get("h"))
        if xd is None or yd is None:
            return None
        if -l/2 < xd < l/2 and -h/2 < yd < h/2:
            return "move"
        d_left = abs(xd - (-l/2)) if -h/2 - HIT_THRESHOLD <= yd <= h/2 + HIT_THRESHOLD else np.inf
        d_right = abs(xd - l/2) if -h/2 - HIT_THRESHOLD <= yd <= h/2 + HIT_THRESHOLD else np.inf
        d_bottom = abs(yd - (-h/2)) if -l/2 - HIT_THRESHOLD <= xd <= l/2 + HIT_THRESHOLD else np.inf
        d_top = abs(yd - h/2) if -l/2 - HIT_THRESHOLD <= xd <= l/2 + HIT_THRESHOLD else np.inf
        best = [(d_left, "resize_l_left"), (d_right, "resize_l"), (d_bottom, "resize_h_bottom"), (d_top, "resize_h")]
        best.sort(key=lambda t: t[0])
        if best[0][0] <= HIT_THRESHOLD:
            return best[0][1]
        return None

    def _side_press(self, event):
        if event.inaxes != self._ax_side or event.button != 1:
            return
        self._drag_mode = self._side_hit(event.xdata, event.ydata)
        if self._drag_mode:
            self._drag_axis = "side"
            self._drag_start = (
                event.xdata, event.ydata,
                _safe_float(self._bbox_info.get("x")), _safe_float(self._bbox_info.get("y")), _safe_float(self._bbox_info.get("z")),
                _safe_float(self._bbox_info.get("l")), _safe_float(self._bbox_info.get("h")),
            )

    def _side_motion(self, event):
        if event.inaxes != self._ax_side:
            if self.canvas_side:
                self.canvas_side.setCursor(QCursor(Qt.ArrowCursor))
            return
        if self._drag_axis != "side":
            hit = self._side_hit(event.xdata, event.ydata)
            if hit in ("resize_l", "resize_l_left"):
                self.canvas_side.setCursor(QCursor(Qt.SizeHorCursor))
            elif hit in ("resize_h", "resize_h_bottom"):
                self.canvas_side.setCursor(QCursor(Qt.SizeVerCursor))
            elif hit == "move":
                self.canvas_side.setCursor(QCursor(Qt.SizeAllCursor))
            else:
                self.canvas_side.setCursor(QCursor(Qt.ArrowCursor))
            return
        if self._drag_mode is None:
            return
        l = _safe_float(self._bbox_info.get("l"))
        h = _safe_float(self._bbox_info.get("h"))
        yaw = _safe_float(self._bbox_info.get("yaw"))
        cx, cy, cz = _safe_float(self._bbox_info.get("x")), _safe_float(self._bbox_info.get("y")), _safe_float(self._bbox_info.get("z"))
        sx, sy, scx, scy, scz = (self._drag_start[:5] if self._drag_start is not None and len(self._drag_start) >= 5 else (event.xdata, event.ydata, cx, cy, cz))
        sl = self._drag_start[5] if self._drag_start is not None and len(self._drag_start) >= 7 else l
        sh = self._drag_start[6] if self._drag_start is not None and len(self._drag_start) >= 7 else h
        if self._drag_mode == "resize_l":
            dx = ((event.xdata if event.xdata is not None else sx) - sx) * RESIZE_DRAG_SENSITIVITY
            new_l = np.clip(sl + dx, MIN_SIZE, MAX_SIZE)
            disp = (new_l - sl) / 2
            self._bbox_info["l"] = new_l
            self._bbox_info["x"] = scx + np.cos(yaw) * disp
            self._bbox_info["y"] = scy + np.sin(yaw) * disp
        elif self._drag_mode == "resize_l_left":
            dx = ((event.xdata if event.xdata is not None else sx) - sx) * RESIZE_DRAG_SENSITIVITY
            new_l = np.clip(sl - dx, MIN_SIZE, MAX_SIZE)
            disp = (sl - new_l) / 2
            self._bbox_info["l"] = new_l
            self._bbox_info["x"] = scx + np.cos(yaw) * disp
            self._bbox_info["y"] = scy + np.sin(yaw) * disp
        elif self._drag_mode == "resize_h":
            dy = ((event.ydata if event.ydata is not None else sy) - sy) * RESIZE_DRAG_SENSITIVITY
            new_h = np.clip(sh + dy, MIN_SIZE, MAX_SIZE)
            disp = (new_h - sh) / 2
            self._bbox_info["h"] = new_h
            self._bbox_info["z"] = scz + disp
        elif self._drag_mode == "resize_h_bottom":
            dy = ((event.ydata if event.ydata is not None else sy) - sy) * RESIZE_DRAG_SENSITIVITY
            new_h = np.clip(sh - dy, MIN_SIZE, MAX_SIZE)
            disp = (sh - new_h) / 2
            self._bbox_info["h"] = new_h
            self._bbox_info["z"] = scz + disp
        elif self._drag_mode == "move" and self._drag_start is not None and len(self._drag_start) >= 5:
            sx, sy, scx, scy, scz = self._drag_start[0], self._drag_start[1], self._drag_start[2], self._drag_start[3], self._drag_start[4]
            dx = (event.xdata or sx) - sx
            dz = (event.ydata or sy) - sy
            self._bbox_info["x"] = scx + np.cos(yaw) * dx
            self._bbox_info["y"] = scy + np.sin(yaw) * dx
            self._bbox_info["z"] = scz + dz
        self._sync_rects(reset_limits=False)
        self._emit_bbox_edited()

    def _side_release(self, event):
        if self._drag_axis == "side":
            self._drag_axis = None
            self._drag_mode = None
            self._drag_start = None

    def _rear_hit(self, xd, yd):
        """按到各边最近距离判定，避免角点误判"""
        w = _safe_float(self._bbox_info.get("w"))
        h = _safe_float(self._bbox_info.get("h"))
        if xd is None or yd is None:
            return None
        if -w/2 < xd < w/2 and -h/2 < yd < h/2:
            return "move"
        d_left = abs(xd - (-w/2)) if -h/2 - HIT_THRESHOLD <= yd <= h/2 + HIT_THRESHOLD else np.inf
        d_right = abs(xd - w/2) if -h/2 - HIT_THRESHOLD <= yd <= h/2 + HIT_THRESHOLD else np.inf
        d_bottom = abs(yd - (-h/2)) if -w/2 - HIT_THRESHOLD <= xd <= w/2 + HIT_THRESHOLD else np.inf
        d_top = abs(yd - h/2) if -w/2 - HIT_THRESHOLD <= xd <= w/2 + HIT_THRESHOLD else np.inf
        best = [(d_left, "resize_w_left"), (d_right, "resize_w"), (d_bottom, "resize_h_bottom"), (d_top, "resize_h")]
        best.sort(key=lambda t: t[0])
        if best[0][0] <= HIT_THRESHOLD:
            return best[0][1]
        return None

    def _rear_press(self, event):
        if event.inaxes != self._ax_rear or event.button != 1:
            return
        self._drag_mode = self._rear_hit(event.xdata, event.ydata)
        if self._drag_mode:
            self._drag_axis = "rear"
            self._drag_start = (
                event.xdata, event.ydata,
                _safe_float(self._bbox_info.get("x")), _safe_float(self._bbox_info.get("y")), _safe_float(self._bbox_info.get("z")),
                _safe_float(self._bbox_info.get("w")), _safe_float(self._bbox_info.get("h")),
            )

    def _rear_motion(self, event):
        if event.inaxes != self._ax_rear:
            if self.canvas_rear:
                self.canvas_rear.setCursor(QCursor(Qt.ArrowCursor))
            return
        if self._drag_axis != "rear":
            hit = self._rear_hit(event.xdata, event.ydata)
            if hit in ("resize_w", "resize_w_left"):
                self.canvas_rear.setCursor(QCursor(Qt.SizeHorCursor))
            elif hit in ("resize_h", "resize_h_bottom"):
                self.canvas_rear.setCursor(QCursor(Qt.SizeVerCursor))
            elif hit == "move":
                self.canvas_rear.setCursor(QCursor(Qt.SizeAllCursor))
            else:
                self.canvas_rear.setCursor(QCursor(Qt.ArrowCursor))
            return
        if self._drag_mode is None:
            return
        w = _safe_float(self._bbox_info.get("w"))
        h = _safe_float(self._bbox_info.get("h"))
        yaw = _safe_float(self._bbox_info.get("yaw"))
        cx, cy, cz = _safe_float(self._bbox_info.get("x")), _safe_float(self._bbox_info.get("y")), _safe_float(self._bbox_info.get("z"))
        sx, sy, scx, scy, scz = (self._drag_start[:5] if self._drag_start is not None and len(self._drag_start) >= 5 else (event.xdata, event.ydata, cx, cy, cz))
        sw = self._drag_start[5] if self._drag_start is not None and len(self._drag_start) >= 7 else w
        sh = self._drag_start[6] if self._drag_start is not None and len(self._drag_start) >= 7 else h
        if self._drag_mode == "resize_w":
            dx = ((event.xdata if event.xdata is not None else sx) - sx) * RESIZE_DRAG_SENSITIVITY
            new_w = np.clip(sw + dx, MIN_SIZE, MAX_SIZE)
            disp = (new_w - sw) / 2
            self._bbox_info["w"] = new_w
            self._bbox_info["x"] = scx + np.sin(yaw) * disp
            self._bbox_info["y"] = scy - np.cos(yaw) * disp
        elif self._drag_mode == "resize_w_left":
            dx = ((event.xdata if event.xdata is not None else sx) - sx) * RESIZE_DRAG_SENSITIVITY
            new_w = np.clip(sw - dx, MIN_SIZE, MAX_SIZE)
            disp = (sw - new_w) / 2
            self._bbox_info["w"] = new_w
            self._bbox_info["x"] = scx + np.sin(yaw) * disp
            self._bbox_info["y"] = scy - np.cos(yaw) * disp
        elif self._drag_mode == "resize_h":
            dy = ((event.ydata if event.ydata is not None else sy) - sy) * RESIZE_DRAG_SENSITIVITY
            new_h = np.clip(sh + dy, MIN_SIZE, MAX_SIZE)
            disp = (new_h - sh) / 2
            self._bbox_info["h"] = new_h
            self._bbox_info["z"] = scz + disp
        elif self._drag_mode == "resize_h_bottom":
            dy = ((event.ydata if event.ydata is not None else sy) - sy) * RESIZE_DRAG_SENSITIVITY
            new_h = np.clip(sh - dy, MIN_SIZE, MAX_SIZE)
            disp = (sh - new_h) / 2
            self._bbox_info["h"] = new_h
            self._bbox_info["z"] = scz + disp
        elif self._drag_mode == "move" and self._drag_start is not None and len(self._drag_start) >= 5:
            sx, sy, scx, scy, scz = self._drag_start[0], self._drag_start[1], self._drag_start[2], self._drag_start[3], self._drag_start[4]
            dy = (event.xdata or sx) - sx
            dz = (event.ydata or sy) - sy
            self._bbox_info["x"] = scx + np.sin(yaw) * dy
            self._bbox_info["y"] = scy - np.cos(yaw) * dy
            self._bbox_info["z"] = scz + dz
        self._sync_rects(reset_limits=False)
        self._emit_bbox_edited()

    def _rear_release(self, event):
        if self._drag_axis == "rear":
            self._drag_axis = None
            self._drag_mode = None
            self._drag_start = None
