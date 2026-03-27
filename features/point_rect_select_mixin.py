import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from utils.bbox_pick import points_in_screen_rect


LIST_POINT_SELECT_CAP = 8000  # 列表展示上限，避免一次框选过多点时界面卡死


class PointRectSelectMixin:
    def _toggle_points_rect_select_mode(self):
        """点云框选：每次点击工具栏仅允许拖拽一次矩形；再次框选需重新点击。"""
        self.points_rect_select_mode = not self.points_rect_select_mode
        self.points_rect_select_action.setChecked(self.points_rect_select_mode)
        if self.points_rect_select_mode:
            # 开启下一轮框选：恢复之前高亮的异色，但不取消左侧弹框（只在取消框选时才隐藏）
            self._restore_points_color_only()
            self.box_select_mode = False
            self.box_select_action.setChecked(False)
            # 确保 Dock/label 已创建，避免 AttributeError
            self._ensure_point_select_dock()
            self.point_select_info_label.setText("已就绪：可框选点云")
            self.frame_info_label.setText("点云框选：请拖拽一次矩形")
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
                btn.setStyleSheet(
                    "QToolButton { background-color: transparent; border-radius: 6px; padding: 6px 10px; }"
                )
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
        # 首次创建时先隐藏：应在完成一次拖拽框选后再显示
        dock.hide()
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
                total, shown, "，其余行省略" if total > LIST_POINT_SELECT_CAP else ""
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

