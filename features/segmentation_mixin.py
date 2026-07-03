import json
import os
from html import escape
from pathlib import Path

import numpy as np
import pyqtgraph.opengl as gl
from matplotlib.path import Path as MplPath

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox

from dialogs.segmentation_label_dialog import (
    SegmentationLabelPickDialog,
    SegmentationLabelSettingsDialog,
)
from utils.bbox_pick import ray_from_screen, ray_plane_z_intersect, world_to_screen


class SegmentationMixin:
    def _init_segmentation_state(self):
        self._segmentation_mode = False
        self._segmentation_labels = self._load_segmentation_label_config()
        self._segmentation_keys = None
        self._segmentation_polygon = []
        self._segmentation_items = []
        self._segmentation_hover = None
        self._segmentation_annotation_path = None
        self._segmentation_has_annotation = False
        self._segmentation_display_enabled = False
        self._segmentation_action = getattr(self, "_segmentation_action", None)
        self._segmentation_legend = None

    def _default_segmentation_labels(self):
        return [{"name": "background", "key": 0, "color": "#FFFFFF"}]

    def _segmentation_config_path(self):
        return Path.home() / ".pcdview_segmentation_labels.json"

    def _load_segmentation_label_config(self):
        path = self._segmentation_config_path()
        defaults = self._default_segmentation_labels()
        if not path.is_file():
            return defaults
        try:
            with open(path, "r", encoding="UTF-8") as f:
                data = json.load(f)
        except Exception:
            return defaults
        labels = data.get("labels", data) if isinstance(data, dict) else data
        if not isinstance(labels, list):
            return defaults
        cleaned = []
        for item in labels:
            if not isinstance(item, dict):
                continue
            try:
                key = int(item.get("key"))
            except (TypeError, ValueError):
                continue
            name = str(item.get("name") or "label_{}".format(key)).strip()
            color = str(item.get("color") or "#FFFFFF").strip()
            if not color.startswith("#") or len(color) != 7:
                color = "#FFFFFF" if key == 0 else "#E74C3C"
            cleaned.append({"name": name, "key": key, "color": color.upper()})
        by_key = {item["key"]: item for item in cleaned}
        by_key.setdefault(0, defaults[0])
        by_key[0]["color"] = "#FFFFFF"
        return [by_key[key] for key in sorted(by_key)]

    def _save_segmentation_label_config(self):
        path = self._segmentation_config_path()
        with open(path, "w", encoding="UTF-8") as f:
            json.dump({"labels": self._segmentation_labels}, f, indent=2, ensure_ascii=False)

    def _open_segmentation_label_settings(self):
        dlg = SegmentationLabelSettingsDialog(self._segmentation_labels, None)
        dlg.setWindowModality(Qt.ApplicationModal)
        if dlg.exec_() != QDialog.Accepted:
            return
        self._segmentation_labels = dlg.selected_labels()
        try:
            self._save_segmentation_label_config()
        except Exception as exc:
            QMessageBox.warning(None, "分割标签设置", "保存标签配置失败: {}".format(exc))
        self._update_segmentation_legend()
        self.vis_fram(updata_color_bar=False, preserve_current_bboxes=True)

    def _refresh_segmentation_display(self):
        if hasattr(self, "vis_fram"):
            self.vis_fram(updata_color_bar=False, preserve_current_bboxes=True)

    def _toggle_segmentation_mode(self):
        if bool(getattr(self, "_segmentation_mode", False)):
            self._exit_segmentation_mode()
            return
        self._segmentation_mode = True
        if hasattr(self, "_segmentation_action") and self._segmentation_action is not None:
            self._segmentation_action.setChecked(True)
        self._prepare_segmentation_mode_entry()
        self._load_segmentation_for_current_frame()
        self._segmentation_display_enabled = True
        self._set_segmentation_legend_visible(True)
        self._refresh_segmentation_display()
        self._set_status_message("点云分割标注：右键添加顶点，右键点击首点闭合；左键和滚轮仍可操作视角")

    def _exit_segmentation_mode(self):
        self._segmentation_mode = False
        if hasattr(self, "_segmentation_action") and self._segmentation_action is not None:
            self._segmentation_action.setChecked(False)
        self._clear_segmentation_polygon()
        self._segmentation_display_enabled = False
        self._set_segmentation_legend_visible(False)
        self.glwidget.unsetCursor()
        self._refresh_segmentation_display()
        self._set_status_message("已退出点云分割标注模式")
        return True

    def _prepare_segmentation_mode_entry(self):
        self.box_select_mode = False
        self.box_select_start = None
        self.box_select_start_logical = None
        if hasattr(self, "box_select_action"):
            self.box_select_action.setChecked(False)
        self.points_rect_select_mode = False
        if hasattr(self, "points_rect_select_action"):
            self.points_rect_select_action.setChecked(False)
        if hasattr(self, "_update_points_rect_button_style"):
            self._update_points_rect_button_style(False)
        overlay = getattr(self, "box_select_overlay", None)
        if overlay is not None and hasattr(overlay, "clear_rect"):
            overlay.clear_rect()

    def _segmentation_txt_path_for_pcd(self, pcd_file=None):
        pcd = pcd_file or getattr(self, "pcd_file", None)
        if not pcd:
            return None
        return os.path.splitext(os.path.abspath(pcd))[0] + ".txt"

    def _sync_segmentation_for_loaded_frame(self):
        if bool(getattr(self, "_segmentation_mode", False)):
            return self._load_segmentation_for_current_frame()
        self._segmentation_annotation_path = self._segmentation_txt_path_for_pcd()
        self._segmentation_keys = None
        self._segmentation_has_annotation = False
        self._segmentation_display_enabled = False
        self._update_segmentation_legend()
        return False

    def _ensure_segmentation_keys(self):
        raw_points = getattr(self, "raw_points", None)
        n = int(len(raw_points)) if raw_points is not None else 0
        if n <= 0:
            self._segmentation_keys = np.zeros(0, dtype=np.int32)
            return
        if self._segmentation_keys is None or len(self._segmentation_keys) != n:
            self._segmentation_keys = np.zeros(n, dtype=np.int32)

    def _load_segmentation_for_current_frame(self, path=None, show_errors=False, as_frame_annotation=True):
        self._ensure_segmentation_keys()
        n = len(self._segmentation_keys)
        frame_txt_path = self._segmentation_txt_path_for_pcd()
        txt_path = path or frame_txt_path
        self._segmentation_annotation_path = frame_txt_path
        if not txt_path or not os.path.isfile(txt_path):
            self._segmentation_keys = np.zeros(n, dtype=np.int32)
            self._segmentation_has_annotation = False
            self._segmentation_display_enabled = False
            self._update_segmentation_legend()
            return False
        try:
            values = np.loadtxt(txt_path, dtype=np.int64)
            values = np.atleast_1d(values).astype(np.int32)
        except Exception as exc:
            if show_errors:
                QMessageBox.warning(None, "加载分割标注", "读取失败: {}".format(exc))
            self._segmentation_keys = np.zeros(n, dtype=np.int32)
            self._segmentation_has_annotation = False
            self._segmentation_display_enabled = False
            self._update_segmentation_legend()
            return False
        if len(values) != n:
            if show_errors:
                QMessageBox.warning(
                    None,
                    "加载分割标注",
                    "标注点数 {} 与当前点云点数 {} 不一致。".format(len(values), n),
                )
            self._segmentation_keys = np.zeros(n, dtype=np.int32)
            self._segmentation_has_annotation = False
            self._segmentation_display_enabled = False
            self._update_segmentation_legend()
            return False
        self._segmentation_keys = values
        self._segmentation_has_annotation = True
        if as_frame_annotation:
            self._segmentation_annotation_path = txt_path
            self._segmentation_display_enabled = bool(getattr(self, "_segmentation_mode", False))
        else:
            self._segmentation_annotation_path = frame_txt_path
            self._segmentation_display_enabled = True
        self._update_segmentation_legend()
        return True

    def _load_segmentation_txt_dialog(self):
        if not hasattr(self, "raw_points") or self.raw_points is None or len(self.raw_points) == 0:
            QMessageBox.information(None, "加载分割标注", "当前没有点云。")
            return
        start_dir = os.path.dirname(os.path.abspath(getattr(self, "pcd_file", "") or os.getcwd()))
        path, _ = QFileDialog.getOpenFileName(None, "加载分割标注 TXT", start_dir, "Text Files (*.txt);;All Files (*)")
        if not path:
            return
        if self._load_segmentation_for_current_frame(path=path, show_errors=True, as_frame_annotation=False):
            self.vis_fram(updata_color_bar=False, preserve_current_bboxes=True)
            self._set_status_message(
                "已导入分割标注: {}，后续自动保存到当前帧同名txt".format(os.path.basename(path))
            )

    def _save_segmentation_for_current_frame(self):
        self._ensure_segmentation_keys()
        path = self._segmentation_annotation_path or self._segmentation_txt_path_for_pcd()
        if not path:
            return False
        try:
            np.savetxt(path, np.asarray(self._segmentation_keys, dtype=np.int32), fmt="%d")
        except Exception as exc:
            QMessageBox.warning(None, "保存分割标注", "保存失败: {}".format(exc))
            return False
        self._segmentation_annotation_path = path
        self._segmentation_has_annotation = True
        self._segmentation_display_enabled = True
        return True

    def _hex_to_rgba(self, color, alpha=1.0):
        text = str(color or "#FFFFFF").strip()
        if not text.startswith("#") or len(text) != 7:
            text = "#FFFFFF"
        return (
            int(text[1:3], 16) / 255.0,
            int(text[3:5], 16) / 255.0,
            int(text[5:7], 16) / 255.0,
            float(alpha),
        )

    def _segmentation_color_for_unknown_key(self, key):
        palette = [
            "#00A6D6",
            "#7C3AED",
            "#F97316",
            "#10B981",
            "#E11D48",
            "#A3E635",
            "#F59E0B",
            "#6366F1",
        ]
        return palette[abs(int(key)) % len(palette)]

    def _apply_segmentation_colors(self, rgba, keep_inside_mask=None):
        keys = getattr(self, "_segmentation_keys", None)
        if not bool(getattr(self, "_segmentation_mode", False)) and not bool(getattr(self, "_segmentation_display_enabled", False)):
            return rgba
        if keys is None or len(keys) == 0:
            return rgba
        keys_use = None
        if len(keys) == len(rgba):
            keys_use = keys
        elif keep_inside_mask is not None and len(keys) == len(keep_inside_mask):
            keys_use = np.asarray(keys)[np.asarray(keep_inside_mask, dtype=bool)]
        if keys_use is None or len(keys_use) != len(rgba):
            return rgba
        label_by_key = {int(item["key"]): item for item in self._segmentation_labels}
        out = np.asarray(rgba, dtype=np.float32).copy()
        for key in np.unique(keys_use):
            key = int(key)
            label = label_by_key.get(key)
            color = label.get("color") if label is not None else self._segmentation_color_for_unknown_key(key)
            out[np.asarray(keys_use) == key] = self._hex_to_rgba(color)
        return out

    def _clear_segmentation_polygon(self):
        for item in getattr(self, "_segmentation_items", []):
            try:
                self.glwidget.removeItem(item)
            except Exception:
                pass
        self._segmentation_items = []
        self._segmentation_polygon = []
        self._segmentation_hover = None
        self.glwidget.update()

    def _refresh_segmentation_polygon_items(self):
        for item in getattr(self, "_segmentation_items", []):
            try:
                self.glwidget.removeItem(item)
            except Exception:
                pass
        self._segmentation_items = []
        pts2 = np.asarray(getattr(self, "_segmentation_polygon", []), dtype=np.float32)
        if len(pts2) == 0:
            self.glwidget.update()
            return
        z = self._segmentation_overlay_z()
        pts3 = np.column_stack([pts2[:, 0], pts2[:, 1], np.full(len(pts2), z)]).astype(np.float32)
        p_item = gl.GLScatterPlotItem(pos=pts3, color=(1.0, 0.82, 0.0, 1.0), size=8.0)
        self._segmentation_items.append(p_item)
        self.glwidget.addItem(p_item)
        draw = pts3
        if self._segmentation_hover is not None:
            hover = np.asarray([[self._segmentation_hover[0], self._segmentation_hover[1], z]], dtype=np.float32)
            draw = np.vstack([pts3, hover])
        if len(draw) >= 2:
            l_item = gl.GLLinePlotItem(pos=draw, color=(1.0, 0.82, 0.0, 1.0), width=2.0, antialias=True, mode="line_strip")
            self._segmentation_items.append(l_item)
            self.glwidget.addItem(l_item)
        if self._segmentation_hover_is_first_vertex():
            self._add_segmentation_snap_ring(self._segmentation_polygon[0])
        self.glwidget.update()

    def _segmentation_hover_is_first_vertex(self):
        polygon = getattr(self, "_segmentation_polygon", [])
        hover = getattr(self, "_segmentation_hover", None)
        if hover is None or len(polygon) < 3:
            return False
        try:
            return bool(np.allclose(np.asarray(hover, dtype=np.float64), np.asarray(polygon[0], dtype=np.float64)))
        except Exception:
            return False

    def _add_segmentation_snap_ring(self, xy, color=(1.0, 0.92, 0.0, 1.0), radius_px=13.0, segments=48):
        if xy is None:
            return
        z = self._segmentation_overlay_z()
        center = np.asarray([float(xy[0]), float(xy[1]), z], dtype=np.float64)
        try:
            world_radius = float(self.glwidget.pixelSize(center)) * float(radius_px)
        except Exception:
            world_radius = 0.2
        angles = np.linspace(0.0, 2.0 * np.pi, int(segments) + 1)
        pts = np.zeros((len(angles), 3), dtype=np.float32)
        pts[:, 0] = center[0] + np.cos(angles) * world_radius
        pts[:, 1] = center[1] + np.sin(angles) * world_radius
        pts[:, 2] = center[2]
        item = gl.GLLinePlotItem(pos=pts, color=color, width=3.0, antialias=True, mode="line_strip")
        self._segmentation_items.append(item)
        self.glwidget.addItem(item)

    def _segmentation_overlay_z(self):
        pts = getattr(self, "raw_points", None)
        if pts is None or len(pts) == 0:
            return 0.0
        try:
            return float(np.nanmedian(pts[:, 2]))
        except Exception:
            return 0.0

    def _segmentation_first_vertex_hit(self, mx, my, threshold=12.0):
        if len(getattr(self, "_segmentation_polygon", [])) < 3:
            return False
        first = np.asarray([[self._segmentation_polygon[0][0], self._segmentation_polygon[0][1], self._segmentation_overlay_z()]])
        screen = world_to_screen(self.glwidget, first)
        if screen is None:
            return False
        sx, sy = screen[0]
        if not np.isfinite(sx) or not np.isfinite(sy):
            return False
        return (float(mx) - sx) ** 2 + (float(my) - sy) ** 2 <= float(threshold) ** 2

    def _segmentation_add_vertex_from_screen(self, mx, my):
        if self._segmentation_first_vertex_hit(mx, my):
            self._finish_segmentation_polygon()
            return True
        ray = ray_from_screen(self.glwidget, mx, my)
        pt = ray_plane_z_intersect(ray[0], ray[1], self._segmentation_overlay_z()) if ray is not None else None
        if pt is None:
            self._set_status_message("无法把当前点击位置投影到分割绘制平面")
            return True
        self._segmentation_polygon.append([float(pt[0]), float(pt[1])])
        self._segmentation_hover = None
        self._refresh_segmentation_polygon_items()
        self._set_status_message("分割多边形：已添加 {} 个顶点".format(len(self._segmentation_polygon)))
        return True

    def _finish_segmentation_polygon(self):
        polygon_world = np.asarray(getattr(self, "_segmentation_polygon", []), dtype=np.float64)
        if len(polygon_world) < 3:
            self._set_status_message("至少需要 3 个顶点才能闭合分割多边形")
            return False
        if not hasattr(self, "raw_points") or self.raw_points is None or len(self.raw_points) == 0:
            self._set_status_message("当前没有可标注点云")
            return False
        z = self._segmentation_overlay_z()
        poly3 = np.column_stack(
            [polygon_world[:, 0], polygon_world[:, 1], np.full(len(polygon_world), z)]
        )
        polygon_screen = world_to_screen(self.glwidget, poly3)
        screen = world_to_screen(self.glwidget, np.asarray(self.raw_points[:, :3], dtype=np.float64))
        if screen is None:
            self._set_status_message("无法投影当前点云")
            return False
        valid = np.isfinite(screen[:, 0]) & np.isfinite(screen[:, 1])
        mask = np.zeros(len(screen), dtype=bool)
        if polygon_screen is None or not np.all(np.isfinite(polygon_screen)):
            self._set_status_message("无法投影当前分割多边形")
            return False
        mask[valid] = MplPath(np.asarray(polygon_screen, dtype=np.float64)).contains_points(screen[valid])
        count = int(np.count_nonzero(mask))
        if count <= 0:
            self._clear_segmentation_polygon()
            self._set_status_message("多边形内没有点")
            return True
        dlg = SegmentationLabelPickDialog(self._segmentation_labels, count, None)
        dlg.setWindowModality(Qt.ApplicationModal)
        if dlg.exec_() != QDialog.Accepted or dlg.selected_key is None:
            return True
        self._ensure_segmentation_keys()
        self._segmentation_keys[mask] = int(dlg.selected_key)
        self._save_segmentation_for_current_frame()
        self._clear_segmentation_polygon()
        self._update_segmentation_legend()
        self.vis_fram(updata_color_bar=False, preserve_current_bboxes=True)
        self._set_status_message("已将 {} 个点标注为 key={}".format(count, int(dlg.selected_key)))
        return True

    def _handle_segmentation_key_event(self, event):
        if not bool(getattr(self, "_segmentation_mode", False)):
            return False
        if event.type() != QEvent.KeyPress:
            return False
        if event.key() == Qt.Key_Escape:
            if self._segmentation_polygon:
                self._clear_segmentation_polygon()
                self._set_status_message("已取消当前分割多边形")
            else:
                self._exit_segmentation_mode()
            return True
        if event.key() == Qt.Key_Backspace:
            if self._segmentation_polygon:
                self._segmentation_polygon.pop()
                self._refresh_segmentation_polygon_items()
                self._set_status_message("分割多边形：已删除上一个顶点")
            return True
        return False

    def _handle_segmentation_mouse_event(self, event, mx, my):
        if not bool(getattr(self, "_segmentation_mode", False)):
            return False
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            return self._segmentation_add_vertex_from_screen(mx, my)
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.RightButton:
            return True
        if event.type() == QEvent.MouseMove:
            if self._segmentation_polygon:
                if self._segmentation_first_vertex_hit(mx, my):
                    self.glwidget.setCursor(Qt.PointingHandCursor)
                    self._segmentation_hover = self._segmentation_polygon[0]
                else:
                    self.glwidget.setCursor(Qt.CrossCursor)
                    ray = ray_from_screen(self.glwidget, mx, my)
                    pt = ray_plane_z_intersect(ray[0], ray[1], self._segmentation_overlay_z()) if ray is not None else None
                    self._segmentation_hover = [float(pt[0]), float(pt[1])] if pt is not None else None
                self._refresh_segmentation_polygon_items()
            return False
        return False

    def _set_segmentation_legend_visible(self, visible):
        if not hasattr(self, "segmentation_legend_panel"):
            return
        if visible:
            self._update_segmentation_legend()
            self._update_segmentation_legend_geometry()
            self.segmentation_legend_panel.show()
            self.segmentation_legend_panel.raise_()
        else:
            self.segmentation_legend_panel.hide()

    def _update_segmentation_legend_geometry(self):
        if not hasattr(self, "segmentation_legend_panel") or self.glwidget.width() <= 0 or self.glwidget.height() <= 0:
            return
        margin = 14
        hint = self.segmentation_legend_panel.sizeHint()
        w = min(260, max(150, hint.width()))
        h = min(320, max(72, hint.height()))
        x = margin
        y = max(margin, self.glwidget.height() - h - margin)
        self.segmentation_legend_panel.setGeometry(x, y, w, h)
        self.segmentation_legend_panel.raise_()

    def _update_segmentation_legend(self):
        if not hasattr(self, "segmentation_legend_label"):
            return
        keys = getattr(self, "_segmentation_keys", None)
        total = int(len(keys)) if keys is not None else 0
        counts = {}
        if keys is not None and len(keys) > 0:
            uniq, cnt = np.unique(keys, return_counts=True)
            counts = {int(k): int(c) for k, c in zip(uniq, cnt)}
        def row_html(color, name, key, count):
            return (
                '<tr>'
                '<td style="padding:3px 6px 3px 0;"><span style="color:{};">■</span></td>'
                '<td style="padding:3px 10px 3px 0;">{}</td>'
                '<td style="padding:3px 10px 3px 0; text-align:right;">{}</td>'
                '<td style="padding:3px 0; text-align:right;">{}</td>'
                '</tr>'
            ).format(
                color,
                escape(str(name)),
                int(key),
                int(count),
            )

        table_rows = []
        for label in self._segmentation_labels:
            key = int(label["key"])
            table_rows.append(row_html(label.get("color", "#FFFFFF"), label.get("name", ""), key, counts.get(key, 0)))
        unknown = sorted(k for k in counts if k not in {int(item["key"]) for item in self._segmentation_labels})
        for key in unknown:
            table_rows.append(row_html(self._segmentation_color_for_unknown_key(key), "未知标签", key, counts.get(key, 0)))

        html = (
            '<div style="font-weight:700; margin-bottom:4px;">点云分割标注</div>'
            '<div style="margin-bottom:6px;">总点数: {}</div>'
            '<table cellspacing="0" cellpadding="0" style="width:100%; border-collapse:collapse;">'
            '<tr>'
            '<th style="padding:2px 6px 4px 0; text-align:left;"></th>'
            '<th style="padding:2px 10px 4px 0; text-align:left;">Name</th>'
            '<th style="padding:2px 10px 4px 0; text-align:right;">Key</th>'
            '<th style="padding:2px 0 4px 0; text-align:right;">Count</th>'
            '</tr>'
            '{}'
            '</table>'
        ).format(total, "".join(table_rows))
        self.segmentation_legend_label.setText(html)
        self._update_segmentation_legend_geometry()
