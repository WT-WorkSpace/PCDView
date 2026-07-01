import json
import os
import re

import numpy as np
import pyqtgraph.opengl as gl
from matplotlib.path import Path as MplPath

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QAction, QFileDialog, QInputDialog, QMenu, QMessageBox

from dialogs.mask_param_dialog import MaskParamDialog
from utils.bbox_pick import ray_from_screen, ray_plane_z_intersect, world_to_screen


class MaskMixin:
    def _init_mask_draw_state(self):
        draw_action = getattr(self, "_mask_draw_action", None)
        draw_menu_action = getattr(self, "_mask_draw_menu_action", None)
        edit_action = getattr(self, "_mask_edit_action", None)
        edit_menu_action = getattr(self, "_mask_edit_menu_action", None)
        self._mask_draw_mode = False
        self._mask_edit_mode = False
        self._mask_draw_flatten_display = False
        self._mask_draw_vertices = []
        self._mask_draw_items = []
        self._mask_draw_action = draw_action
        self._mask_draw_menu_action = draw_menu_action
        self._mask_edit_action = edit_action
        self._mask_edit_menu_action = edit_menu_action
        self._mask_draw_previous_view = None
        self._mask_pan_last_pos = None
        self._mask_draw_click_start = None
        self._mask_draw_hover_xy = None
        self._mask_hover_shape_index = None
        self._mask_hover_vertex_index = None
        self._mask_hover_edge_index = None
        self._mask_drag_vertex = None
        self._mask_hover_items = []
        self._mask_selected_shape_index = None
        self._mask_selected_items = []

    def _mask_color_for_index(self, idx):
        palette = [
            (230, 25, 75),
            (60, 180, 75),
            (0, 130, 200),
            (245, 130, 48),
            (145, 30, 180),
            (70, 240, 240),
            (240, 50, 230),
            (210, 245, 60),
            (250, 190, 212),
            (0, 128, 128),
            (220, 190, 255),
            (170, 110, 40),
            (255, 250, 200),
            (128, 0, 0),
            (170, 255, 195),
            (0, 0, 128),
        ]
        return palette[int(idx) % len(palette)]

    def _qrgb_to_rgba_f(self, rgb):
        r, g, b = [int(v) for v in rgb]
        return (r / 255.0, g / 255.0, b / 255.0, 1.0)

    def _clear_mask_items(self):
        if not self._mask_items:
            return
        for it in self._mask_items:
            try:
                self.glwidget.removeItem(it)
            except Exception:
                pass
        self._mask_items = []

    def _clear_mask_hover_items(self):
        for it in getattr(self, "_mask_hover_items", []):
            try:
                self.glwidget.removeItem(it)
            except Exception:
                pass
        self._mask_hover_items = []

    def _clear_mask_selected_items(self):
        for it in getattr(self, "_mask_selected_items", []):
            try:
                self.glwidget.removeItem(it)
            except Exception:
                pass
        self._mask_selected_items = []
        self._mask_selected_shape_index = None

    def _add_mask_ring_marker(self, xy, color=(1.0, 1.0, 0.0, 1.0), radius_px=10.0, segments=48):
        if xy is None:
            return
        z = float(self._mask_params.get("point_z", 0.0))
        center = np.array([float(xy[0]), float(xy[1]), z], dtype=np.float64)
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
        self._mask_hover_items.append(item)
        self.glwidget.addItem(item)

    def _add_mask_cross_marker(self, xy, color=(1.0, 1.0, 0.0, 1.0), radius_px=9.0):
        if xy is None:
            return
        z = float(self._mask_params.get("point_z", 0.0))
        center = np.array([float(xy[0]), float(xy[1]), z], dtype=np.float64)
        try:
            world_radius = float(self.glwidget.pixelSize(center)) * float(radius_px)
        except Exception:
            world_radius = 0.2
        x, y = float(center[0]), float(center[1])
        pts = np.array([
            [x - world_radius, y, z],
            [x + world_radius, y, z],
            [x, y - world_radius, z],
            [x, y + world_radius, z],
        ], dtype=np.float32)
        for seg in (pts[:2], pts[2:]):
            item = gl.GLLinePlotItem(pos=seg, color=color, width=3.0, antialias=True, mode="lines")
            self._mask_hover_items.append(item)
            self.glwidget.addItem(item)

    def _select_mask_shape(self, shape_index, data=None):
        self._clear_mask_selected_items()
        if shape_index is None:
            return
        if data is None:
            data, _ = self._load_mask_data_for_edit()
        shapes = data.get("shapes", [])
        shape_index = int(shape_index)
        if shape_index < 0 or shape_index >= len(shapes):
            return
        pts = shapes[shape_index].get("points", [])
        if not isinstance(pts, list) or len(pts) < 3:
            return
        arr2 = np.asarray(pts, dtype=np.float32)
        z = float(self._mask_params.get("point_z", 0.0)) + 0.01
        arr3 = np.zeros((len(arr2), 3), dtype=np.float32)
        arr3[:, 0] = arr2[:, 0]
        arr3[:, 1] = arr2[:, 1]
        arr3[:, 2] = z
        line = gl.GLLinePlotItem(
            pos=np.vstack([arr3, arr3[:1]]),
            color=(1.0, 1.0, 1.0, 0.45),
            width=max(2.0, float(self._mask_params.get("line_width", 2.0))),
            antialias=True,
            mode="line_strip",
        )
        self._mask_selected_items.append(line)
        self.glwidget.addItem(line)
        vertex_item = gl.GLScatterPlotItem(
            pos=arr3,
            color=(1.0, 1.0, 1.0, 0.75),
            size=max(4.0, float(self._mask_params.get("point_size", 4.0)) + 2.0),
        )
        self._mask_selected_items.append(vertex_item)
        self.glwidget.addItem(vertex_item)
        self._mask_selected_shape_index = shape_index

    def _load_mask_shapes(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("shapes", [])

    def _build_mask_items(self, shapes):
        items = []
        point_size = float(self._mask_params.get("point_size", 4.0))
        line_width = float(self._mask_params.get("line_width", 2.0))
        point_z = float(self._mask_params.get("point_z", 0.0))

        for shape_index, shp in enumerate(shapes):
            pts = shp.get("points", [])
            if not pts:
                continue
            arr2 = np.asarray(pts, dtype=np.float32)
            if arr2.ndim != 2 or arr2.shape[1] < 2:
                continue
            arr3 = np.zeros((arr2.shape[0], 3), dtype=np.float32)
            arr3[:, 0] = arr2[:, 0]
            arr3[:, 1] = arr2[:, 1]
            arr3[:, 2] = point_z

            shp_type = str(shp.get("shape_type", "")).lower()
            rgb = self._mask_color_for_index(shape_index) if shp_type == "polygon" else self._mask_params.get("line_color", (255, 255, 0))
            line_color = self._qrgb_to_rgba_f(rgb)
            point_color = self._qrgb_to_rgba_f(rgb)

            # 所有类型都绘制点（便于观察顶点）
            p_item = gl.GLScatterPlotItem(pos=arr3, color=point_color, size=point_size)
            items.append(p_item)

            if shp_type in ("line", "polygon") and len(arr3) >= 2:
                if shp_type == "polygon":
                    arr3_line = np.vstack([arr3, arr3[0:1]])
                else:
                    arr3_line = arr3
                l_item = gl.GLLinePlotItem(
                    pos=arr3_line,
                    color=line_color,
                    width=line_width,
                    antialias=True,
                    mode="line_strip",
                )
                items.append(l_item)
            if shp_type == "polygon" and len(arr3) >= 3:
                label = str(shp.get("label", "")).strip()
                if label:
                    center = np.mean(arr3, axis=0)
                    t_item = gl.GLTextItem(
                        text=label,
                        pos=(float(center[0]), float(center[1]), float(point_z)),
                        color=line_color,
                        font=QFont("Helvetica", 10),
                    )
                    items.append(t_item)
        return items

    def _collect_mask_polygons_xy(self, shapes):
        polys = []
        for shp in shapes:
            shp_type = str(shp.get("shape_type", "")).lower()
            if shp_type != "polygon":
                continue
            pts = shp.get("points", [])
            if not pts or len(pts) < 3:
                continue
            arr2 = np.asarray(pts, dtype=np.float64)
            if arr2.ndim == 2 and arr2.shape[1] >= 2:
                polys.append(arr2[:, :2])
        return polys

    def _mask_keep_inside_points(self, points_xyz):
        """
        仅保留 mask polygon 内点（忽略 z 轴）。
        返回 bool mask（长度与 points_xyz 相同）；无有效 polygon 时全保留。
        """
        if points_xyz is None or len(points_xyz) == 0:
            return np.array([], dtype=bool)
        if not bool(self._mask_params.get("keep_inside_points", False)):
            return np.ones(len(points_xyz), dtype=bool)

        json_path = self._mask_params.get("json_path", "")
        if not json_path or not os.path.isfile(json_path):
            return np.ones(len(points_xyz), dtype=bool)

        try:
            shapes = self._load_mask_shapes(json_path)
            polys = self._collect_mask_polygons_xy(shapes)
            if not polys:
                return np.ones(len(points_xyz), dtype=bool)

            xy = np.asarray(points_xyz[:, :2], dtype=np.float64)
            keep = np.zeros(len(xy), dtype=bool)
            for poly in polys:
                path = MplPath(poly)
                keep |= path.contains_points(xy)
            return keep
        except Exception:
            return np.ones(len(points_xyz), dtype=bool)

    def _rebuild_mask(self):
        self._clear_mask_items()
        json_path = self._mask_params.get("json_path", "")
        if not json_path or not os.path.isfile(json_path):
            return False, "未设置有效的Mask JSON文件"
        try:
            shapes = self._load_mask_shapes(json_path)
            self._mask_items = self._build_mask_items(shapes)
            for it in self._mask_items:
                self.glwidget.addItem(it)
            self.glwidget.update()
            return True, "Mask已加载: %d 个图元" % len(self._mask_items)
        except Exception as e:
            self._mask_items = []
            return False, "Mask加载失败: %s" % e

    def _toggle_mask_visibility(self, checked=False):
        if checked:
            ok, msg = self._rebuild_mask()
            if not ok:
                # 失败时回退按钮状态
                if self._mask_toggle_action is not None:
                    self._mask_toggle_action.setChecked(False)
                self._mask_visible = False
                self._set_status_message(msg)
                QMessageBox.warning(self, "Mask", msg)
                return
            self._mask_visible = True
            self._set_status_message(msg)
        else:
            self._clear_mask_items()
            self._mask_visible = False
            self._set_status_message("Mask已关闭")

    def _open_mask_settings(self):
        def _on_change_live(new_params, _key):
            """
            实时预览（轻量）：
            - 仅对 Mask 样式/geometry 进行重建：不影响点云主绘制（降低“拖动时联动全局卡顿”）。
            - 只有当「仅保留圈内点」(keep_inside_points) 被切换，或 JSON 路径变化且保留圈内点开启时，才刷新点云（vis_fram）。
            - 当选择 JSON 文件时自动开启 Mask 并显示。
            """
            old_visible = bool(self._mask_visible)
            self._mask_params = new_params

            json_path = self._mask_params.get("json_path", "")
            keep_inside = bool(self._mask_params.get("keep_inside_points", False))

            # 选择 JSON 后自动打开 Mask 并显示
            if _key == "json_path" and json_path and not old_visible:
                if self._mask_toggle_action is not None:
                    self._mask_toggle_action.setChecked(True)
                self._toggle_mask_visibility(True)
                old_visible = bool(self._mask_visible)

            # 若Mask正在显示：只重建mask图元（点/线大小颜色等）
            if old_visible:
                self._rebuild_mask()

            # 分开联动：只有在需要过滤点云时才刷新主界面点云
            if _key in ("keep_inside_points", "json_path") and keep_inside:
                self.vis_fram(updata_color_bar=False, preserve_current_bboxes=True)
            elif _key == "keep_inside_points" and not keep_inside:
                # 取消仅保留圈内点：恢复点云
                self.vis_fram(updata_color_bar=False, preserve_current_bboxes=True)

            # 若 keep_inside_points 未开启，则 style/point_size/颜色改动只重建 mask，不刷新点云
            # 以实现“对主界面联动拆分”的效果。

        dlg = MaskParamDialog(None, self._mask_params, on_change=_on_change_live, owner=self)
        dlg.setWindowModality(Qt.ApplicationModal)
        if dlg.exec_() != 1:  # QDialog.Accepted
            return
        self._mask_params = dlg.get_params()
        # 若当前Mask处于显示状态，参数修改后立即重建生效
        if self._mask_visible:
            ok, msg = self._rebuild_mask()
            self._set_status_message(msg)
        # 保证“仅保留圈内点”在确认后立即生效
        self.vis_fram(updata_color_bar=False, preserve_current_bboxes=True)

    def _convert_tanway_mask_id(self, identifier):
        zone_id_int = int(identifier.split("-")[0])
        lane_id_int = int(identifier.split("-")[1])
        return zone_id_int | (lane_id_int << 3)

    def _mask_label_to_export_id(self, label, fallback, require_tanway_format=False):
        text = str(label or "").strip()
        if re.fullmatch(r"\d+-\d+", text):
            return self._convert_tanway_mask_id(text)
        if require_tanway_format:
            raise ValueError("tanway txt导出要求地图编号必须是xx-xx形式，当前编号为: {}".format(text or "空"))
        if not text:
            return int(fallback)
        try:
            return int(text)
        except ValueError:
            pass
        return int(fallback)

    def _build_mask_export_map(self, params, require_tanway_format=False):
        json_path = str(params.get("json_path", "") or "").strip()
        if not json_path or not os.path.isfile(json_path):
            raise ValueError("请先选择有效的Mask JSON文件。")

        x_min = float(params.get("export_x_min", -40.0))
        x_max = float(params.get("export_x_max", 110.0))
        y_min = float(params.get("export_y_min", -70.0))
        y_max = float(params.get("export_y_max", 80.0))
        pixel = int(params.get("export_pixel", 10))
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("导出范围无效：最大值必须大于最小值。")
        if pixel <= 0:
            raise ValueError("导出像素必须大于0。")

        mask_width = int(np.ceil((x_max - x_min) * pixel))
        mask_height = int(np.ceil((y_max - y_min) * pixel))
        if mask_width <= 0 or mask_height <= 0:
            raise ValueError("导出尺寸无效。")
        if mask_width * mask_height > 100_000_000:
            raise ValueError("导出尺寸过大：{} x {}，请缩小范围或降低像素。".format(mask_width, mask_height))

        shapes = self._load_mask_shapes(json_path)
        mask_map = np.zeros((mask_width, mask_height), dtype=np.int32)
        x_indices_all = np.arange(mask_width, dtype=np.float64)

        for shape_index, shape in enumerate(shapes):
            if str(shape.get("shape_type", "")).lower() != "polygon":
                continue
            polygon = shape.get("points", [])
            if not polygon or len(polygon) < 3:
                continue
            poly = np.asarray(polygon, dtype=np.float64)
            if poly.ndim != 2 or poly.shape[1] < 2:
                continue
            poly = poly[:, :2]
            poly_x_min, poly_y_min = np.min(poly, axis=0)
            poly_x_max, poly_y_max = np.max(poly, axis=0)
            ix0 = max(0, int(np.floor((poly_x_min - x_min) * pixel)))
            ix1 = min(mask_width, int(np.ceil((poly_x_max - x_min) * pixel)) + 1)
            jy0 = max(0, int(np.floor((y_max - poly_y_max) * pixel)))
            jy1 = min(mask_height, int(np.ceil((y_max - poly_y_min) * pixel)) + 1)
            if ix0 >= ix1 or jy0 >= jy1:
                continue

            ys = y_max - np.arange(jy0, jy1, dtype=np.float64) / float(pixel)
            export_id = self._mask_label_to_export_id(
                shape.get("label", ""),
                shape_index + 1,
                require_tanway_format=require_tanway_format,
            )
            path = MplPath(poly)
            block_rows = max(1, int(1_000_000 / max(1, jy1 - jy0)))
            for block_ix0 in range(ix0, ix1, block_rows):
                block_ix1 = min(ix1, block_ix0 + block_rows)
                xs = x_min + x_indices_all[block_ix0:block_ix1] / float(pixel)
                grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
                points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
                inside = path.contains_points(points).reshape(grid_x.shape)
                sub_map = mask_map[block_ix0:block_ix1, jy0:jy1]
                sub_map[inside] = export_id

        return json_path, mask_map, (mask_width, mask_height)

    def _export_mask_tanway_txt(self, params):
        try:
            json_path, mask_map, (mask_width, mask_height) = self._build_mask_export_map(
                params,
                require_tanway_format=True,
            )
        except Exception as exc:
            QMessageBox.warning(self, "导出tanway_txt失败", str(exc))
            return

        start_dir = os.path.dirname(os.path.abspath(json_path))
        default_name = os.path.splitext(os.path.basename(json_path))[0] + "_tanway.txt"
        save_path, _ = QFileDialog.getSaveFileName(
            None,
            "导出tanway_txt",
            os.path.join(start_dir, default_name),
            "Text Files (*.txt)",
        )
        if not save_path:
            return
        if not save_path.lower().endswith(".txt"):
            save_path += ".txt"

        try:
            tanway_map = np.rot90(mask_map.T, 3)
            np.savetxt(save_path, tanway_map, fmt="%d", delimiter=" ")
            self._mask_params.update(params)
            self._set_status_message("tanway_txt已导出: {}".format(save_path))
            QMessageBox.information(
                self,
                "导出成功",
                "{}\n尺寸: {} x {}".format(save_path, tanway_map.shape[0], tanway_map.shape[1]),
            )
        except Exception as exc:
            QMessageBox.warning(self, "导出tanway_txt失败", str(exc))

    def _export_mask_npy(self, params):
        try:
            json_path, mask_map, (mask_width, mask_height) = self._build_mask_export_map(params)
        except Exception as exc:
            QMessageBox.warning(self, "导出npy失败", str(exc))
            return

        start_dir = os.path.dirname(os.path.abspath(json_path))
        default_name = os.path.splitext(os.path.basename(json_path))[0] + "_mask.npy"
        save_path, _ = QFileDialog.getSaveFileName(
            None,
            "导出npy",
            os.path.join(start_dir, default_name),
            "NumPy Files (*.npy)",
        )
        if not save_path:
            return
        if not save_path.lower().endswith(".npy"):
            save_path += ".npy"

        try:
            np.save(save_path, mask_map)
            self._mask_params.update(params)
            self._set_status_message("npy已导出: {}".format(save_path))
            QMessageBox.information(
                self,
                "导出成功",
                "{}\n尺寸: {} x {}".format(save_path, mask_width, mask_height),
            )
        except Exception as exc:
            QMessageBox.warning(self, "导出npy失败", str(exc))

    def _toggle_mask_draw_mode(self, checked=False):
        checked = bool(checked)
        if checked == bool(getattr(self, "_mask_draw_mode", False)):
            return
        if checked:
            if not bool(getattr(self, "_mask_edit_mode", False)):
                self._enter_mask_edit_mode()
            self._enter_mask_draw_mode()
        else:
            self._leave_mask_draw_mode(restore_view=False, return_to_edit=True)

    def _toggle_mask_edit_mode(self, checked=False):
        checked = bool(checked)
        if checked == bool(getattr(self, "_mask_edit_mode", False)):
            return
        if checked:
            self._enter_mask_edit_mode()
        else:
            self._leave_mask_edit_mode(restore_view=True)

    def _enter_mask_draw_mode(self):
        if not hasattr(self, "_mask_draw_vertices"):
            self._init_mask_draw_state()
        self._mask_draw_mode = True
        self._mask_edit_mode = True
        self._sync_mask_draw_action_checked(True)
        self._sync_mask_edit_action_checked(True)
        self._mask_draw_flatten_display = True
        self._mask_params["point_z"] = 0.0
        self._mask_draw_vertices = []
        self._mask_draw_hover_xy = None
        self._mask_drag_vertex = None
        self._mask_pan_last_pos = None
        self._mask_draw_click_start = None
        if self._mask_draw_previous_view is None:
            try:
                self._mask_draw_previous_view = self.glwidget.cameraParams()
            except Exception:
                self._mask_draw_previous_view = None

        if hasattr(self, "box_select_mode"):
            self.box_select_mode = False
        if hasattr(self, "box_select_action"):
            self.box_select_action.setChecked(False)
        if hasattr(self, "points_rect_select_mode"):
            self.points_rect_select_mode = False
        if hasattr(self, "points_rect_select_action"):
            self.points_rect_select_action.setChecked(False)
        if hasattr(self, "_extrinsic_calib_mode") and self._extrinsic_calib_mode:
            self._disable_extrinsic_calib()

        self._set_mask_draw_action_active(True)
        self._set_mask_edit_action_active(True)
        self._set_mask_edit_button_visible(True)
        self.glwidget.setCursor(Qt.CrossCursor)
        if self._mask_params.get("json_path") and os.path.isfile(self._mask_params.get("json_path")):
            self._mask_visible = True
            if self._mask_toggle_action is not None:
                self._mask_toggle_action.setChecked(True)
            self._rebuild_mask()
        self.vis_fram(updata_color_bar=False, preserve_current_bboxes=True)
        self._set_status_message("Mask绘制：右键添加顶点，右键点击首点闭合；完成后需再次点击左上角绘制Mask")

    def _leave_mask_draw_mode(self, restore_view=False, return_to_edit=False):
        self._mask_draw_mode = False
        self._sync_mask_draw_action_checked(False)
        if return_to_edit:
            self._mask_edit_mode = True
            self._sync_mask_edit_action_checked(True)
        if not bool(getattr(self, "_mask_edit_mode", False)):
            self._mask_draw_flatten_display = False
        self._mask_pan_last_pos = None
        self._mask_drag_vertex = None
        self._clear_mask_draw_temp_items()
        self._set_mask_draw_action_active(False)
        if not bool(getattr(self, "_mask_edit_mode", False)):
            self._set_mask_edit_button_visible(False)
            self.glwidget.unsetCursor()
        else:
            self._set_mask_edit_button_visible(True)
            self.glwidget.setCursor(Qt.CrossCursor)
        if restore_view and self._mask_draw_previous_view:
            self._restore_mask_draw_previous_view()
        self.vis_fram(updata_color_bar=False, preserve_current_bboxes=True)
        if bool(getattr(self, "_mask_edit_mode", False)):
            self._set_status_message("地图绘制模式：点击左上角「绘制Mask」开始绘制一个新多边形")
        else:
            self._set_status_message("Mask绘制已关闭")

    def _enter_mask_edit_mode(self):
        if not hasattr(self, "_mask_draw_vertices"):
            self._init_mask_draw_state()
        if getattr(self, "_mask_draw_mode", False):
            self._leave_mask_draw_mode(restore_view=False)
        self._mask_edit_mode = True
        self._sync_mask_edit_action_checked(True)
        self._sync_mask_draw_action_checked(False)
        self._mask_draw_flatten_display = True
        self._mask_params["point_z"] = 0.0
        self._mask_drag_vertex = None
        self._mask_pan_last_pos = None
        try:
            self._mask_draw_previous_view = self.glwidget.cameraParams()
        except Exception:
            self._mask_draw_previous_view = None
        self._set_mask_edit_action_active(True)
        self._set_mask_draw_action_active(False)
        self._set_mask_edit_button_visible(True)
        self.glwidget.setCursor(Qt.CrossCursor)
        if self._mask_params.get("json_path") and os.path.isfile(self._mask_params.get("json_path")):
            self._mask_visible = True
            if self._mask_toggle_action is not None:
                self._mask_toggle_action.setChecked(True)
            self._rebuild_mask()
        self.vis_fram(updata_color_bar=False, preserve_current_bboxes=True)
        self._set_status_message("地图绘制模式：点击左上角「绘制Mask」开始绘制一个新多边形；也可编辑已有多边形")

    def _leave_mask_edit_mode(self, restore_view=False):
        self._mask_edit_mode = False
        if getattr(self, "_mask_draw_mode", False):
            self._leave_mask_draw_mode(restore_view=False, return_to_edit=False)
        self._sync_mask_edit_action_checked(False)
        self._mask_draw_flatten_display = False
        self._mask_drag_vertex = None
        self._mask_pan_last_pos = None
        self._clear_mask_hover_items()
        self._clear_mask_selected_items()
        self._set_mask_edit_action_active(False)
        self._set_mask_edit_button_visible(False)
        self.glwidget.unsetCursor()
        if restore_view and self._mask_draw_previous_view:
            self._restore_mask_draw_previous_view()
        self.vis_fram(updata_color_bar=False, preserve_current_bboxes=True)
        self._set_status_message("Mask编辑已关闭")

    def _restore_mask_draw_previous_view(self):
        view_data = self._mask_draw_previous_view
        if not view_data:
            return
        try:
            self.glwidget.setCameraPosition(
                pos=view_data.get("center"),
                distance=view_data.get("distance"),
                elevation=view_data.get("elevation"),
                azimuth=view_data.get("azimuth"),
            )
        except Exception:
            pass

    def _set_mask_draw_action_active(self, active):
        if hasattr(self, "_set_toolbar_action_active"):
            self._set_toolbar_action_active(getattr(self, "_mask_draw_action", None), active)
        btn = getattr(self, "mask_edit_btn", None)
        if btn is not None:
            btn.blockSignals(True)
            btn.setChecked(bool(active))
            btn.blockSignals(False)

    def _set_mask_edit_action_active(self, active):
        if hasattr(self, "_set_toolbar_action_active"):
            self._set_toolbar_action_active(getattr(self, "_mask_edit_action", None), active)

    def _sync_mask_draw_action_checked(self, checked):
        for action in (
            getattr(self, "_mask_draw_action", None),
            getattr(self, "_mask_draw_menu_action", None),
        ):
            if action is None:
                continue
            action.blockSignals(True)
            action.setChecked(bool(checked))
            action.blockSignals(False)
        btn = getattr(self, "mask_edit_btn", None)
        if btn is not None:
            btn.blockSignals(True)
            btn.setChecked(bool(checked))
            btn.blockSignals(False)

    def _sync_mask_edit_action_checked(self, checked):
        for action in (
            getattr(self, "_mask_edit_action", None),
            getattr(self, "_mask_edit_menu_action", None),
        ):
            if action is None:
                continue
            action.blockSignals(True)
            action.setChecked(bool(checked))
            action.blockSignals(False)

    def _set_mask_edit_button_visible(self, visible):
        btn = getattr(self, "mask_edit_btn", None)
        if btn is None:
            return
        if visible:
            if hasattr(self, "_update_mask_edit_button_geometry"):
                self._update_mask_edit_button_geometry()
            btn.show()
            btn.raise_()
        else:
            btn.hide()

    def _lock_mask_draw_view(self):
        dist = self.glwidget.opts.get("distance", 15)
        center = self.glwidget.opts.get("center", None)
        try:
            self.glwidget.setCameraPosition(pos=center, distance=dist, elevation=90, azimuth=0)
        except TypeError:
            self.glwidget.setCameraPosition(distance=dist, elevation=90, azimuth=0)

    def _clear_mask_draw_temp_items(self):
        for it in getattr(self, "_mask_draw_items", []):
            try:
                self.glwidget.removeItem(it)
            except Exception:
                pass
        self._mask_draw_items = []
        self._mask_draw_vertices = []
        self._clear_mask_hover_items()
        self.glwidget.update()

    def _load_mask_data_for_edit(self):
        json_path = str(self._mask_params.get("json_path", "") or "").strip()
        if not json_path or not os.path.isfile(json_path):
            return {"version": "PCDView", "flags": {}, "shapes": []}, json_path
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        if "shapes" not in data or not isinstance(data["shapes"], list):
            data["shapes"] = []
        data.setdefault("version", "PCDView")
        data.setdefault("flags", {})
        return data, json_path

    def _save_mask_data_for_edit(self, data, json_path=None):
        json_path = json_path or str(self._mask_params.get("json_path", "") or "").strip()
        if not json_path:
            return False
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._mask_params["json_path"] = json_path
        self._mask_visible = True
        if self._mask_toggle_action is not None:
            self._mask_toggle_action.setChecked(True)
        self._rebuild_mask()
        return True

    def _mask_polygon_shape_indices(self, data):
        out = []
        for i, shp in enumerate(data.get("shapes", [])):
            if str(shp.get("shape_type", "")).lower() != "polygon":
                continue
            pts = shp.get("points", [])
            if isinstance(pts, list) and len(pts) >= 3:
                out.append(i)
        return out

    def _mask_screen_vertices(self, pts2):
        z = float(self._mask_params.get("point_z", 0.0))
        arr = np.zeros((len(pts2), 3), dtype=np.float64)
        arr[:, 0:2] = np.asarray(pts2, dtype=np.float64)[:, :2]
        arr[:, 2] = z
        screen = world_to_screen(self.glwidget, arr)
        if screen is None:
            return None
        return np.asarray(screen, dtype=np.float64).reshape(-1, 2)

    @staticmethod
    def _point_segment_distance(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        denom = vx * vx + vy * vy
        if denom <= 1e-9:
            return float(((px - ax) ** 2 + (py - ay) ** 2) ** 0.5)
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
        cx, cy = ax + t * vx, ay + t * vy
        return float(((px - cx) ** 2 + (py - cy) ** 2) ** 0.5)

    @staticmethod
    def _polygon_area_abs(pts):
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
            return float("inf")
        x = arr[:, 0]
        y = arr[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)

    def _hit_mask_edit_target(self, screen_x, screen_y, vertex_eps=10.0, edge_eps=8.0):
        data, _ = self._load_mask_data_for_edit()
        best_shape_hit = None
        best_shape_area = None
        for shape_index in reversed(self._mask_polygon_shape_indices(data)):
            shp = data["shapes"][shape_index]
            pts = shp.get("points", [])
            screen = self._mask_screen_vertices(pts)
            if screen is None or len(screen) < 3:
                continue
            d = np.sqrt((screen[:, 0] - screen_x) ** 2 + (screen[:, 1] - screen_y) ** 2)
            if len(d) and float(np.nanmin(d)) <= vertex_eps:
                return {
                    "type": "vertex",
                    "shape_index": shape_index,
                    "vertex_index": int(np.nanargmin(d)),
                    "data": data,
                }
            best_edge = None
            best_edge_dist = None
            for i in range(len(screen)):
                j = (i + 1) % len(screen)
                dist = self._point_segment_distance(
                    screen_x, screen_y,
                    screen[i, 0], screen[i, 1],
                    screen[j, 0], screen[j, 1],
                )
                if best_edge_dist is None or dist < best_edge_dist:
                    best_edge_dist = dist
                    best_edge = i
            if best_edge_dist is not None and best_edge_dist <= edge_eps:
                return {"type": "edge", "shape_index": shape_index, "edge_index": int(best_edge), "data": data}
            try:
                if MplPath(np.asarray(pts, dtype=np.float64)[:, :2]).contains_point(self._screen_to_mask_xy(screen_x, screen_y)):
                    area = self._polygon_area_abs(pts)
                    if best_shape_area is None or area < best_shape_area:
                        best_shape_area = area
                        best_shape_hit = {"type": "shape", "shape_index": shape_index, "data": data}
            except Exception:
                pass
        return best_shape_hit

    def _update_mask_hover_from_screen(self, screen_x, screen_y):
        self._clear_mask_hover_items()
        hit = self._hit_mask_edit_target(screen_x, screen_y)
        self._mask_hover_shape_index = None
        self._mask_hover_vertex_index = None
        self._mask_hover_edge_index = None
        if hit is None:
            self._clear_mask_selected_items()
            self.glwidget.setCursor(Qt.CrossCursor)
            return
        self._mask_hover_shape_index = hit.get("shape_index")
        if hit["type"] == "vertex":
            self._clear_mask_selected_items()
            self._mask_hover_vertex_index = hit.get("vertex_index")
            pts = hit["data"]["shapes"][hit["shape_index"]].get("points", [])
            if 0 <= int(hit["vertex_index"]) < len(pts):
                self._add_mask_ring_marker(pts[int(hit["vertex_index"])], color=(1.0, 0.92, 0.0, 1.0), radius_px=12.0)
            self.glwidget.setCursor(Qt.PointingHandCursor)
            self._set_status_message("拖动顶点可编辑Mask；右键删除该点")
        elif hit["type"] == "edge":
            self._clear_mask_selected_items()
            self._mask_hover_edge_index = hit.get("edge_index")
            xy = self._screen_to_mask_xy(screen_x, screen_y)
            self._add_mask_cross_marker(xy, color=(1.0, 0.92, 0.0, 1.0), radius_px=10.0)
            self.glwidget.setCursor(Qt.PointingHandCursor)
            self._set_status_message("点击线段可在该位置插入新顶点")
        else:
            self._select_mask_shape(hit.get("shape_index"), hit.get("data"))
            self.glwidget.setCursor(Qt.OpenHandCursor)
            self._set_status_message("右键可删除多边形或修改标号")

    def _screen_to_mask_xy(self, screen_x, screen_y):
        ray = ray_from_screen(self.glwidget, screen_x, screen_y)
        if ray is None:
            return None
        pt = ray_plane_z_intersect(ray[0], ray[1], float(self._mask_params.get("point_z", 0.0)))
        if pt is None:
            return None
        return float(pt[0]), float(pt[1])

    def _mask_first_vertex_hit(self, screen_x, screen_y, threshold_px=12.0):
        if len(self._mask_draw_vertices) < 3:
            return False
        z = float(self._mask_params.get("point_z", 0.0))
        first = np.array([[self._mask_draw_vertices[0][0], self._mask_draw_vertices[0][1], z]], dtype=np.float64)
        screen = world_to_screen(self.glwidget, first)
        if screen is None or np.any(np.isnan(screen)):
            return False
        sx, sy = np.asarray(screen).reshape(-1, 2)[0]
        return float((sx - screen_x) ** 2 + (sy - screen_y) ** 2) ** 0.5 <= float(threshold_px)

    def _mask_draw_snap_xy(self, screen_x, screen_y):
        if self._mask_first_vertex_hit(screen_x, screen_y):
            return list(self._mask_draw_vertices[0])
        xy = self._screen_to_mask_xy(screen_x, screen_y)
        return list(xy) if xy is not None else None

    def _add_mask_draw_vertex_from_screen(self, screen_x, screen_y):
        if self._mask_first_vertex_hit(screen_x, screen_y):
            self._finish_mask_polygon()
            return
        xy = self._screen_to_mask_xy(screen_x, screen_y)
        if xy is None:
            self._set_status_message("无法把当前点击位置投影到Z=0平面")
            return
        self._mask_draw_vertices.append([float(xy[0]), float(xy[1])])
        self._refresh_mask_draw_temp_items()
        self._set_status_message("Mask绘制：已添加 {} 个顶点".format(len(self._mask_draw_vertices)))

    def _refresh_mask_draw_temp_items(self):
        for it in getattr(self, "_mask_draw_items", []):
            try:
                self.glwidget.removeItem(it)
            except Exception:
                pass
        self._mask_draw_items = []
        if not self._mask_draw_vertices:
            self.glwidget.update()
            return
        z = float(self._mask_params.get("point_z", 0.0))
        pts2 = np.asarray(self._mask_draw_vertices, dtype=np.float32)
        pts3 = np.zeros((len(pts2), 3), dtype=np.float32)
        pts3[:, 0] = pts2[:, 0]
        pts3[:, 1] = pts2[:, 1]
        pts3[:, 2] = z
        point_color = self._qrgb_to_rgba_f(self._mask_params.get("point_color", (255, 0, 0)))
        line_color = self._qrgb_to_rgba_f(self._mask_params.get("line_color", (0, 100, 0)))
        p_item = gl.GLScatterPlotItem(
            pos=pts3,
            color=point_color,
            size=float(self._mask_params.get("point_size", 8.0)),
        )
        self._mask_draw_items.append(p_item)
        self.glwidget.addItem(p_item)
        if len(pts3) >= 1 and self._mask_draw_hover_xy is not None:
            hover = np.array([[self._mask_draw_hover_xy[0], self._mask_draw_hover_xy[1], z]], dtype=np.float32)
            draw_line = np.vstack([pts3, hover])
            l_item = gl.GLLinePlotItem(
                pos=draw_line,
                color=line_color,
                width=float(self._mask_params.get("line_width", 2.0)),
                antialias=True,
                mode="line_strip",
            )
            self._mask_draw_items.append(l_item)
            self.glwidget.addItem(l_item)
        elif len(pts3) >= 2:
            draw_line = pts3
            l_item = gl.GLLinePlotItem(
                pos=draw_line,
                color=line_color,
                width=float(self._mask_params.get("line_width", 2.0)),
                antialias=True,
                mode="line_strip",
            )
            self._mask_draw_items.append(l_item)
            self.glwidget.addItem(l_item)
        self.glwidget.update()

    def _finish_mask_polygon(self):
        if len(getattr(self, "_mask_draw_vertices", [])) < 3:
            self._set_status_message("至少需要3个顶点才能闭合Mask")
            return
        label, ok = QInputDialog.getText(self, "命名Mask", "Mask名称:")
        if not ok:
            return
        label = str(label).strip()
        if not label:
            QMessageBox.warning(self, "命名Mask", "Mask名称不能为空。")
            return
        json_path = self._ensure_mask_json_path()
        if not json_path:
            return
        try:
            self._append_mask_polygon_to_json(json_path, label, self._mask_draw_vertices)
        except Exception as exc:
            QMessageBox.warning(self, "保存Mask失败", str(exc))
            return
        self._mask_params["json_path"] = json_path
        self._mask_visible = True
        if self._mask_toggle_action is not None:
            self._mask_toggle_action.setChecked(True)
        self._clear_mask_draw_temp_items()
        self._rebuild_mask()
        self._leave_mask_draw_mode(restore_view=False, return_to_edit=True)
        self._set_status_message("Mask已保存: {}。如需继续绘制，请再次点击左上角「绘制Mask」".format(label))

    def _ensure_mask_json_path(self):
        json_path = str(self._mask_params.get("json_path", "") or "").strip()
        if json_path:
            return json_path
        start_dir = ""
        if getattr(self, "pcd_file", None):
            start_dir = os.path.dirname(os.path.abspath(self.pcd_file))
        path, _ = QFileDialog.getSaveFileName(None, "保存Mask JSON", start_dir, "JSON Files (*.json)")
        if not path:
            return ""
        if not path.lower().endswith(".json"):
            path += ".json"
        return path

    def _append_mask_polygon_to_json(self, json_path, label, vertices):
        data = {}
        if os.path.isfile(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        if "shapes" not in data or not isinstance(data["shapes"], list):
            data["shapes"] = []
        data.setdefault("version", "PCDView")
        data.setdefault("flags", {})
        data["shapes"].append({
            "label": label,
            "points": [[float(x), float(y)] for x, y in vertices],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        })
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _insert_mask_vertex_on_edge(self, hit, xy):
        data = hit["data"]
        shape_index = hit["shape_index"]
        edge_index = hit["edge_index"]
        pts = data["shapes"][shape_index].get("points", [])
        insert_at = int(edge_index) + 1
        pts.insert(insert_at, [float(xy[0]), float(xy[1])])
        data["shapes"][shape_index]["points"] = pts
        self._save_mask_data_for_edit(data)
        self._mask_drag_vertex = {"shape_index": shape_index, "vertex_index": insert_at, "data": data}
        self._set_status_message("已在线段中插入新顶点")

    def _move_mask_vertex_to_screen(self, screen_x, screen_y):
        drag = self._mask_drag_vertex
        if not drag:
            return None
        xy = self._screen_to_mask_xy(screen_x, screen_y)
        if xy is None:
            return None
        data, json_path = self._load_mask_data_for_edit()
        si = int(drag["shape_index"])
        vi = int(drag["vertex_index"])
        if si >= len(data.get("shapes", [])):
            return None
        pts = data["shapes"][si].get("points", [])
        if vi >= len(pts):
            return None
        pts[vi] = [float(xy[0]), float(xy[1])]
        data["shapes"][si]["points"] = pts
        self._save_mask_data_for_edit(data, json_path)
        self._clear_mask_hover_items()
        self._add_mask_ring_marker(xy, color=(1.0, 0.92, 0.0, 1.0), radius_px=12.0)
        return xy

    def _delete_mask_vertex(self, hit):
        data = hit["data"]
        si = hit["shape_index"]
        vi = hit["vertex_index"]
        pts = data["shapes"][si].get("points", [])
        if len(pts) <= 3:
            self._set_status_message("多边形至少保留3个顶点")
            return
        pts.pop(int(vi))
        data["shapes"][si]["points"] = pts
        self._save_mask_data_for_edit(data)
        self._set_status_message("已删除顶点")

    def _show_mask_shape_context_menu(self, hit, global_pos):
        if hit is None or hit.get("shape_index") is None:
            return False
        data = hit.get("data")
        if not isinstance(data, dict):
            data, _ = self._load_mask_data_for_edit()
        shape_index = int(hit["shape_index"])
        shapes = data.get("shapes", [])
        if shape_index < 0 or shape_index >= len(shapes):
            return False
        shape = shapes[shape_index]
        label = str(shape.get("label", "") or "")

        menu = QMenu(self)
        rename_action = QAction("修改标号", menu)
        delete_action = QAction("删除多边形", menu)
        menu.addAction(rename_action)
        menu.addAction(delete_action)
        chosen = menu.exec_(global_pos)
        if chosen == rename_action:
            new_label, ok = QInputDialog.getText(self, "修改标号", "Mask名称:", text=label)
            if not ok:
                return True
            new_label = str(new_label).strip()
            if not new_label:
                QMessageBox.warning(self, "修改标号", "Mask名称不能为空。")
                return True
            shape["label"] = new_label
            self._save_mask_data_for_edit(data)
            self._set_status_message("Mask标号已修改: {}".format(new_label))
            return True
        if chosen == delete_action:
            ret = QMessageBox.question(
                self,
                "删除多边形",
                "确定删除多边形 '{}' 吗？".format(label or shape_index),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return True
            shapes.pop(shape_index)
            data["shapes"] = shapes
            self._save_mask_data_for_edit(data)
            self._clear_mask_hover_items()
            self._set_status_message("Mask多边形已删除")
            return True
        return True

    def _show_mask_vertex_context_menu(self, hit, global_pos):
        if hit is None or hit.get("type") != "vertex":
            return False
        menu = QMenu(self)
        delete_action = QAction("删除该点", menu)
        menu.addAction(delete_action)
        chosen = menu.exec_(global_pos)
        if chosen == delete_action:
            self._delete_mask_vertex(hit)
            self._clear_mask_hover_items()
            self._clear_mask_selected_items()
            return True
        return True

    def _handle_mask_draw_key_event(self, event):
        if not (bool(getattr(self, "_mask_draw_mode", False)) or bool(getattr(self, "_mask_edit_mode", False))):
            return False
        if event.type() != QEvent.KeyPress:
            return False
        if event.key() == Qt.Key_Escape:
            if bool(getattr(self, "_mask_draw_mode", False)):
                self._clear_mask_draw_temp_items()
                self._set_status_message("已取消当前Mask多边形")
            else:
                self._clear_mask_hover_items()
                self._clear_mask_selected_items()
            return True
        if bool(getattr(self, "_mask_draw_mode", False)) and event.key() == Qt.Key_Backspace:
            return self._delete_last_mask_draw_vertex()
        return False

    def _delete_last_mask_draw_vertex(self):
        if not bool(getattr(self, "_mask_draw_mode", False)):
            return False
        vertices = getattr(self, "_mask_draw_vertices", [])
        if not vertices:
            return True
        vertices.pop()
        self._refresh_mask_draw_temp_items()
        self._set_status_message("Mask绘制：已删除上一个顶点")
        return True

    def _handle_mask_draw_mouse_event(self, event, mx, my):
        draw_mode = bool(getattr(self, "_mask_draw_mode", False))
        edit_mode = bool(getattr(self, "_mask_edit_mode", False))
        if not (draw_mode or edit_mode):
            return False
        if event.type() == QEvent.MouseButtonDblClick:
            if draw_mode and event.button() == Qt.RightButton:
                self._mask_draw_click_start = None
                self._add_mask_draw_vertex_from_screen(mx, my)
                return True
            if edit_mode:
                return True
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                if draw_mode:
                    return False
                if edit_mode:
                    hit = self._hit_mask_edit_target(mx, my)
                    if hit is None:
                        self._clear_mask_selected_items()
                        return False
                    if hit["type"] == "vertex":
                        self._mask_drag_vertex = hit
                        return True
                    if hit["type"] == "edge":
                        xy = self._screen_to_mask_xy(mx, my)
                        if xy is not None:
                            self._insert_mask_vertex_on_edge(hit, xy)
                        return True
                    if hit["type"] == "shape":
                        self._select_mask_shape(hit.get("shape_index"), hit.get("data"))
                        return True
                    return True
            if event.button() == Qt.RightButton:
                if draw_mode:
                    self._mask_draw_click_start = None
                    self._add_mask_draw_vertex_from_screen(mx, my)
                    return True
                if edit_mode:
                    hit = self._hit_mask_edit_target(mx, my)
                    if hit is not None and hit.get("type") == "vertex":
                        return self._show_mask_vertex_context_menu(hit, event.globalPos())
                    if hit is not None and hit.get("type") == "shape":
                        self._select_mask_shape(hit.get("shape_index"), hit.get("data"))
                        return self._show_mask_shape_context_menu(hit, event.globalPos())
                    return True
            if event.button() == Qt.MiddleButton:
                self._mask_pan_last_pos = event.pos()
                return True
        if event.type() == QEvent.MouseButtonRelease:
            if draw_mode and event.button() == Qt.RightButton and self._mask_draw_click_start is not None:
                self._mask_draw_click_start = None
                return True
            if draw_mode and event.button() == Qt.LeftButton:
                return False
            if event.button() == Qt.LeftButton and self._mask_drag_vertex is not None:
                xy = self._move_mask_vertex_to_screen(mx, my)
                self._mask_drag_vertex = None
                if xy is not None:
                    self._clear_mask_hover_items()
                    self._add_mask_ring_marker(xy, color=(1.0, 0.92, 0.0, 1.0), radius_px=12.0)
                self._set_status_message("Mask顶点已更新")
                return True
            if event.button() == Qt.LeftButton and self._mask_pan_last_pos is not None:
                self._mask_pan_last_pos = None
                return True
            if (not draw_mode) and edit_mode and event.button() == Qt.LeftButton:
                return False
            if event.button() == Qt.MiddleButton:
                self._mask_pan_last_pos = None
                return True
            return True
        if event.type() == QEvent.MouseMove:
            if draw_mode and event.buttons() & Qt.LeftButton:
                return False
            if (not draw_mode) and edit_mode and event.buttons() & Qt.LeftButton and self._mask_drag_vertex is not None:
                self._move_mask_vertex_to_screen(mx, my)
                return True
            if (not draw_mode) and edit_mode and event.buttons() & Qt.LeftButton:
                return False
            if event.buttons() & Qt.MiddleButton and self._mask_pan_last_pos is not None:
                diff = event.pos() - self._mask_pan_last_pos
                self._mask_pan_last_pos = event.pos()
                self.glwidget.pan(diff.x(), diff.y(), 0, relative="view-upright")
                self._lock_mask_draw_view()
                return True
            if draw_mode and self._mask_draw_vertices:
                self._mask_draw_hover_xy = self._mask_draw_snap_xy(mx, my)
                self._clear_mask_hover_items()
                if self._mask_first_vertex_hit(mx, my):
                    self.glwidget.setCursor(Qt.PointingHandCursor)
                    self._add_mask_ring_marker(self._mask_draw_vertices[0], color=(1.0, 0.92, 0.0, 1.0), radius_px=13.0)
                else:
                    self.glwidget.setCursor(Qt.CrossCursor)
                    self._add_mask_cross_marker(self._mask_draw_hover_xy, color=(1.0, 0.92, 0.0, 1.0), radius_px=8.0)
                self._refresh_mask_draw_temp_items()
            elif (not draw_mode) and edit_mode:
                self._update_mask_hover_from_screen(mx, my)
            return True
        if event.type() == QEvent.Wheel:
            if edit_mode:
                return False
            delta = event.angleDelta().x() or event.angleDelta().y()
            if delta:
                self.glwidget.opts["distance"] *= 0.999 ** delta
                self._lock_mask_draw_view()
                self.glwidget.update()
            return True
        return False
