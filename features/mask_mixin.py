import json
import os

import numpy as np
import pyqtgraph.opengl as gl
from matplotlib.path import Path as MplPath

from PyQt5.QtWidgets import QMessageBox

from dialogs.mask_param_dialog import MaskParamDialog


class MaskMixin:
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

    def _load_mask_shapes(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("shapes", [])

    def _build_mask_items(self, shapes):
        items = []
        point_color = self._qrgb_to_rgba_f(self._mask_params.get("point_color", (255, 90, 90)))
        line_color = self._qrgb_to_rgba_f(self._mask_params.get("line_color", (255, 255, 0)))
        point_size = float(self._mask_params.get("point_size", 4.0))
        line_width = float(self._mask_params.get("line_width", 2.0))
        point_z = float(self._mask_params.get("point_z", 0.0))

        for shp in shapes:
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

            # 所有类型都绘制点（便于观察顶点）
            p_item = gl.GLScatterPlotItem(pos=arr3, color=point_color, size=point_size)
            items.append(p_item)

            shp_type = str(shp.get("shape_type", "")).lower()
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
                self.frame_info_label.setText(msg)
                QMessageBox.warning(self, "Mask", msg)
                return
            self._mask_visible = True
            self.frame_info_label.setText(msg)
        else:
            self._clear_mask_items()
            self._mask_visible = False
            self.frame_info_label.setText("Mask已关闭")

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
                self.vis_fram(updata_color_bar=False)
            elif _key == "keep_inside_points" and not keep_inside:
                # 取消仅保留圈内点：恢复点云
                self.vis_fram(updata_color_bar=False)

            # 若 keep_inside_points 未开启，则 style/point_size/颜色改动只重建 mask，不刷新点云
            # 以实现“对主界面联动拆分”的效果。

        dlg = MaskParamDialog(self, self._mask_params, on_change=_on_change_live)
        if dlg.exec_() != 1:  # QDialog.Accepted
            return
        self._mask_params = dlg.get_params()
        # 若当前Mask处于显示状态，参数修改后立即重建生效
        if self._mask_visible:
            ok, msg = self._rebuild_mask()
            self.frame_info_label.setText(msg)
        # 保证“仅保留圈内点”在确认后立即生效
        self.vis_fram(updata_color_bar=False)

