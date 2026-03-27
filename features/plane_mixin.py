import numpy as np
import pyqtgraph.opengl as gl

from PyQt5 import QtGui

from dialogs.plane_param_dialog import PlaneParamDialog


class PlaneMixin:
    def _toggle_add_plane(self, checked=False):
        """添加/移除平面（用于工具栏按钮：点击一次出现，点击一次消失）"""
        if checked:
            # 只用当前参数创建，不弹对话框
            self._create_plane_from_params(self._plane_params)
        else:
            self._remove_plane()

    def _open_plane_param_dialog(self):
        """弹出修改平面参数窗口，返回参数字典；取消返回 None。"""
        p = self._plane_params or {}
        plane_type = p.get("plane_type", "网格")
        dlg = PlaneParamDialog(
            self,
            plane_type_default=plane_type,
            plane_length_default=float(p.get("plane_length", 100.0)),
            plane_width_default=float(p.get("plane_width", 100.0)),
            grid_spacing_default=float(p.get("grid_spacing", 10.0)),
            center_default=p.get("center", (0.0, 0.0, -1.7)),
            color_default=p.get("color_rgb", (180, 180, 180)),
            alpha_default=float(
                p.get("alpha", PlaneParamDialog._default_alpha_for_type(plane_type))
            ),
        )
        result = dlg.exec_()
        if result != 1:  # QDialog.Accepted == 1
            return None
        return dlg.get_params()

    def _modify_plane_params(self):
        """Tools -> 修改平面参数：修改参数，并在平面已存在时立即重建。"""
        params = self._open_plane_param_dialog()
        if params is None:
            return
        self._plane_params = params
        if self._plane_item is not None:
            self._create_plane_from_params(params)

    def _remove_plane(self):
        if self._plane_item is None:
            return
        try:
            if isinstance(self._plane_item, (list, tuple)):
                for it in self._plane_item:
                    if it is not None:
                        self.glwidget.removeItem(it)
            else:
                self.glwidget.removeItem(self._plane_item)
        finally:
            self._plane_item = None

    def _create_plane_from_params(self, params):
        """根据参数在 x-y 平面创建一个水平平面（z=中心点 z）。"""
        if not params:
            return
        self._remove_plane()

        plane_type = params.get("plane_type", "网格")
        plane_length = float(params.get("plane_length", 100.0))
        plane_width = float(params.get("plane_width", 100.0))
        grid_spacing = float(params.get("grid_spacing", 10.0))
        cx, cy, cz = params.get("center", (0.0, 0.0, -1.7))
        color_rgb = params.get("color_rgb", (180, 180, 180))
        alpha = float(params.get("alpha", 120.0))

        if plane_length <= 0 or plane_width <= 0:
            return
        if grid_spacing <= 0:
            grid_spacing = 10.0

        half_l = plane_length / 2.0
        half_w = plane_width / 2.0

        r, g, b = [int(x) for x in color_rgb]
        # UI 透明度定义：255 最透明；数值越大越透明
        ui_alpha = int(max(0, min(255, alpha)))
        a = 255 - ui_alpha

        if plane_type == "网格":
            grid = gl.GLGridItem()
            grid.setSpacing(x=grid_spacing, y=grid_spacing, z=1)
            grid.setSize(x=plane_length, y=plane_width, z=1)
            grid.setColor((r, g, b, a))
            self.glwidget.addItem(grid)
            grid.translate(cx, cy, cz)
            self._plane_item = grid
        else:
            verts = np.array(
                [
                    [-half_l, -half_w, 0.0],
                    [half_l, -half_w, 0.0],
                    [half_l, half_w, 0.0],
                    [-half_l, half_w, 0.0],
                ],
                dtype=np.float32,
            )
            faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            md = gl.MeshData(vertexes=verts, faces=faces)
            mesh = gl.GLMeshItem(
                meshdata=md,
                smooth=False,
                drawEdges=False,
                drawFaces=True,
                # GLMeshItem 在 paint() 中会直接 glColor4f(*color)。
                # 若传整数元组(0~255)可能被钳制/失真，所以这里传 QColor 以确保透明度正确归一化。
                color=QtGui.QColor(r, g, b, a),
                glOptions="translucent",
            )
            self.glwidget.addItem(mesh)
            mesh.translate(cx, cy, cz)
            self._plane_item = mesh

        self.glwidget.update()

