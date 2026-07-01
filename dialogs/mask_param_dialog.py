import json
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
)


class MaskParamDialog(QDialog):
    def __init__(self, parent=None, params=None, on_change=None, owner=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("Mask设置")
        self._params = params or {}
        self._on_change = on_change
        self._owner = owner if owner is not None else parent

        layout = QFormLayout(self)

        # JSON 文件选择
        self.json_path_edit = QLabel(self._params.get("json_path", "未选择"))
        self.json_path_edit.setWordWrap(True)
        file_row = QHBoxLayout()
        self.pick_json_btn = QPushButton("选择JSON文件")
        self.pick_json_btn.clicked.connect(self._pick_json_file)
        file_row.addWidget(self.pick_json_btn)
        file_row.addWidget(self.json_path_edit, 1)
        layout.addRow("Mask文件", file_row)

        self.point_size_spin = QDoubleSpinBox(self)
        self.point_size_spin.setDecimals(2)
        self.point_size_spin.setRange(0.1, 100.0)
        self.point_size_spin.setValue(float(self._params.get("point_size", 4.0)))
        self.point_size_spin.valueChanged.connect(lambda *_: self._emit_change("point_size"))
        layout.addRow("点大小", self.point_size_spin)

        self.line_width_spin = QDoubleSpinBox(self)
        self.line_width_spin.setDecimals(2)
        self.line_width_spin.setRange(0.1, 100.0)
        self.line_width_spin.setValue(float(self._params.get("line_width", 2.0)))
        self.line_width_spin.valueChanged.connect(lambda *_: self._emit_change("line_width"))
        layout.addRow("线粗细", self.line_width_spin)

        self.z_value_spin = QDoubleSpinBox(self)
        self.z_value_spin.setDecimals(3)
        self.z_value_spin.setRange(-1e6, 1e6)
        self.z_value_spin.setSingleStep(0.1)
        self.z_value_spin.setValue(float(self._params.get("point_z", 0.0)))
        self.z_value_spin.valueChanged.connect(lambda *_: self._emit_change("point_z"))
        layout.addRow("JSON点Z值", self.z_value_spin)

        self.keep_inside_checkbox = QCheckBox("仅保留圈内点")
        self.keep_inside_checkbox.setChecked(bool(self._params.get("keep_inside_points", False)))
        self.keep_inside_checkbox.stateChanged.connect(lambda *_: self._emit_change("keep_inside_points"))
        layout.addRow("", self.keep_inside_checkbox)

        self.export_x_min_spin = QDoubleSpinBox(self)
        self.export_x_min_spin.setDecimals(3)
        self.export_x_min_spin.setRange(-1e6, 1e6)
        self.export_x_min_spin.setValue(float(self._params.get("export_x_min", -40.0)))
        self.export_x_max_spin = QDoubleSpinBox(self)
        self.export_x_max_spin.setDecimals(3)
        self.export_x_max_spin.setRange(-1e6, 1e6)
        self.export_x_max_spin.setValue(float(self._params.get("export_x_max", 110.0)))
        x_range_row = QHBoxLayout()
        x_range_row.addWidget(self.export_x_min_spin)
        x_range_row.addWidget(QLabel("到"))
        x_range_row.addWidget(self.export_x_max_spin)
        layout.addRow("导出X轴范围", x_range_row)

        self.export_y_min_spin = QDoubleSpinBox(self)
        self.export_y_min_spin.setDecimals(3)
        self.export_y_min_spin.setRange(-1e6, 1e6)
        self.export_y_min_spin.setValue(float(self._params.get("export_y_min", -70.0)))
        self.export_y_max_spin = QDoubleSpinBox(self)
        self.export_y_max_spin.setDecimals(3)
        self.export_y_max_spin.setRange(-1e6, 1e6)
        self.export_y_max_spin.setValue(float(self._params.get("export_y_max", 80.0)))
        y_range_row = QHBoxLayout()
        y_range_row.addWidget(self.export_y_min_spin)
        y_range_row.addWidget(QLabel("到"))
        y_range_row.addWidget(self.export_y_max_spin)
        layout.addRow("导出Y轴范围", y_range_row)

        self.export_pixel_spin = QSpinBox(self)
        self.export_pixel_spin.setRange(1, 1000)
        self.export_pixel_spin.setValue(int(self._params.get("export_pixel", 10)))
        layout.addRow("导出像素(份/m)", self.export_pixel_spin)

        self.export_tanway_btn = QPushButton("导出tanway_txt")
        self.export_tanway_btn.clicked.connect(self._export_tanway_txt)
        self.export_npy_btn = QPushButton("导出npy")
        self.export_npy_btn.clicked.connect(self._export_npy)
        export_row = QHBoxLayout()
        export_row.addWidget(self.export_tanway_btn)
        export_row.addWidget(self.export_npy_btn)
        layout.addRow("", export_row)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        layout.addRow("文件信息", self.info_label)
        self._update_json_info()
        self._apply_export_defaults_from_json(force=False)

        self.btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)
        layout.addRow(self.btn_box)

    def _pick_json_file(self):
        path, _ = QFileDialog.getOpenFileName(None, "选择Mask JSON文件", "", "JSON Files (*.json)")
        if path:
            self.json_path_edit.setText(path)
            self._update_json_info()
            self._apply_export_defaults_from_json(force=True)
            self._emit_change("json_path")

    def _update_json_info(self):
        path = self.json_path_edit.text().strip()
        if not path or path == "未选择" or not os.path.isfile(path):
            self.info_label.setText("未加载有效JSON文件。")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            self.info_label.setText("已读取JSON：shapes数量=%d" % len(shapes))
        except Exception as e:
            self.info_label.setText("JSON读取失败: %s" % e)

    def _apply_export_defaults_from_json(self, force=False):
        path = self.json_path_edit.text().strip()
        if not path or path == "未选择" or not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        pcd_range = data.get("pcd_range")
        if isinstance(pcd_range, list) and len(pcd_range) >= 4:
            if force or not any(k in self._params for k in ("export_x_min", "export_x_max", "export_y_min", "export_y_max")):
                self.export_x_min_spin.setValue(float(pcd_range[0]))
                self.export_y_min_spin.setValue(float(pcd_range[1]))
                self.export_x_max_spin.setValue(float(pcd_range[2]))
                self.export_y_max_spin.setValue(float(pcd_range[3]))

        img_resolution = data.get("img_resolution")
        if img_resolution is not None and (force or "export_pixel" not in self._params):
            self.export_pixel_spin.setValue(max(1, int(float(img_resolution))))

    def _export_tanway_txt(self):
        owner = self._owner
        if owner is None or not hasattr(owner, "_export_mask_tanway_txt"):
            QMessageBox.warning(self, "导出tanway_txt", "当前窗口不支持导出。")
            return
        owner._export_mask_tanway_txt(self.get_params())

    def _export_npy(self):
        owner = self._owner
        if owner is None or not hasattr(owner, "_export_mask_npy"):
            QMessageBox.warning(self, "导出npy", "当前窗口不支持导出。")
            return
        owner._export_mask_npy(self.get_params())

    def get_params(self):
        return {
            "json_path": self.json_path_edit.text().strip() if self.json_path_edit.text().strip() != "未选择" else "",
            "point_size": float(self.point_size_spin.value()),
            "line_width": float(self.line_width_spin.value()),
            "point_z": float(self.z_value_spin.value()),
            "keep_inside_points": bool(self.keep_inside_checkbox.isChecked()),
            "export_x_min": float(self.export_x_min_spin.value()),
            "export_x_max": float(self.export_x_max_spin.value()),
            "export_y_min": float(self.export_y_min_spin.value()),
            "export_y_max": float(self.export_y_max_spin.value()),
            "export_pixel": int(self.export_pixel_spin.value()),
        }

    def _emit_change(self, key):
        if self._on_change is None:
            return
        try:
            self._on_change(self.get_params(), key)
        except Exception:
            pass
