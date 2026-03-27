from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)


class PlaneParamDialog(QDialog):
    GRID_DEFAULT_COLOR = (180, 180, 180)
    FACE_DEFAULT_COLOR = (180, 180, 180)
    GRID_DEFAULT_ALPHA_UI = 100.0
    FACE_DEFAULT_ALPHA_UI = 200.0

    def __init__(
        self,
        parent=None,
        plane_type_default="网格",
        plane_length_default=100.0,
        plane_width_default=100.0,
        grid_spacing_default=10.0,
        center_default=(0.0, 0.0, -1.7),
        color_default=(180, 180, 180),
        alpha_default=120,
    ):
        super().__init__(parent)
        self.setWindowTitle("修改平面参数")

        layout = QFormLayout(self)

        self.type_combo = QComboBox(self)
        self.type_combo.addItems(["网格", "面"])
        if plane_type_default in ["网格", "面"]:
            self.type_combo.setCurrentText(plane_type_default)
        layout.addRow("平面类型", self.type_combo)

        self.length_spin = QDoubleSpinBox(self)
        self.length_spin.setDecimals(3)
        self.length_spin.setRange(0.001, 1e6)
        self.length_spin.setSingleStep(1.0)
        self.length_spin.setValue(float(plane_length_default))
        layout.addRow("平面长(m)", self.length_spin)

        self.width_spin = QDoubleSpinBox(self)
        self.width_spin.setDecimals(3)
        self.width_spin.setRange(0.001, 1e6)
        self.width_spin.setSingleStep(1.0)
        self.width_spin.setValue(float(plane_width_default))
        layout.addRow("平面宽(m)", self.width_spin)

        self.spacing_spin = QDoubleSpinBox(self)
        self.spacing_spin.setDecimals(3)
        self.spacing_spin.setRange(0.001, 1e6)
        self.spacing_spin.setSingleStep(1.0)
        self.spacing_spin.setValue(float(grid_spacing_default))
        layout.addRow("每个小网格长度(m)", self.spacing_spin)

        cx, cy, cz = center_default
        self.cx_spin = QDoubleSpinBox(self)
        self.cx_spin.setDecimals(3)
        self.cx_spin.setRange(-1e6, 1e6)
        self.cx_spin.setSingleStep(0.1)
        self.cx_spin.setValue(float(cx))

        self.cy_spin = QDoubleSpinBox(self)
        self.cy_spin.setDecimals(3)
        self.cy_spin.setRange(-1e6, 1e6)
        self.cy_spin.setSingleStep(0.1)
        self.cy_spin.setValue(float(cy))

        self.cz_spin = QDoubleSpinBox(self)
        self.cz_spin.setDecimals(3)
        self.cz_spin.setRange(-1e6, 1e6)
        self.cz_spin.setSingleStep(0.1)
        self.cz_spin.setValue(float(cz))

        center_row = QHBoxLayout()
        center_row.addWidget(self.cx_spin)
        center_row.addWidget(self.cy_spin)
        center_row.addWidget(self.cz_spin)
        layout.addRow("中心点(x,y,z)", center_row)

        # 颜色选择
        self.color_btn = QPushButton("选择颜色", self)
        self._color = QColor(*color_default)
        self._alpha = int(alpha_default)
        self._update_color_preview()
        self.color_btn.clicked.connect(self._choose_color)
        layout.addRow("颜色", self.color_btn)

        alpha_row = QHBoxLayout()
        self.alpha_spin = QDoubleSpinBox(self)
        self.alpha_spin.setDecimals(0)
        self.alpha_spin.setRange(0, 255)
        self.alpha_spin.setSingleStep(5)
        self.alpha_spin.setValue(float(self._alpha))
        alpha_row.addWidget(QLabel("透明度(0~255，255最透明)"))
        alpha_row.addWidget(self.alpha_spin)
        layout.addRow("透明度", alpha_row)

        self.btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)
        layout.addRow(self.btn_box)

        # 切换平面类型时，按类型自动套用默认灰色+默认透明度
        self.type_combo.currentTextChanged.connect(self._on_plane_type_changed)

        # 若传入值不合法，按类型默认值初始化
        if alpha_default is None:
            self.alpha_spin.setValue(self._default_alpha_for_type(self.type_combo.currentText()))

    def _choose_color(self):
        c = QColorDialog.getColor(self._color, self, "选择颜色")
        if not c.isValid():
            return
        self._color = c
        self._update_color_preview()

    def _update_color_preview(self):
        r = int(self._color.red())
        g = int(self._color.green())
        b = int(self._color.blue())
        # 用按钮背景色做预览
        self.color_btn.setStyleSheet("QPushButton { background-color: rgb(%d,%d,%d); }" % (r, g, b))

    @classmethod
    def _default_alpha_for_type(cls, plane_type):
        return cls.GRID_DEFAULT_ALPHA_UI if plane_type == "网格" else cls.FACE_DEFAULT_ALPHA_UI

    @classmethod
    def _default_color_for_type(cls, plane_type):
        return cls.GRID_DEFAULT_COLOR if plane_type == "网格" else cls.FACE_DEFAULT_COLOR

    def _on_plane_type_changed(self, plane_type):
        # 类型切换时应用该类型默认灰色+默认透明度
        dr, dg, db = self._default_color_for_type(plane_type)
        self._color = QColor(int(dr), int(dg), int(db))
        self._update_color_preview()
        self.alpha_spin.setValue(float(self._default_alpha_for_type(plane_type)))

    def get_params(self):
        return {
            "plane_type": self.type_combo.currentText(),
            "plane_length": float(self.length_spin.value()),
            "plane_width": float(self.width_spin.value()),
            "grid_spacing": float(self.spacing_spin.value()),
            "center": (
                float(self.cx_spin.value()),
                float(self.cy_spin.value()),
                float(self.cz_spin.value()),
            ),
            "color_rgb": (int(self._color.red()), int(self._color.green()), int(self._color.blue())),
            "alpha": float(self.alpha_spin.value()),
        }

