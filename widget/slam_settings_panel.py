import os

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)


class SlamSettingsPanel(QFrame):
    """Compact SLAM settings overlay displayed inside the 3D view."""

    applyRequested = pyqtSignal()
    panelGeometryChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_directory = ""
        self.setObjectName("SlamSettingsPanel")
        self.setStyleSheet(
            "QFrame#SlamSettingsPanel { background-color: rgba(255, 255, 255, 242); "
            "border: 1px solid #c8ccd2; border-radius: 7px; } "
            "QLabel { color: #1c1e21; background: transparent; } "
            "QLabel#SlamPanelTitle { font-weight: bold; font-size: 14px; } "
            "QLabel#SlamPanelStatus { color: #c62828; font-size: 11px; } "
            "QWidget#SlamPathRow, QWidget#SlamHistoryRow { background: transparent; } "
            "QCheckBox { color: #1c1e21; background: transparent; }"
        )

        title = QLabel("SLAM 设置", self)
        title.setObjectName("SlamPanelTitle")
        self.collapse_button = QPushButton("隐藏", self)
        self.collapse_button.setToolTip("隐藏/显示 SLAM 设置内容")
        self.collapse_button.setStyleSheet(
            "QPushButton { background-color: #e4e6eb; color: #1c1e21; "
            "padding: 4px 6px; font-size: 12px; } "
            "QPushButton:hover { background-color: #d8dadf; }"
        )
        self.collapse_button.clicked.connect(self._toggle_collapsed)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.collapse_button)

        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText("选择位姿 TXT")
        browse_button = QPushButton("...", self)
        browse_button.setFixedWidth(30)
        browse_button.setToolTip("加载位姿 TXT")
        browse_button.clicked.connect(self._browse)
        path_row = QWidget(self)
        path_row.setObjectName("SlamPathRow")
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(6)
        path_layout.addWidget(QLabel("位姿文件：", self))
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(browse_button)

        self.history_spin = QSpinBox(self)
        self.history_spin.setRange(0, 1000)
        self.history_spin.setSuffix(" 帧")
        self.history_spin.setToolTip("0 表示只显示当前帧")
        self.history_spin.setMinimumWidth(68)
        self.history_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        history_row = QWidget(self)
        history_row.setObjectName("SlamHistoryRow")
        history_layout = QHBoxLayout(history_row)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(6)
        history_layout.addWidget(QLabel("叠加历史帧：", self))
        history_layout.addWidget(self.history_spin, 1)

        self.history_transparent_check = QCheckBox("历史帧透明", self)
        self.history_transparent_check.setToolTip(
            "勾选后越早的历史帧越透明；取消后所有历史帧完全不透明"
        )

        form = QVBoxLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        form.addWidget(path_row)
        form.addWidget(history_row)
        form.addWidget(self.history_transparent_check)

        self.status_label = QLabel("", self)
        self.status_label.setObjectName("SlamPanelStatus")
        self.status_label.setWordWrap(True)
        self.status_label.hide()

        self.apply_button = QPushButton("开始建图", self)
        self.apply_button.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 7px 14px; } "
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self.apply_button.clicked.connect(self.applyRequested.emit)

        self.body_widget = QWidget(self)
        self.body_widget.setObjectName("SlamPanelBody")
        self.body_widget.setStyleSheet("QWidget#SlamPanelBody { background: transparent; }")
        body_layout = QVBoxLayout(self.body_widget)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)
        body_layout.addLayout(form)
        body_layout.addWidget(self.status_label)
        body_layout.addWidget(self.apply_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        layout.addLayout(header_layout)
        layout.addWidget(self.body_widget)

        self._collapsed = False

    def _toggle_collapsed(self):
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed):
        self._collapsed = bool(collapsed)
        self.body_widget.setVisible(not self._collapsed)
        self.collapse_button.setText("显示" if self._collapsed else "隐藏")
        self.collapse_button.setToolTip(
            "显示 SLAM 设置内容" if self._collapsed else "隐藏 SLAM 设置内容"
        )
        self.updateGeometry()
        self.panelGeometryChanged.emit()

    def _browse(self):
        current = self.path_edit.text().strip()
        start = current if current else self.start_directory
        path, _ = QFileDialog.getOpenFileName(
            self, "加载 SLAM 位姿", start, "TXT Files (*.txt);;All Files (*)"
        )
        if path:
            self.path_edit.setText(os.path.normpath(path))
            self.set_message("")

    def set_settings(self, pose_path, history_frames, history_transparent, start_directory):
        self.start_directory = start_directory or ""
        self.path_edit.setText(pose_path or "")
        self.history_spin.setValue(max(0, int(history_frames)))
        self.history_transparent_check.setChecked(bool(history_transparent))
        self.set_message("")

    def pose_path(self):
        return self.path_edit.text().strip()

    def history_frames(self):
        return self.history_spin.value()

    def history_transparent(self):
        return self.history_transparent_check.isChecked()

    def set_running(self, running):
        self.apply_button.setText("更新建图" if running else "开始建图")

    def set_message(self, message, error=True):
        message = str(message or "")
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            "color: #c62828; background: transparent;" if error
            else "color: #2e7d32; background: transparent;"
        )
        self.status_label.setVisible(bool(message))
        self.updateGeometry()
        self.panelGeometryChanged.emit()
