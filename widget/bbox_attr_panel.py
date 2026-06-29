# -*- coding: utf-8 -*-
"""目标框属性编辑区域。"""
import os

from PyQt5.QtWidgets import (
    QWidget, QComboBox, QLineEdit, QLabel, QVBoxLayout, QGridLayout,
    QCheckBox, QButtonGroup, QHBoxLayout, QGraphicsDropShadowEffect, QPushButton,
    QSizePolicy,
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, pyqtSignal


class BboxAttributePanel(QWidget):
    attrSettingsRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bbox_index = None
        self._bbox_info = {}
        self._on_bbox_edited = None
        self._attr_defs = []
        self._custom_controls = {}
        self._custom_rows = []
        self._loading = False

        self.setObjectName("BboxAttributePanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        arrow_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "icons", "combo_down.svg")
        ).replace("\\", "/")
        self.setStyleSheet(
            "#BboxAttributePanel {"
            "background-color: #f8fafc;"
            "border: 1px solid #c7d0da;"
            "border-radius: 8px;"
            "}"
            "#BboxAttributePanel QWidget#BboxAttributeForm {"
            "background-color: #f8fafc;"
            "}"
            "#BboxAttributePanel QWidget#BboxAttributeOptions {"
            "background-color: #f8fafc;"
            "}"
            "#BboxAttributePanel QLabel { color: #111827; font-size: 14px; }"
            "#BboxAttributePanel QLabel#BboxAttributeTitle {"
            "font-size: 16px; font-weight: 700; padding-bottom: 6px;"
            "}"
            "#BboxAttributePanel QPushButton#BboxAttrSettingsButton {"
            "background: #e5e7eb; color: #111827; border: 1px solid #cbd5e1;"
            "border-radius: 5px; padding: 4px 10px; font-size: 13px;"
            "}"
            "#BboxAttributePanel QPushButton#BboxAttrSettingsButton:hover {"
            "background: #dbeafe; border-color: #93c5fd;"
            "}"
            "#BboxAttributePanel QLabel#BboxAttributeField {"
            "background-color: #f8fafc; font-weight: 700; min-width: 68px;"
            "}"
            "#BboxAttributePanel QLineEdit, "
            "#BboxAttributePanel QComboBox {"
            "background: #ffffff; color: #111827; border: 1px solid #8b98a5;"
            "border-radius: 5px; min-height: 34px; padding: 4px 8px;"
            "font-size: 14px;"
            "selection-background-color: #2563eb;"
            "}"
            "#BboxAttributePanel QComboBox::drop-down {"
            "width: 24px; border-left: 1px solid #cbd5e1;"
            "}"
            "#BboxAttributePanel QComboBox::down-arrow {"
            "image: url(%s); width: 10px; height: 6px;"
            "}"
            "#BboxAttributePanel QCheckBox {"
            "color: #111827; font-size: 14px; spacing: 6px;"
            "min-height: 26px;"
            "}"
            "#BboxAttributePanel QCheckBox::indicator {"
            "width: 15px; height: 15px;"
            "}"
            "#BboxAttributePanel QLineEdit:focus, "
            "#BboxAttributePanel QComboBox:focus {"
            "border: 2px solid #2563eb;"
            "}"
            % arrow_path
        )
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 3)
        shadow.setColor(QColor(15, 23, 42, 90))
        self.setGraphicsEffect(shadow)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(18, 14, 18, 16)
        outer_layout.setSpacing(10)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel("未选择目标框")
        self.title_label.setObjectName("BboxAttributeTitle")
        self.settings_btn = QPushButton("标注设置")
        self.settings_btn.setObjectName("BboxAttrSettingsButton")
        self.settings_btn.clicked.connect(self.attrSettingsRequested.emit)
        header_layout.addWidget(self.title_label, 1)
        header_layout.addWidget(self.settings_btn, 0, Qt.AlignRight)
        outer_layout.addLayout(header_layout)

        form = QWidget(self)
        form.setObjectName("BboxAttributeForm")
        form.setAttribute(Qt.WA_StyledBackground, True)
        form.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        outer_layout.addWidget(form)

        self.form_layout = QGridLayout(form)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self.form_layout.setHorizontalSpacing(12)
        self.form_layout.setVerticalSpacing(9)

        self.form_layout.setColumnStretch(1, 1)
        self.form_layout.setColumnMinimumWidth(1, 230)

        self.clear_bbox()

    @staticmethod
    def _field_label(text):
        label = QLabel(text)
        label.setObjectName("BboxAttributeField")
        return label

    @staticmethod
    def _option_checks(values):
        widget = QWidget()
        widget.setObjectName("BboxAttributeOptions")
        widget.setAttribute(Qt.WA_StyledBackground, True)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        group = QButtonGroup(widget)
        group.setExclusive(True)
        for value in values:
            text = "不写入" if value is None else str(value)
            btn_id = -1 if value is None else int(value)
            check = QCheckBox(text)
            group.addButton(check, btn_id)
            layout.addWidget(check)
        layout.addStretch()
        return group, widget

    def update_bbox(self, bbox_info, bbox_index=None, on_bbox_edited=None, class_names=None, attr_defs=None):
        self._bbox_index = bbox_index
        self._bbox_info = dict(bbox_info or {})
        self._on_bbox_edited = on_bbox_edited
        self.set_attr_defs(attr_defs)
        title_index = bbox_index + 1 if bbox_index is not None else "-"
        self.title_label.setText("Cuboid {}  目标框属性".format(title_index))
        self._set_controls_enabled(True)
        self._sync_controls(class_names)

    def set_attr_defs(self, attr_defs):
        self._attr_defs = [dict(item) for item in (attr_defs or [])]
        self._rebuild_custom_attrs()

    def preferred_height(self):
        header_h = max(self.title_label.sizeHint().height(), self.settings_btn.sizeHint().height())
        rows_h = 0
        for label, widget in self._custom_rows:
            rows_h += max(label.sizeHint().height(), widget.sizeHint().height(), 34)
        spacing_h = max(0, len(self._custom_rows) - 1) * self.form_layout.verticalSpacing()
        margins = self.layout().contentsMargins()
        return margins.top() + header_h + self.layout().spacing() + rows_h + spacing_h + margins.bottom()

    def clear_bbox(self):
        self._loading = True
        try:
            self._bbox_index = None
            self._bbox_info = {}
            self._on_bbox_edited = None
            self.title_label.setText("未选择目标框")
            self._sync_custom_controls()
            self._set_controls_enabled(False)
        finally:
            self._loading = False

    def _set_controls_enabled(self, enabled):
        for control in self._custom_controls.values():
            widget = control.get("widget")
            if widget is not None:
                widget.setEnabled(enabled)

    def _rebuild_custom_attrs(self):
        for label, widget in self._custom_rows:
            self.form_layout.removeWidget(label)
            self.form_layout.removeWidget(widget)
            label.setParent(None)
            widget.setParent(None)
            label.deleteLater()
            widget.deleteLater()
        self._custom_rows = []
        self._custom_controls = {}

        row = 0
        for attr_def in self._attr_defs:
            key = str(attr_def.get("key") or attr_def.get("name") or "").strip()
            label_text = str(attr_def.get("label") or attr_def.get("name") or key).strip()
            if not key or not label_text:
                continue
            attr_type = attr_def.get("type") or "text"
            options = [str(v) for v in (attr_def.get("options") or []) if str(v).strip()]
            label = self._field_label(label_text)
            widget, control = self._make_custom_control(key, attr_type, options, attr_def)
            self.form_layout.addWidget(label, row, 0)
            self.form_layout.addWidget(widget, row, 1)
            self._custom_rows.append((label, widget))
            self._custom_controls[key] = control
            row += 1
        self.form_layout.invalidate()
        self.updateGeometry()
        self.adjustSize()

    def _make_custom_control(self, key, attr_type, options, attr_def=None):
        attr_def = attr_def or {}
        if attr_type == "select":
            combo = QComboBox()
            if key != "class_name":
                if bool(attr_def.get("allow_empty", True)):
                    combo.addItem("不写入", None)
            combo.setEditable(False)
            combo.setInsertPolicy(QComboBox.NoInsert)
            for value in options:
                combo.addItem(value, value)
            combo.currentTextChanged.connect(self._on_attr_changed)
            return combo, {"type": attr_type, "widget": combo, "default": attr_def.get("default")}
        if attr_type == "check":
            widget = QWidget()
            widget.setObjectName("BboxAttributeOptions")
            widget.setAttribute(Qt.WA_StyledBackground, True)
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(12)
            multi = bool(attr_def.get("multi", True))
            checks = []
            group = None
            if not multi:
                group = QButtonGroup(widget)
                group.setExclusive(True)
                if bool(attr_def.get("allow_empty", True)):
                    empty_check = QCheckBox("不写入")
                    group.addButton(empty_check, -1)
                    layout.addWidget(empty_check)
                    checks.append(empty_check)
            for value in options:
                check = QCheckBox(value)
                check.stateChanged.connect(self._on_attr_changed)
                if group is not None:
                    group.addButton(check)
                layout.addWidget(check)
                checks.append(check)
            layout.addStretch()
            if group is not None:
                group.buttonClicked.connect(self._on_attr_changed)
            return widget, {"type": attr_type, "widget": widget, "checks": checks, "multi": multi, "group": group}
        edit = QLineEdit()
        if key == "link_id":
            placeholder = "多个 ID 用逗号分隔"
        elif bool(attr_def.get("allow_empty", True)):
            placeholder = "空表示不写入"
        else:
            placeholder = ""
        edit.setPlaceholderText(placeholder)
        edit.editingFinished.connect(self._on_attr_changed)
        return edit, {"type": "text", "widget": edit}

    def _sync_controls(self, class_names=None):
        self._loading = True
        try:
            self._sync_custom_controls()
        finally:
            self._loading = False

    def _sync_custom_controls(self):
        for key, control in self._custom_controls.items():
            value = self._bbox_info.get(key)
            if key == "class_name" and value is None:
                value = "others"
            attr_type = control.get("type")
            if attr_type == "select":
                combo = control["widget"]
                idx = combo.findData(value)
                if idx < 0:
                    idx = combo.findData(control.get("default"))
                if idx < 0:
                    idx = 0 if combo.count() else -1
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            elif attr_type == "check":
                if not control.get("multi", True):
                    target = "不写入" if value is None else str(value)
                    matched = False
                    for check in control.get("checks", []):
                        is_match = check.text() == target
                        check.setChecked(is_match)
                        matched = matched or is_match
                    if not matched and control.get("checks"):
                        control["checks"][0].setChecked(True)
                else:
                    selected = value if isinstance(value, (list, tuple, set)) else ([] if value is None else [value])
                    selected = {str(v) for v in selected}
                    for check in control.get("checks", []):
                        check.setChecked(check.text() in selected)
            else:
                text = self._format_link_id(value) if key == "link_id" else self._format_optional_value(value)
                control["widget"].setText(text)

    @staticmethod
    def _format_optional_value(value):
        return "" if value is None else str(value)

    @staticmethod
    def _format_link_id(value):
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value if v is not None)
        return str(value)

    @staticmethod
    def _optional_int(value):
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _set_group_value(group, value):
        btn_id = -1 if value is None else int(value)
        button = group.button(btn_id)
        if button is None:
            button = group.button(-1)
        if button is not None:
            button.setChecked(True)

    @staticmethod
    def _group_value(group):
        checked_id = group.checkedId()
        return None if checked_id < 0 else checked_id

    @staticmethod
    def _parse_optional_text(text):
        text = text.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return text

    @classmethod
    def _parse_link_id(cls, text):
        text = text.strip()
        if not text:
            return None
        parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
        vals = [cls._parse_optional_text(p) for p in parts]
        return vals[0] if len(vals) == 1 else vals

    def _on_attr_changed(self, *args):
        if self._loading or self._bbox_index is None:
            return
        for key, control in self._custom_controls.items():
            attr_type = control.get("type")
            if attr_type == "select":
                value = control["widget"].currentData()
                if key == "class_name":
                    value = value or "others"
                self._bbox_info[key] = value
            elif attr_type == "check":
                checked = [check.text() for check in control.get("checks", []) if check.isChecked()]
                if not control.get("multi", True):
                    value = checked[0] if checked and checked[0] != "不写入" else None
                    if key in ("confidence", "movement_state"):
                        value = self._optional_int(value)
                    self._bbox_info[key] = value
                else:
                    self._bbox_info[key] = checked or None
            else:
                text = control["widget"].text().strip()
                if key == "id":
                    value = self._parse_optional_text(text)
                elif key == "link_id":
                    value = self._parse_link_id(text)
                else:
                    value = text or None
                self._bbox_info[key] = value
        if self._on_bbox_edited and self._bbox_index is not None:
            edited_attrs = {
                key: self._bbox_info.get(key)
                for key in self._custom_controls.keys()
            }
            self._on_bbox_edited(self._bbox_index, edited_attrs)
