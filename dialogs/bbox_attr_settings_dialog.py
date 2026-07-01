# -*- coding: utf-8 -*-
"""目标框自定义属性设置对话框。"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


ATTR_TYPE_LABELS = {
    "select": "下拉",
    "check": "勾选",
    "text": "填入",
}
ATTR_LABEL_TYPES = {v: k for k, v in ATTR_TYPE_LABELS.items()}
RESERVED_ATTR_NAMES = {
    "x", "y", "z", "l", "w", "h", "yaw", "roll", "pitch",
    "class_name", "id", "link_id", "link_ID", "confidence", "movement_state",
    "type", "ID", "tag.link_id", "tag.link_ID", "tag.confidence", "tag.movement_state",
}


class BboxAttrSettingsDialog(QDialog):
    def __init__(self, attr_defs=None, history_browse_enabled=False, auto_create_empty_json=False, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("自定义标注属性")
        self.resize(760, 440)

        layout = QVBoxLayout(self)

        history_group = QGroupBox("历史帧", self)
        history_layout = QVBoxLayout(history_group)
        self.history_overlay_radio = QRadioButton(
            "历史帧叠加模式（按住 Shift 显示所有历史帧）",
            history_group,
        )
        self.history_browse_radio = QRadioButton(
            "历史帧播放模式（按住 Shift 后左键下一帧、右键上一帧）",
            history_group,
        )
        self.history_overlay_radio.setChecked(not bool(history_browse_enabled))
        self.history_browse_radio.setChecked(bool(history_browse_enabled))
        history_layout.addWidget(self.history_overlay_radio)
        history_layout.addWidget(self.history_browse_radio)
        layout.addWidget(history_group)

        save_group = QGroupBox("保存", self)
        save_layout = QVBoxLayout(save_group)
        self.auto_create_empty_json_check = QCheckBox("无标注框的帧也自动生成空 JSON 文件（[]）", save_group)
        self.auto_create_empty_json_check.setChecked(bool(auto_create_empty_json))
        save_layout.addWidget(self.auto_create_empty_json_check)
        layout.addWidget(save_group)

        attr_group = QGroupBox("目标框属性", self)
        attr_layout = QVBoxLayout(attr_group)
        tip = QLabel("每行一个属性。key 与 JSON 字段对齐；tag 下字段使用 tag.xxx；下拉/勾选类型的选项用逗号分隔；默认值为空表示不预填。")
        tip.setWordWrap(True)
        attr_layout.addWidget(tip)

        self.table = QTableWidget(0, 6, self)
        self.table.setHorizontalHeaderLabels(["属性名", "key", "方式", "可不写入", "默认值", "选项"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked)
        attr_layout.addWidget(self.table)

        tools = QHBoxLayout()
        self.add_btn = QPushButton("添加属性")
        self.remove_btn = QPushButton("删除选中")
        self.add_btn.clicked.connect(self._add_empty_row)
        self.remove_btn.clicked.connect(self._remove_selected_rows)
        tools.addWidget(self.add_btn)
        tools.addWidget(self.remove_btn)
        tools.addStretch()
        attr_layout.addLayout(tools)
        layout.addWidget(attr_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for attr_def in attr_defs or []:
            self._add_row(attr_def)

    def _add_empty_row(self):
        self._add_row({"key": "", "name": "", "type": "text", "options": []})

    def _add_row(self, attr_def):
        row = self.table.rowCount()
        self.table.insertRow(row)
        key = str(attr_def.get("key") or attr_def.get("name") or "").strip()
        field_key = str(attr_def.get("field_key") or key).strip()
        if not bool(attr_def.get("system", False)) and key.startswith("tag.") and "." in key:
            field_key = key.split(".", 1)[1].strip()
        label = str(attr_def.get("label") or attr_def.get("name") or attr_def.get("key") or "")
        name_item = QTableWidgetItem(label)
        name_item.setData(Qt.UserRole + 1, bool(attr_def.get("system", False)))
        name_item.setData(Qt.UserRole + 2, bool(attr_def.get("multi", True)))
        name_item.setData(Qt.UserRole + 3, field_key)
        self.table.setItem(row, 0, name_item)

        key_item = QTableWidgetItem(key)
        if bool(attr_def.get("system", False)):
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, 1, key_item)

        type_combo = QComboBox(self.table)
        type_combo.addItems([ATTR_TYPE_LABELS["text"], ATTR_TYPE_LABELS["select"], ATTR_TYPE_LABELS["check"]])
        type_key = attr_def.get("type") or "text"
        type_combo.setCurrentText(ATTR_TYPE_LABELS.get(type_key, ATTR_TYPE_LABELS["text"]))
        self.table.setCellWidget(row, 2, type_combo)

        allow_empty = QCheckBox(self.table)
        allow_empty.setChecked(bool(attr_def.get("allow_empty", True)))
        self.table.setCellWidget(row, 3, allow_empty)

        default_value = attr_def.get("default", "")
        if isinstance(default_value, (list, tuple)):
            default_value = ",".join(str(v) for v in default_value)
        self.table.setItem(row, 4, QTableWidgetItem("" if default_value is None else str(default_value)))

        options = attr_def.get("options") or []
        self.table.setItem(row, 5, QTableWidgetItem(",".join(str(v) for v in options)))

    def _remove_selected_rows(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            item = self.table.item(row, 0)
            if item is not None and bool(item.data(Qt.UserRole + 1)):
                continue
            self.table.removeRow(row)

    def _collect_defs(self):
        attr_defs = []
        seen = set()
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            key_item = self.table.item(row, 1)
            default_item = self.table.item(row, 4)
            options_item = self.table.item(row, 5)
            label = (name_item.text() if name_item else "").strip()
            key = (key_item.text() if key_item else "").strip()
            system = bool(name_item.data(Qt.UserRole + 1)) if name_item else False
            multi = bool(name_item.data(Qt.UserRole + 2)) if name_item else True
            field_key = str(name_item.data(Qt.UserRole + 3) or key).strip()
            if not system and key.startswith("tag.") and "." in key:
                field_key = key.split(".", 1)[1].strip()
            if not label:
                continue
            if not key:
                raise ValueError("{} 的 key 不能为空".format(label))
            if not field_key:
                field_key = key
            if not system and key in RESERVED_ATTR_NAMES:
                raise ValueError("{} 是系统字段，不能作为自定义属性名".format(key))
            if key in seen:
                raise ValueError("key 重复：{}".format(key))
            seen.add(key)

            type_combo = self.table.cellWidget(row, 2)
            type_key = ATTR_LABEL_TYPES.get(type_combo.currentText(), "text") if type_combo else "text"
            allow_empty_widget = self.table.cellWidget(row, 3)
            allow_empty = bool(allow_empty_widget.isChecked()) if allow_empty_widget else True
            default_text = (default_item.text() if default_item else "").strip()
            options_text = (options_item.text() if options_item else "").strip()
            options = [v.strip() for v in options_text.replace(";", ",").split(",") if v.strip()]
            if type_key in ("select", "check") and not options:
                raise ValueError("{} 需要至少一个选项".format(key))
            item = {
                "key": key,
                "name": label,
                "label": label,
                "type": type_key,
                "options": options,
                "allow_empty": allow_empty,
            }
            if field_key != key:
                item["field_key"] = field_key
            if default_text:
                if type_key == "check" and multi:
                    item["default"] = [v.strip() for v in default_text.replace(";", ",").split(",") if v.strip()]
                else:
                    item["default"] = default_text
            if system:
                item["system"] = True
            if type_key == "check" and not multi:
                item["multi"] = False
            attr_defs.append(item)
        return attr_defs

    def _accept_if_valid(self):
        try:
            self._attr_defs = self._collect_defs()
        except ValueError as exc:
            QMessageBox.warning(self, "自定义标注属性", str(exc))
            return
        self.accept()

    def attr_defs(self):
        return list(getattr(self, "_attr_defs", []))

    def history_browse_enabled(self):
        return bool(self.history_browse_radio.isChecked())

    def auto_create_empty_json_enabled(self):
        return bool(self.auto_create_empty_json_check.isChecked())
