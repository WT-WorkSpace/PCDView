from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QLineEdit,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


def _color_to_hex(color):
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        return color.upper()
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return "#{:02X}{:02X}{:02X}".format(int(color[0]), int(color[1]), int(color[2]))
    return "#FFFFFF"


class ImeFriendlyLabelTable(QTableWidget):
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            item = self.currentItem()
            if item is not None and item.flags() & Qt.ItemIsEditable and self.currentColumn() in (0, 1):
                item.setText("")
                self.editItem(item)
                event.accept()
                return
        super().keyPressEvent(event)


class ImeFriendlyLineEditDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setAttribute(Qt.WA_InputMethodEnabled, True)
        editor.setInputMethodHints(Qt.ImhNone)
        return editor


class SegmentationLabelSettingsDialog(QDialog):
    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("分割标签设置")
        self.resize(560, 420)
        self._labels = [dict(item) for item in labels]

        layout = QVBoxLayout(self)
        hint = QLabel("Name 可使用中文；Key 只能填写整数；Key=0 为默认背景标签，保留白色。", self)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.table = ImeFriendlyLabelTable(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Name", "Key", "Color"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked |
            QAbstractItemView.SelectedClicked |
            QAbstractItemView.EditKeyPressed
        )
        self.table.setAttribute(Qt.WA_InputMethodEnabled, True)
        self.table.viewport().setAttribute(Qt.WA_InputMethodEnabled, True)
        self._name_delegate = ImeFriendlyLineEditDelegate(self.table)
        self.table.setItemDelegateForColumn(0, self._name_delegate)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        row_buttons = QHBoxLayout()
        add_btn = QPushButton("新增", self)
        remove_btn = QPushButton("删除", self)
        color_btn = QPushButton("选择颜色", self)
        add_btn.clicked.connect(self._add_row)
        remove_btn.clicked.connect(self._remove_row)
        color_btn.clicked.connect(self._choose_color)
        row_buttons.addWidget(add_btn)
        row_buttons.addWidget(remove_btn)
        row_buttons.addWidget(color_btn)
        row_buttons.addStretch(1)
        layout.addLayout(row_buttons)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._fill_table()

    def _fill_table(self):
        self.table.setRowCount(0)
        for item in self._labels:
            self._append_row(item.get("name", ""), item.get("key", 0), _color_to_hex(item.get("color", "#FFFFFF")))

    def _append_row(self, name, key, color):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(name)))
        key_item = QTableWidgetItem(str(int(key)))
        if int(key) == 0:
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, 1, key_item)
        color_item = QTableWidgetItem(_color_to_hex(color))
        color_item.setBackground(QColor(_color_to_hex(color)))
        if int(key) == 0:
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, 2, color_item)

    def _add_row(self):
        used = {label["key"] for label in self.labels()}
        key = 1
        while key in used:
            key += 1
        self._append_row("label_{}".format(key), key, "#E74C3C")
        self.table.selectRow(self.table.rowCount() - 1)

    def _remove_row(self):
        row = self.table.currentRow()
        if row < 0:
            return
        key_item = self.table.item(row, 1)
        if key_item is not None and key_item.text().strip() == "0":
            QMessageBox.information(self, "分割标签设置", "默认背景标签不能删除。")
            return
        self.table.removeRow(row)

    def _choose_color(self):
        row = self.table.currentRow()
        if row < 0:
            return
        key_item = self.table.item(row, 1)
        if key_item is not None and key_item.text().strip() == "0":
            QMessageBox.information(self, "分割标签设置", "默认背景标签固定为白色。")
            return
        cur = self.table.item(row, 2).text() if self.table.item(row, 2) is not None else "#E74C3C"
        color = QColorDialog.getColor(QColor(cur), self, "选择标签颜色")
        if not color.isValid():
            return
        item = self.table.item(row, 2) or QTableWidgetItem()
        item.setText(color.name().upper())
        item.setBackground(color)
        self.table.setItem(row, 2, item)

    def labels(self):
        labels = []
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text().strip() if self.table.item(row, 0) else ""
            key_text = self.table.item(row, 1).text().strip() if self.table.item(row, 1) else ""
            color = self.table.item(row, 2).text().strip() if self.table.item(row, 2) else "#FFFFFF"
            try:
                key = int(key_text)
            except (TypeError, ValueError):
                key = None
            labels.append({"name": name, "key": key, "color": _color_to_hex(color)})
        return labels

    def selected_labels(self):
        return [dict(item) for item in self._labels]

    def _accept(self):
        labels = self.labels()
        if any(not item.get("name") for item in labels):
            QMessageBox.warning(self, "分割标签设置", "Name 不能为空，可使用中文。")
            return
        keys = [item.get("key") for item in labels]
        if any(key is None for key in keys):
            QMessageBox.warning(self, "分割标签设置", "Key 不能为空，且只能填写整数。")
            return
        if len(set(keys)) != len(keys):
            QMessageBox.warning(self, "分割标签设置", "Key 不能重复。")
            return
        by_key = {item["key"]: item for item in labels}
        if 0 not in by_key:
            QMessageBox.warning(self, "分割标签设置", "必须保留 Key=0 的背景标签。")
            return
        by_key[0]["name"] = by_key[0].get("name") or "background"
        by_key[0]["color"] = "#FFFFFF"
        self._labels = [by_key[key] for key in sorted(by_key)]
        self.accept()


class SegmentationLabelPickDialog(QDialog):
    def __init__(self, labels, point_count, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("选择分割标签")
        self.resize(360, 320)
        self.selected_key = None

        layout = QVBoxLayout(self)
        title = QLabel("圈选到 {} 个点，请选择要写入的标签：".format(int(point_count)), self)
        title.setWordWrap(True)
        layout.addWidget(title)

        for label in labels:
            btn = QPushButton("{}  ({})".format(label.get("name", ""), label.get("key", "")), self)
            btn.setStyleSheet(
                "QPushButton {{ text-align: left; padding: 8px 10px; border: 1px solid #D1D5DB; "
                "border-left: 18px solid {}; border-radius: 4px; background: white; }} "
                "QPushButton:hover {{ background: #F3F4F6; }}".format(_color_to_hex(label.get("color")))
            )
            btn.clicked.connect(lambda checked=False, key=int(label.get("key", 0)): self._choose(key))
            layout.addWidget(btn)
        layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose(self, key):
        self.selected_key = int(key)
        self.accept()
