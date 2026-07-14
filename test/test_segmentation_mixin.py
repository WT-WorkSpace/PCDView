import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication, QAbstractItemView, QDialog, QLineEdit, QTableWidgetItem

import dialogs.segmentation_label_dialog as segmentation_label_dialog
import features.segmentation_mixin as segmentation_mixin
from dialogs.help_dialog import HELP_MANUAL_HTML
from dialogs.segmentation_label_dialog import SegmentationLabelSettingsDialog
from features.segmentation_mixin import SegmentationMixin


class DummySegmentation(SegmentationMixin):
    def __init__(self, temp_dir, point_count=4):
        self._temp_dir = Path(temp_dir)
        self.raw_points = np.zeros((point_count, 3), dtype=np.float32)
        self.pcd_file = str(self._temp_dir / "frame_0001.pcd")
        Path(self.pcd_file).write_text("# dummy pcd path for txt naming\n", encoding="utf-8")
        self.box_select_mode = False
        self.box_select_start = None
        self.box_select_start_logical = None
        self.points_rect_select_mode = False
        self.glwidget = DummyGlWidget()
        self._init_segmentation_state()

    def _segmentation_config_path(self):
        return self._temp_dir / "segmentation_labels.json"

    def _update_segmentation_legend(self):
        pass

    def _set_segmentation_legend_visible(self, visible):
        self.legend_visible = bool(visible)

    def _set_status_message(self, message):
        self.status_message = message

    def _update_points_rect_button_style(self, active):
        self.point_rect_style_active = bool(active)


class DummyAction:
    def __init__(self):
        self.checked = None

    def setChecked(self, checked):
        self.checked = bool(checked)


class DummyOverlay:
    def __init__(self):
        self.cleared = False

    def clear_rect(self):
        self.cleared = True


class DummyGlWidget:
    def __init__(self):
        self.cursor_unset = False
        self.items = []

    def unsetCursor(self):
        self.cursor_unset = True

    def update(self):
        pass

    def setCursor(self, cursor):
        self.cursor = cursor

    def addItem(self, item):
        self.items.append(item)

    def removeItem(self, item):
        if item in self.items:
            self.items.remove(item)

    def pixelSize(self, center):
        return 0.01


class DummyLabel:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


class DummySize:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    def width(self):
        return self._width

    def height(self):
        return self._height


class DummyLegendPanel:
    def __init__(self, size_width=168, size_height=86):
        self._size_width = size_width
        self._size_height = size_height
        self.geometry = None
        self.raised = False

    def sizeHint(self):
        return DummySize(self._size_width, self._size_height)

    def setGeometry(self, x, y, w, h):
        self.geometry = (x, y, w, h)

    def raise_(self):
        self.raised = True


class DummySizedGlWidget(DummyGlWidget):
    def __init__(self, width=1200, height=800):
        super().__init__()
        self._width = width
        self._height = height

    def width(self):
        return self._width

    def height(self):
        return self._height


class DummyLegendSegmentation(SegmentationMixin):
    def __init__(self):
        self._segmentation_labels = []
        self._segmentation_keys = None
        self.segmentation_legend_label = DummyLabel()
        self.glwidget = DummyGlWidget()


class FakeGlItem:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyEvent:
    def __init__(self, event_type, button=None, key=None):
        self._type = event_type
        self._button = button
        self._key = key

    def type(self):
        return self._type

    def button(self):
        return self._button

    def key(self):
        return self._key


class SegmentationMixinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_default_label_and_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp)
            self.assertEqual(dummy._segmentation_labels[0]["key"], 0)
            self.assertEqual(dummy._segmentation_labels[0]["color"], "#FFFFFF")

            dummy._segmentation_labels = [
                {"name": "background", "key": 0, "color": "#FFFFFF"},
                {"name": "vehicle", "key": 2, "color": "#123456"},
            ]
            dummy._save_segmentation_label_config()

            reloaded = DummySegmentation(tmp)
            self.assertEqual(reloaded._segmentation_labels[1]["name"], "vehicle")
            self.assertEqual(reloaded._segmentation_labels[1]["key"], 2)
            self.assertEqual(reloaded._segmentation_labels[1]["color"], "#123456")

    def test_label_config_roundtrip_preserves_chinese_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp)
            dummy._segmentation_labels = [
                {"name": "背景", "key": 0, "color": "#FFFFFF"},
                {"name": "车辆", "key": 1, "color": "#FF0000"},
            ]
            dummy._save_segmentation_label_config()

            reloaded = DummySegmentation(tmp)
            self.assertEqual(reloaded._segmentation_labels[0]["name"], "背景")
            self.assertEqual(reloaded._segmentation_labels[1]["name"], "车辆")

    def test_segmentation_legend_uses_table_layout(self):
        dummy = DummyLegendSegmentation()
        dummy._segmentation_labels = [
            {"name": "背景", "key": 0, "color": "#FFFFFF"},
            {"name": "车辆<&>", "key": 1, "color": "#FF0000"},
        ]
        dummy._segmentation_keys = np.array([0, 1, 1, 7], dtype=np.int32)

        dummy._update_segmentation_legend()
        html = dummy.segmentation_legend_label.text
        self.assertIn("<table", html)
        self.assertIn("Name", html)
        self.assertIn("Key", html)
        self.assertIn("Count", html)
        self.assertIn("背景", html)
        self.assertIn("车辆&lt;&amp;&gt;", html)
        self.assertIn("未知标签", html)

    def test_segmentation_legend_geometry_uses_compact_size(self):
        dummy = DummyLegendSegmentation()
        dummy.glwidget = DummySizedGlWidget()
        dummy.segmentation_legend_panel = DummyLegendPanel(size_width=168, size_height=86)

        dummy._update_segmentation_legend_geometry()

        self.assertEqual(dummy.segmentation_legend_panel.geometry[2], 168)
        self.assertEqual(dummy.segmentation_legend_panel.geometry[3], 86)

    def test_txt_save_load_one_key_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp, point_count=4)
            dummy._ensure_segmentation_keys()
            dummy._segmentation_keys[:] = [0, 3, 3, 1]

            self.assertTrue(dummy._save_segmentation_for_current_frame())
            txt_path = os.path.splitext(dummy.pcd_file)[0] + ".txt"
            with open(txt_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read().strip().splitlines(), ["0", "3", "3", "1"])

            dummy._segmentation_keys[:] = 0
            self.assertTrue(dummy._load_segmentation_for_current_frame())
            self.assertEqual(dummy._segmentation_keys.tolist(), [0, 3, 3, 1])
            self.assertTrue(dummy._segmentation_has_annotation)
            self.assertFalse(dummy._segmentation_display_enabled)

    def test_load_rejects_point_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp, point_count=4)
            txt_path = os.path.splitext(dummy.pcd_file)[0] + ".txt"
            Path(txt_path).write_text("1\n2\n3\n", encoding="utf-8")

            self.assertFalse(dummy._load_segmentation_for_current_frame())
            self.assertEqual(dummy._segmentation_keys.tolist(), [0, 0, 0, 0])
            self.assertFalse(dummy._segmentation_has_annotation)

    def test_imported_txt_saves_back_to_current_frame_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp, point_count=4)
            external_txt = Path(tmp) / "external_annotation.txt"
            external_txt.write_text("4\n4\n0\n0\n", encoding="utf-8")

            self.assertTrue(
                dummy._load_segmentation_for_current_frame(
                    path=str(external_txt),
                    as_frame_annotation=False,
                )
            )
            self.assertEqual(dummy._segmentation_keys.tolist(), [4, 4, 0, 0])
            self.assertTrue(dummy._segmentation_display_enabled)

            dummy._segmentation_keys[:] = [9, 9, 9, 9]
            self.assertTrue(dummy._save_segmentation_for_current_frame())

            frame_txt = Path(os.path.splitext(dummy.pcd_file)[0] + ".txt")
            self.assertEqual(frame_txt.read_text(encoding="utf-8").strip().splitlines(), ["9", "9", "9", "9"])
            self.assertEqual(external_txt.read_text(encoding="utf-8").strip().splitlines(), ["4", "4", "0", "0"])

    def test_segmentation_colors_apply_background_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp, point_count=4)
            dummy._segmentation_labels = [
                {"name": "background", "key": 0, "color": "#FFFFFF"},
                {"name": "vehicle", "key": 2, "color": "#FF0000"},
            ]
            dummy._segmentation_keys = np.array([0, 2, 0, 2], dtype=np.int32)
            base = np.zeros((4, 4), dtype=np.float32)
            base[:, 3] = 1.0

            unchanged = dummy._apply_segmentation_colors(base)
            self.assertTrue(np.array_equal(unchanged, base))

            dummy._segmentation_mode = True
            colored = dummy._apply_segmentation_colors(base)
            np.testing.assert_allclose(colored[0], [1.0, 1.0, 1.0, 1.0])
            np.testing.assert_allclose(colored[1], [1.0, 0.0, 0.0, 1.0])

    def test_reopened_annotation_does_not_auto_color_until_segmentation_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp, point_count=4)
            Path(os.path.splitext(dummy.pcd_file)[0] + ".txt").write_text("0\n2\n0\n2\n", encoding="utf-8")
            self.assertFalse(dummy._sync_segmentation_for_loaded_frame())
            base = np.zeros((4, 4), dtype=np.float32)
            base[:, 3] = 1.0

            unchanged = dummy._apply_segmentation_colors(base)
            self.assertTrue(np.array_equal(unchanged, base))

            dummy._segmentation_mode = True
            self.assertTrue(dummy._sync_segmentation_for_loaded_frame())
            colored = dummy._apply_segmentation_colors(base)
            self.assertFalse(np.array_equal(colored, base))

    def test_normal_frame_sync_does_not_read_large_segmentation_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp, point_count=4)
            Path(os.path.splitext(dummy.pcd_file)[0] + ".txt").write_text("0\n2\n0\n2\n", encoding="utf-8")
            original_loadtxt = segmentation_mixin.np.loadtxt
            calls = []

            def fail_if_called(*args, **kwargs):
                calls.append(args)
                raise AssertionError("np.loadtxt should not run during normal frame sync")

            try:
                segmentation_mixin.np.loadtxt = fail_if_called
                self.assertFalse(dummy._sync_segmentation_for_loaded_frame())
            finally:
                segmentation_mixin.np.loadtxt = original_loadtxt

            self.assertEqual(calls, [])
            self.assertIsNone(dummy._segmentation_keys)
            self.assertFalse(dummy._segmentation_has_annotation)
            self.assertFalse(dummy._segmentation_display_enabled)

    def test_unknown_loaded_keys_get_visible_colors(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp, point_count=3)
            dummy._segmentation_labels = [{"name": "background", "key": 0, "color": "#FFFFFF"}]
            dummy._segmentation_keys = np.array([0, 42, 42], dtype=np.int32)
            dummy._segmentation_has_annotation = True
            dummy._segmentation_display_enabled = True
            base = np.zeros((3, 4), dtype=np.float32)
            base[:, 3] = 1.0

            colored = dummy._apply_segmentation_colors(base)
            np.testing.assert_allclose(colored[0], [1.0, 1.0, 1.0, 1.0])
            self.assertFalse(np.array_equal(colored[1], base[1]))
            np.testing.assert_allclose(colored[1], colored[2])

    def test_later_polygon_assignment_overwrites_previous_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp, point_count=5)
            dummy._ensure_segmentation_keys()
            first_mask = np.array([True, True, False, False, False])
            second_mask = np.array([False, True, True, False, False])

            dummy._segmentation_keys[first_mask] = 1
            dummy._segmentation_keys[second_mask] = 2

            self.assertEqual(dummy._segmentation_keys.tolist(), [1, 2, 2, 0, 0])

    def test_entering_segmentation_mode_clears_selection_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp, point_count=3)
            dummy.box_select_mode = True
            dummy.box_select_start = (10, 20)
            dummy.box_select_start_logical = (5, 6)
            dummy.points_rect_select_mode = True
            dummy.box_select_action = DummyAction()
            dummy.points_rect_select_action = DummyAction()
            dummy._segmentation_action = DummyAction()
            dummy.box_select_overlay = DummyOverlay()

            dummy._toggle_segmentation_mode()

            self.assertTrue(dummy._segmentation_mode)
            self.assertFalse(dummy.box_select_mode)
            self.assertFalse(dummy.points_rect_select_mode)
            self.assertIsNone(dummy.box_select_start)
            self.assertIsNone(dummy.box_select_start_logical)
            self.assertFalse(dummy.box_select_action.checked)
            self.assertFalse(dummy.points_rect_select_action.checked)
            self.assertFalse(dummy.point_rect_style_active)
            self.assertTrue(dummy.box_select_overlay.cleared)

    def test_segmentation_mouse_events_only_consume_right_button(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp, point_count=3)
            dummy._segmentation_mode = True
            calls = []
            dummy._segmentation_add_vertex_from_screen = lambda mx, my: calls.append((mx, my)) or True

            self.assertFalse(
                dummy._handle_segmentation_mouse_event(
                    DummyEvent(QEvent.MouseButtonPress, button=Qt.LeftButton), 1, 2
                )
            )
            self.assertTrue(
                dummy._handle_segmentation_mouse_event(
                    DummyEvent(QEvent.MouseButtonPress, button=Qt.RightButton), 3, 4
                )
            )
            self.assertTrue(
                dummy._handle_segmentation_mouse_event(
                    DummyEvent(QEvent.MouseButtonRelease, button=Qt.RightButton), 3, 4
                )
            )
            self.assertEqual(calls, [(3, 4)])

    def test_segmentation_key_events_consume_cancel_and_undo_only_in_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp, point_count=3)
            escape = DummyEvent(QEvent.KeyPress, key=Qt.Key_Escape)
            backspace = DummyEvent(QEvent.KeyPress, key=Qt.Key_Backspace)
            other = DummyEvent(QEvent.KeyPress, key=Qt.Key_A)

            self.assertFalse(dummy._handle_segmentation_key_event(escape))
            dummy._segmentation_mode = True
            dummy._segmentation_polygon = [[0.0, 0.0], [1.0, 1.0]]

            self.assertTrue(dummy._handle_segmentation_key_event(backspace))
            self.assertEqual(dummy._segmentation_polygon, [[0.0, 0.0]])
            self.assertTrue(dummy._handle_segmentation_key_event(escape))
            self.assertEqual(dummy._segmentation_polygon, [])
            self.assertFalse(dummy._handle_segmentation_key_event(other))

    def test_escape_exits_segmentation_mode_when_no_polygon_is_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp, point_count=3)
            dummy._segmentation_mode = True
            dummy._segmentation_action = DummyAction()
            dummy._segmentation_polygon = []

            self.assertTrue(dummy._handle_segmentation_key_event(DummyEvent(QEvent.KeyPress, key=Qt.Key_Escape)))
            self.assertFalse(dummy._segmentation_mode)
            self.assertFalse(dummy._segmentation_action.checked)
            self.assertTrue(dummy.glwidget.cursor_unset)

    def test_segmentation_hover_first_vertex_draws_snap_ring(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp, point_count=3)
            dummy._segmentation_polygon = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
            dummy._segmentation_hover = [0.0, 0.0]
            original_scatter = segmentation_mixin.gl.GLScatterPlotItem
            original_line = segmentation_mixin.gl.GLLinePlotItem
            try:
                segmentation_mixin.gl.GLScatterPlotItem = lambda **kwargs: FakeGlItem(**kwargs)
                segmentation_mixin.gl.GLLinePlotItem = lambda **kwargs: FakeGlItem(**kwargs)
                dummy._refresh_segmentation_polygon_items()
            finally:
                segmentation_mixin.gl.GLScatterPlotItem = original_scatter
                segmentation_mixin.gl.GLLinePlotItem = original_line

            line_items = [item for item in dummy.glwidget.items if item.kwargs.get("mode") == "line_strip"]
            ring_items = [item for item in line_items if item.kwargs.get("width") == 3.0]
            self.assertEqual(len(ring_items), 1)
            self.assertEqual(len(ring_items[0].kwargs["pos"]), 49)

    def test_finish_polygon_assigns_label_and_auto_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = DummySegmentation(tmp, point_count=4)
            dummy.raw_points = np.array(
                [
                    [0.5, 0.5, 0.0],
                    [1.0, 1.0, 0.0],
                    [5.0, 5.0, 0.0],
                    [2.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
            dummy._segmentation_polygon = [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]]
            dummy._segmentation_keys = np.array([1, 1, 1, 1], dtype=np.int32)
            dummy.vis_frame_called = False
            dummy.vis_fram = lambda *args, **kwargs: setattr(dummy, "vis_frame_called", True)

            original_world_to_screen = segmentation_mixin.world_to_screen
            original_dialog = segmentation_mixin.SegmentationLabelPickDialog

            class AutoPickDialog:
                def __init__(self, labels, point_count, parent=None):
                    self.selected_key = 7
                    self.point_count = point_count

                def setWindowModality(self, modality):
                    pass

                def exec_(self):
                    return QDialog.Accepted

            def fake_world_to_screen(glwidget, pts):
                arr = np.asarray(pts, dtype=np.float64)
                return arr[:, :2]

            try:
                segmentation_mixin.world_to_screen = fake_world_to_screen
                segmentation_mixin.SegmentationLabelPickDialog = AutoPickDialog
                self.assertTrue(dummy._finish_segmentation_polygon())
            finally:
                segmentation_mixin.world_to_screen = original_world_to_screen
                segmentation_mixin.SegmentationLabelPickDialog = original_dialog

            self.assertEqual(dummy._segmentation_keys.tolist(), [7, 7, 1, 7])
            self.assertTrue(dummy.vis_frame_called)
            txt_path = os.path.splitext(dummy.pcd_file)[0] + ".txt"
            with open(txt_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read().strip().splitlines(), ["7", "7", "1", "7"])

    def test_label_settings_dialog_rejects_invalid_and_duplicate_keys(self):
        dialog = SegmentationLabelSettingsDialog(
            [
                {"name": "background", "key": 0, "color": "#FFFFFF"},
                {"name": "vehicle", "key": 1, "color": "#FF0000"},
            ],
            None,
        )
        original_warning = segmentation_label_dialog.QMessageBox.warning
        try:
            segmentation_label_dialog.QMessageBox.warning = lambda *args, **kwargs: None
            dialog.table.setItem(1, 1, QTableWidgetItem("abc"))
            dialog._accept()
            self.assertNotEqual(dialog.result(), QDialog.Accepted)

            dialog.table.setItem(1, 1, QTableWidgetItem("0"))
            dialog._accept()
            self.assertNotEqual(dialog.result(), QDialog.Accepted)
        finally:
            segmentation_label_dialog.QMessageBox.warning = original_warning

    def test_label_settings_dialog_rejects_empty_name_and_key(self):
        dialog = SegmentationLabelSettingsDialog(
            [
                {"name": "background", "key": 0, "color": "#FFFFFF"},
                {"name": "vehicle", "key": 1, "color": "#FF0000"},
            ],
            None,
        )
        original_warning = segmentation_label_dialog.QMessageBox.warning
        try:
            segmentation_label_dialog.QMessageBox.warning = lambda *args, **kwargs: None
            dialog.table.item(1, 0).setText("")
            dialog._accept()
            self.assertNotEqual(dialog.result(), QDialog.Accepted)

            dialog.table.item(1, 0).setText("vehicle")
            dialog.table.item(1, 1).setText("")
            dialog._accept()
            self.assertNotEqual(dialog.result(), QDialog.Accepted)
        finally:
            segmentation_label_dialog.QMessageBox.warning = original_warning

    def test_label_settings_dialog_accepts_chinese_name(self):
        dialog = SegmentationLabelSettingsDialog(
            [
                {"name": "背景", "key": 0, "color": "#FFFFFF"},
                {"name": "车辆", "key": 1, "color": "#FF0000"},
            ],
            None,
        )
        dialog._accept()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        labels = dialog.selected_labels()
        self.assertEqual(labels[0]["name"], "背景")
        self.assertEqual(labels[1]["name"], "车辆")

    def test_label_settings_dialog_edit_trigger_keeps_ime_input(self):
        dialog = SegmentationLabelSettingsDialog(
            [
                {"name": "background", "key": 0, "color": "#FFFFFF"},
                {"name": "vehicle", "key": 1, "color": "#FF0000"},
            ],
            None,
        )
        triggers = dialog.table.editTriggers()
        self.assertFalse(bool(triggers & QAbstractItemView.AnyKeyPressed))
        self.assertTrue(bool(triggers & QAbstractItemView.DoubleClicked))
        self.assertTrue(bool(triggers & QAbstractItemView.SelectedClicked))

    def test_label_settings_dialog_name_editor_enables_ime(self):
        dialog = SegmentationLabelSettingsDialog(
            [
                {"name": "background", "key": 0, "color": "#FFFFFF"},
                {"name": "vehicle", "key": 1, "color": "#FF0000"},
            ],
            None,
        )
        self.assertTrue(dialog.table.testAttribute(Qt.WA_InputMethodEnabled))
        self.assertTrue(dialog.table.viewport().testAttribute(Qt.WA_InputMethodEnabled))
        delegate = dialog.table.itemDelegateForColumn(0)
        editor = delegate.createEditor(dialog.table, None, dialog.table.model().index(1, 0))
        self.assertIsInstance(editor, QLineEdit)
        self.assertTrue(editor.testAttribute(Qt.WA_InputMethodEnabled))

    def test_label_settings_dialog_delete_key_clears_editable_name_and_key(self):
        dialog = SegmentationLabelSettingsDialog(
            [
                {"name": "background", "key": 0, "color": "#FFFFFF"},
                {"name": "vehicle", "key": 1, "color": "#FF0000"},
            ],
            None,
        )
        delete_event = QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)

        dialog.table.setCurrentCell(1, 0)
        dialog.table.keyPressEvent(delete_event)
        self.assertEqual(dialog.table.item(1, 0).text(), "")

        dialog.table.item(1, 0).setText("vehicle")
        dialog.table.setCurrentCell(1, 1)
        dialog.table.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Backspace, Qt.NoModifier))
        self.assertEqual(dialog.table.item(1, 1).text(), "")

        dialog.table.setCurrentCell(0, 1)
        dialog.table.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
        self.assertEqual(dialog.table.item(0, 1).text(), "0")

    def test_label_settings_dialog_preserves_background_label(self):
        dialog = SegmentationLabelSettingsDialog(
            [
                {"name": "background", "key": 0, "color": "#000000"},
                {"name": "vehicle", "key": 1, "color": "#FF0000"},
            ],
            None,
        )
        original_info = segmentation_label_dialog.QMessageBox.information
        try:
            segmentation_label_dialog.QMessageBox.information = lambda *args, **kwargs: None
            dialog.table.selectRow(0)
            dialog._remove_row()
            self.assertEqual(dialog.table.rowCount(), 2)
        finally:
            segmentation_label_dialog.QMessageBox.information = original_info

        dialog._accept()
        labels = dialog.selected_labels()
        self.assertEqual(labels[0]["key"], 0)
        self.assertEqual(labels[0]["color"], "#FFFFFF")

    def test_help_manual_documents_segmentation_workflow(self):
        required = [
            "点云分割标注",
            "右键点击",
            "分割标签设置",
            "每行一个数字",
            "加载分割TXT",
            "退出分割模式",
            "Backspace",
            "Esc",
        ]
        for text in required:
            self.assertIn(text, HELP_MANUAL_HTML)

    def test_main_window_smoke_exposes_segmentation_controls(self):
        from qtvis import PointCloudViewer

        viewer = PointCloudViewer()
        try:
            viewer.resize(640, 480)
            viewer.show()
            self._app.processEvents()
            viewer._toggle_segmentation_mode()
            self._app.processEvents()

            tool_actions = viewer.menu_bar.actions()[1].menu().actions()
            self.assertTrue(viewer._segmentation_mode)
            self.assertTrue(viewer.segmentation_legend_panel.isVisible())
            self.assertIn("右键", viewer._segmentation_action.toolTip())
            segmentation_button = viewer.toolbar.widgetForAction(viewer._segmentation_action)
            self.assertIn("#2196F3", segmentation_button.styleSheet())
            segmentation_menu_action = next(action for action in tool_actions if action.text() == "分割设置")
            segmentation_actions = segmentation_menu_action.menu().actions()
            self.assertFalse(any(action.text() == "分割标签设置" for action in tool_actions))
            self.assertFalse(any(action.text() == "退出分割模式" for action in tool_actions))
            self.assertFalse(any(action.text() == "加载分割TXT" for action in tool_actions))
            self.assertTrue(any(action.text() == "分割标签设置" for action in segmentation_actions))
            self.assertTrue(any(action.text() == "退出分割模式" for action in segmentation_actions))
            self.assertTrue(any(action.text() == "加载分割TXT" for action in segmentation_actions))

            viewer._exit_segmentation_mode()
            self._app.processEvents()
            self.assertFalse(viewer._segmentation_mode)
            self.assertFalse(viewer.segmentation_legend_panel.isVisible())
            self.assertNotIn("#2196F3", segmentation_button.styleSheet())
        finally:
            viewer.close()

    def test_directory_listing_excludes_segmentation_txt_sidecars(self):
        from qtvis import PointCloudViewer

        with tempfile.TemporaryDirectory() as tmp:
            for name in ["0001.pcd", "0001.txt", "0002.pcd", "0002.txt", "standalone.txt"]:
                Path(tmp, name).write_text("0\n", encoding="utf-8")

            self.assertEqual(
                PointCloudViewer._list_point_cloud_files(tmp),
                ["0001.pcd", "0002.pcd", "standalone.txt"],
            )


if __name__ == "__main__":
    unittest.main()
