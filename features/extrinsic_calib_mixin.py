# -*- coding: utf-8 -*-
"""在主窗口 3D 视图上进行多雷达外参标定（点云最后一列须为 lidar_id）。"""
import os
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from utils.extrinsic_calib import (
    PARAM_KEYS,
    apply_offsets_to_structured,
    default_lidar_color,
    export_offsets_json,
    get_offset_for_lidar,
    icp_multi_refine_align,
    lidar_id_to_str,
    mask_lidar,
    resolve_lidar1_id,
    same_lidar_id,
    save_corrected_structured_with_wata,
    structured_to_points_array,
    unique_lidar_ids,
    validate_lidar_id_last_column,
)
from utils.load_pcd import get_metadata_from_pcd_file, load_structured_points
from utils.move_pcd import move_pcd_with_xyzrpy

PARAM_LABELS = ("X 平移 (m)", "Y 平移 (m)", "Z 平移 (m)", "Roll (°)", "Pitch (°)", "Yaw (°)")


class ExtrinsicCalibMixin:
    """在主 glwidget 上预览/标定，不独占第二个 3D 窗口。"""

    def _extrinsic_init_state(self):
        self._extrinsic_calib_mode = False
        self._extrinsic_dock = None
        self._extrinsic_snapshot = None
        self._extrinsic_frame_raw = None
        self._session_extrinsic_offsets = None
        self._extrinsic_pcd_metadata = None
        self._extrinsic_lidar_ids = []
        self._extrinsic_offsets = {}
        self._extrinsic_ref_lidar_id = None
        self._extrinsic_lidar_colors = {}
        self._extrinsic_visibility = {}
        self._extrinsic_loading_params = False
        self._extrinsic_loading_vis = False
        self._extrinsic_spins = {}

    def _copy_extrinsic_offsets(self, offsets: dict) -> dict:
        return {lid: list(vals) for lid, vals in offsets.items()}

    def _clear_session_extrinsic_offsets(self):
        """打开新文件/新目录时清除跨帧外参。"""
        self._session_extrinsic_offsets = None
        self._extrinsic_frame_raw = None

    def _session_extrinsic_is_active(self) -> bool:
        return self._session_extrinsic_offsets is not None and len(
            self._session_extrinsic_offsets
        ) > 0

    def _store_extrinsic_raw_frame(self):
        if getattr(self, "structured_points", None) is not None:
            self._extrinsic_frame_raw = self.structured_points.copy()

    def _apply_session_extrinsic_to_frame(self):
        """将已「应用校正」的外参施加到当前帧原始点云（播放/切帧时保持拼接）。"""
        if not self._session_extrinsic_is_active():
            return
        if not hasattr(self, "metadata") or not self.metadata:
            return
        ok, _ = validate_lidar_id_last_column(self.metadata)
        if not ok:
            return
        base = self._extrinsic_frame_raw
        if base is None:
            base = self.structured_points
        if base is None:
            return
        try:
            self.structured_points = apply_offsets_to_structured(
                base,
                self.metadata,
                self._session_extrinsic_offsets,
                degrees=True,
            )
            self._rebuild_raw_points_from_structured()
        except Exception as e:
            print("apply session extrinsic failed:", e)

    def _sync_extrinsic_offsets_for_lidars(self, preserve_existing: bool = True):
        """按当前帧雷达列表对齐 offset 字典，保留已有/session 参数。"""
        if self._extrinsic_snapshot is None and self._extrinsic_frame_raw is not None:
            self._extrinsic_snapshot = self._extrinsic_frame_raw.copy()
        if self._extrinsic_snapshot is None:
            return
        pts = structured_to_points_array(self._extrinsic_snapshot, self.metadata)
        self._extrinsic_lidar_ids = unique_lidar_ids(pts)
        old = self._extrinsic_offsets if preserve_existing else {}
        session = self._session_extrinsic_offsets or {}
        new_offsets = {}
        for lid in self._extrinsic_lidar_ids:
            if preserve_existing and lid in old:
                new_offsets[lid] = list(old[lid])
            else:
                new_offsets[lid] = list(get_offset_for_lidar(lid, session))
        self._extrinsic_offsets = new_offsets
        if (
            self._extrinsic_ref_lidar_id is None
            or not any(
                same_lidar_id(self._extrinsic_ref_lidar_id, lid)
                for lid in self._extrinsic_lidar_ids
            )
        ):
            try:
                self._extrinsic_ref_lidar_id = resolve_lidar1_id(self._extrinsic_lidar_ids)
            except ValueError:
                if self._extrinsic_lidar_ids:
                    self._extrinsic_ref_lidar_id = self._extrinsic_lidar_ids[0]

    def _extrinsic_after_load_frame(self):
        """load_frame / open_file 之后调用：保存原始帧并套用会话外参。"""
        self._store_extrinsic_raw_frame()
        self._apply_session_extrinsic_to_frame()
        if getattr(self, "_extrinsic_calib_mode", False):
            self._extrinsic_on_frame_changed()

    def _sync_extrinsic_action_checked(self, checked: bool):
        for a in (
            getattr(self, "_extrinsic_calib_action", None),
            getattr(self, "_extrinsic_calib_menu_action", None),
        ):
            if a is not None:
                a.blockSignals(True)
                a.setChecked(checked)
                a.blockSignals(False)

    def _toggle_extrinsic_calib(self):
        action = self.sender()
        if action is None:
            action = getattr(self, "_extrinsic_calib_action", None)
        checked = action.isChecked() if action is not None else False
        self._sync_extrinsic_action_checked(checked)

        if not checked:
            if self._extrinsic_calib_mode:
                self._disable_extrinsic_calib()
            return
        if not hasattr(self, "structured_points") or self.structured_points is None:
            QMessageBox.warning(self, "外参标定", "请先在主窗口加载点云")
            self._sync_extrinsic_action_checked(False)
            return
        fields = getattr(self, "metadata", None)
        if not fields:
            QMessageBox.warning(self, "外参标定", "当前帧无字段信息")
            self._sync_extrinsic_action_checked(False)
            return
        ok, msg = validate_lidar_id_last_column(fields)
        if not ok:
            QMessageBox.warning(self, "外参标定", msg)
            self._sync_extrinsic_action_checked(False)
            return
        self._enable_extrinsic_calib()

    def _enable_extrinsic_calib(self):
        self._extrinsic_calib_mode = True
        self._sync_extrinsic_action_checked(True)
        self.box_select_mode = False
        if hasattr(self, "box_select_action"):
            self.box_select_action.setChecked(False)
        self.points_rect_select_mode = False
        if hasattr(self, "points_rect_select_action"):
            self.points_rect_select_action.setChecked(False)
        if hasattr(self, "box_select_overlay"):
            self.box_select_overlay.clear_rect()

        self._store_extrinsic_raw_frame()
        self._extrinsic_snapshot = (
            self._extrinsic_frame_raw.copy()
            if self._extrinsic_frame_raw is not None
            else self.structured_points.copy()
        )
        if getattr(self, "pcd_file", None) and os.path.isfile(self.pcd_file):
            try:
                self._extrinsic_pcd_metadata = get_metadata_from_pcd_file(self.pcd_file)
            except Exception:
                self._extrinsic_pcd_metadata = None
        else:
            self._extrinsic_pcd_metadata = None

        if self._session_extrinsic_is_active():
            self._extrinsic_offsets = self._copy_extrinsic_offsets(
                self._session_extrinsic_offsets
            )
        else:
            self._extrinsic_offsets = {}
        self._sync_extrinsic_offsets_for_lidars(preserve_existing=True)
        self._extrinsic_lidar_colors = {
            lid: default_lidar_color(i)
            for i, lid in enumerate(self._extrinsic_lidar_ids)
        }

        self._ensure_extrinsic_dock()
        self._extrinsic_refresh_ui_from_state()
        self._extrinsic_dock.show()
        self._update_extrinsic_button_style(True)
        self._set_status_message(
            "外参标定：在主视图调节各雷达位姿，完成后点击「应用校正」"
        )
        self.vis_fram(updata_color_bar=False)

    def _disable_extrinsic_calib(self):
        self._extrinsic_calib_mode = False
        self._sync_extrinsic_action_checked(False)
        if self._extrinsic_dock is not None:
            self._extrinsic_dock.hide()
        self._update_extrinsic_button_style(False)
        self._update_frame_info_label()
        self.vis_fram(updata_color_bar=False)

    def _extrinsic_on_frame_changed(self):
        """切帧时：保留会话外参，仅换新帧原始点云底图。"""
        if not getattr(self, "_extrinsic_calib_mode", False):
            return
        if not hasattr(self, "structured_points") or self.structured_points is None:
            self._disable_extrinsic_calib()
            return
        ok, msg = validate_lidar_id_last_column(self.metadata)
        if not ok:
            QMessageBox.warning(self, "外参标定", "新帧不符合要求，已退出标定模式。\n" + msg)
            self._disable_extrinsic_calib()
            return
        if self._extrinsic_frame_raw is not None:
            self._extrinsic_snapshot = self._extrinsic_frame_raw.copy()
        else:
            self._store_extrinsic_raw_frame()
            self._extrinsic_snapshot = self.structured_points.copy()
        if getattr(self, "pcd_file", None) and os.path.isfile(self.pcd_file):
            try:
                self._extrinsic_pcd_metadata = get_metadata_from_pcd_file(self.pcd_file)
            except Exception:
                pass
        self._sync_extrinsic_offsets_for_lidars(preserve_existing=True)
        if not self._extrinsic_lidar_colors:
            self._extrinsic_lidar_colors = {
                lid: default_lidar_color(i)
                for i, lid in enumerate(self._extrinsic_lidar_ids)
            }
        self._extrinsic_refresh_ui_from_state()
        self.vis_fram(updata_color_bar=False)

    def _ensure_extrinsic_dock(self):
        if self._extrinsic_dock is not None:
            return
        dock = QDockWidget("外参标定", self)
        dock.setObjectName("ExtrinsicCalibDock")
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock.setMinimumWidth(300)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setSpacing(8)

        hint = QLabel(
            "可在「配准目标雷达」中选择 ICP 对齐的参考雷达；各雷达外参均可手动微调"
            "（步进 0.1，无范围限制）。满意后点「应用校正」。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#546e7a; font-size:12px; padding:4px;")
        lay.addWidget(hint)

        lidar_group = QGroupBox("雷达")
        lidar_form = QFormLayout(lidar_group)
        self._extrinsic_lidar_combo = QComboBox()
        self._extrinsic_lidar_combo.currentIndexChanged.connect(self._extrinsic_on_lidar_selected)
        lidar_form.addRow("当前雷达", self._extrinsic_lidar_combo)
        self._extrinsic_ref_combo = QComboBox()
        self._extrinsic_ref_combo.currentIndexChanged.connect(self._extrinsic_on_ref_combo_changed)
        lidar_form.addRow("配准目标雷达", self._extrinsic_ref_combo)
        self._extrinsic_vis_widget = QWidget()
        self._extrinsic_vis_layout = QVBoxLayout(self._extrinsic_vis_widget)
        self._extrinsic_vis_layout.setContentsMargins(0, 0, 0, 0)
        lidar_form.addRow("显示", self._extrinsic_vis_widget)
        vis_row = QHBoxLayout()
        b1 = QPushButton("全显")
        b1.clicked.connect(lambda: self._extrinsic_set_all_vis(True))
        b2 = QPushButton("全隐")
        b2.clicked.connect(lambda: self._extrinsic_set_all_vis(False))
        vis_row.addWidget(b1)
        vis_row.addWidget(b2)
        lidar_form.addRow("", vis_row)
        lay.addWidget(lidar_group)

        param_group = QGroupBox("位姿 [dx,dy,dz, roll,pitch,yaw]")
        param_form = QFormLayout(param_group)
        self._extrinsic_spins = {}
        for key, label in zip(PARAM_KEYS, PARAM_LABELS):
            spin = QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setKeyboardTracking(True)
            spin.valueChanged.connect(self._extrinsic_on_param_changed)
            param_form.addRow(label, spin)
            self._extrinsic_spins[key] = spin
        lay.addWidget(param_group)

        self._extrinsic_btn_icp = QPushButton("ICP 多轮精配准")
        self._extrinsic_btn_icp.setStyleSheet(
            "QPushButton { background:#5c6bc0; color:white; font-weight:600; "
            "min-height:32px; border-radius:4px; }"
            "QPushButton:hover { background:#3f51b5; }"
        )
        self._extrinsic_btn_icp.clicked.connect(self._extrinsic_run_icp)
        lay.addWidget(self._extrinsic_btn_icp)

        color_row = QHBoxLayout()
        self._extrinsic_color_preview = QLabel()
        self._extrinsic_color_preview.setFixedSize(48, 22)
        btn_color = QPushButton("雷达颜色…")
        btn_color.clicked.connect(self._extrinsic_pick_color)
        color_row.addWidget(self._extrinsic_color_preview)
        color_row.addWidget(btn_color)
        lay.addLayout(color_row)

        row1 = QHBoxLayout()
        self._extrinsic_btn_reset_one = QPushButton("重置当前")
        self._extrinsic_btn_reset_one.clicked.connect(self._extrinsic_reset_current)
        self._extrinsic_btn_reset_all = QPushButton("重置全部")
        self._extrinsic_btn_reset_all.clicked.connect(self._extrinsic_reset_all)
        row1.addWidget(self._extrinsic_btn_reset_one)
        row1.addWidget(self._extrinsic_btn_reset_all)
        lay.addLayout(row1)

        self._extrinsic_btn_apply = QPushButton("应用校正到当前帧")
        self._extrinsic_btn_apply.setStyleSheet(
            "QPushButton { background:#00897b; color:white; font-weight:600; "
            "min-height:32px; border-radius:4px; }"
        )
        self._extrinsic_btn_apply.clicked.connect(self._extrinsic_apply_commit)
        lay.addWidget(self._extrinsic_btn_apply)

        self._extrinsic_btn_export = QPushButton("导出偏移 JSON")
        self._extrinsic_btn_export.clicked.connect(self._extrinsic_export_json)
        lay.addWidget(self._extrinsic_btn_export)

        self._extrinsic_btn_save_pcd = QPushButton("保存校正后 PCD…")
        self._extrinsic_btn_save_pcd.clicked.connect(self._extrinsic_save_pcd)
        lay.addWidget(self._extrinsic_btn_save_pcd)

        self._extrinsic_btn_batch = QPushButton("批量校正目录…")
        self._extrinsic_btn_batch.clicked.connect(self._extrinsic_batch_folder)
        lay.addWidget(self._extrinsic_btn_batch)

        self._extrinsic_status = QLabel("")
        self._extrinsic_status.setWordWrap(True)
        lay.addWidget(self._extrinsic_status)
        lay.addStretch()

        scroll.setWidget(panel)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._extrinsic_dock = dock

    def _extrinsic_refresh_ui_from_state(self):
        self._extrinsic_lidar_combo.blockSignals(True)
        self._extrinsic_lidar_combo.clear()
        for lid in self._extrinsic_lidar_ids:
            self._extrinsic_lidar_combo.addItem("雷达 {}".format(lidar_id_to_str(lid)), lid)
        self._extrinsic_lidar_combo.blockSignals(False)

        self._extrinsic_ref_combo.blockSignals(True)
        self._extrinsic_ref_combo.clear()
        for lid in self._extrinsic_lidar_ids:
            self._extrinsic_ref_combo.addItem("雷达 {}".format(lidar_id_to_str(lid)), lid)
        ref_idx = 0
        if self._extrinsic_ref_lidar_id is not None:
            for i in range(self._extrinsic_ref_combo.count()):
                if same_lidar_id(
                    self._extrinsic_ref_combo.itemData(i), self._extrinsic_ref_lidar_id
                ):
                    ref_idx = i
                    break
        if self._extrinsic_lidar_ids:
            self._extrinsic_ref_combo.setCurrentIndex(ref_idx)
            self._extrinsic_ref_lidar_id = self._extrinsic_ref_combo.currentData()
        self._extrinsic_ref_combo.blockSignals(False)

        if self._extrinsic_lidar_ids:
            self._extrinsic_lidar_combo.setCurrentIndex(0)
        self._extrinsic_rebuild_vis_checks()
        self._extrinsic_sync_spins()
        self._extrinsic_update_color_preview()
        n = len(structured_to_points_array(self._extrinsic_snapshot, self.metadata))
        self._extrinsic_status.setText(
            "当前帧 {} 点，{} 台雷达".format(n, len(self._extrinsic_lidar_ids))
        )
        self._extrinsic_on_ref_combo_changed()

    def _extrinsic_rebuild_vis_checks(self):
        self._extrinsic_loading_vis = True
        while self._extrinsic_vis_layout.count():
            item = self._extrinsic_vis_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._extrinsic_visibility.clear()
        for lid in self._extrinsic_lidar_ids:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox("雷达 {}".format(lidar_id_to_str(lid)))
            cb.setChecked(True)
            cb.stateChanged.connect(self._extrinsic_on_vis_changed)
            rl.addWidget(cb)
            self._extrinsic_visibility[lid] = cb
            self._extrinsic_vis_layout.addWidget(row)
        self._extrinsic_loading_vis = False

    def _extrinsic_current_lidar(self):
        return self._extrinsic_lidar_combo.currentData()

    def _extrinsic_sync_spins(self):
        lid = self._extrinsic_current_lidar()
        if lid is None:
            return
        self._extrinsic_loading_params = True
        vals = self._extrinsic_offsets.get(lid, [0.0] * 6)
        for i, key in enumerate(PARAM_KEYS):
            self._extrinsic_spins[key].setValue(vals[i])
        self._extrinsic_loading_params = False

    def _extrinsic_on_lidar_selected(self):
        self._extrinsic_sync_spins()
        self._extrinsic_update_color_preview()

    def _extrinsic_on_ref_combo_changed(self):
        ref_lid = self._extrinsic_ref_combo.currentData()
        if ref_lid is not None:
            self._extrinsic_ref_lidar_id = ref_lid
        ref_name = lidar_id_to_str(self._extrinsic_ref_lidar_id or "?")
        self._extrinsic_btn_icp.setText("ICP 多轮精配准（对齐到雷达 {}）".format(ref_name))

    def _extrinsic_on_param_changed(self):
        if self._extrinsic_loading_params:
            return
        lid = self._extrinsic_current_lidar()
        if lid is None:
            return
        self._extrinsic_offsets[lid] = [
            self._extrinsic_spins[k].value() for k in PARAM_KEYS
        ]
        self.vis_fram(updata_color_bar=False)

    def _extrinsic_on_vis_changed(self):
        if not self._extrinsic_loading_vis:
            self.vis_fram(updata_color_bar=False)

    def _extrinsic_set_all_vis(self, visible: bool):
        self._extrinsic_loading_vis = True
        for cb in self._extrinsic_visibility.values():
            cb.setChecked(visible)
        self._extrinsic_loading_vis = False
        self.vis_fram(updata_color_bar=False)

    def _extrinsic_update_color_preview(self):
        lid = self._extrinsic_current_lidar()
        if lid is None:
            return
        r, g, b, a = self._extrinsic_lidar_colors.get(lid, (0.5, 0.5, 0.5, 1.0))
        self._extrinsic_color_preview.setStyleSheet(
            "background:rgba({},{},{},{}); border:1px solid #888;".format(
                int(r * 255), int(g * 255), int(b * 255), a
            )
        )

    def _extrinsic_pick_color(self):
        lid = self._extrinsic_current_lidar()
        if lid is None:
            return
        r, g, b, a = self._extrinsic_lidar_colors.get(lid, default_lidar_color(0))
        c = QColorDialog.getColor(
            QColor(int(r * 255), int(g * 255), int(b * 255)),
            self,
            "雷达颜色",
            QColorDialog.ShowAlphaChannel,
        )
        if c.isValid():
            self._extrinsic_lidar_colors[lid] = (
                c.redF(), c.greenF(), c.blueF(), c.alphaF()
            )
            self._extrinsic_update_color_preview()
            self.vis_fram(updata_color_bar=False)

    def _extrinsic_reset_current(self):
        lid = self._extrinsic_current_lidar()
        if lid is None:
            return
        self._extrinsic_offsets[lid] = [0.0] * 6
        self._extrinsic_sync_spins()
        self.vis_fram(updata_color_bar=False)

    def _extrinsic_reset_all(self):
        self._session_extrinsic_offsets = None
        for lid in self._extrinsic_lidar_ids:
            self._extrinsic_offsets[lid] = [0.0] * 6
        self._extrinsic_sync_spins()
        if self._extrinsic_frame_raw is not None:
            self.structured_points = self._extrinsic_frame_raw.copy()
            self._rebuild_raw_points_from_structured()
        self.vis_fram(updata_color_bar=False)

    def _extrinsic_run_icp(self):
        if self._extrinsic_snapshot is None:
            QMessageBox.information(self, "ICP", "请先加载点云")
            return
        if len(self._extrinsic_lidar_ids) < 2:
            QMessageBox.information(self, "ICP", "至少需要 2 台雷达才能配准")
            return
        ref_lid = self._extrinsic_ref_lidar_id
        if ref_lid is None:
            QMessageBox.warning(self, "ICP", "请先选择配准目标雷达")
            return

        ref_name = lidar_id_to_str(ref_lid)
        reply = QMessageBox.question(
            self,
            "ICP 多轮精配准",
            "方案说明（参考雷达 {} 位姿保持不变）：\n"
            "1. 初对齐：每台雷达对「参考雷达或已对齐雷达」做 ICP，自动选重叠最好的目标；\n"
            "2. 全局迭代：多轮将各雷达对齐到「除自身外所有雷达的融合点云」；\n"
            "3. 逐步缩小对应距离，直至参数收敛。\n\n"
            "结果写入右侧位姿并用于播放/切帧；满意后可点「应用校正」。\n\n"
            "是否继续？".format(ref_name),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            new_offsets, ref_lid, reports, summary = icp_multi_refine_align(
                self._extrinsic_snapshot,
                self.metadata,
                self._extrinsic_offsets,
                self._extrinsic_lidar_ids,
                ref_lid=ref_lid,
                global_refine_rounds=4,
            )
            self._extrinsic_offsets = new_offsets
            self._extrinsic_ref_lidar_id = ref_lid
            self._session_extrinsic_offsets = self._copy_extrinsic_offsets(
                new_offsets
            )
            if self._extrinsic_snapshot is not None:
                self.structured_points = apply_offsets_to_structured(
                    self._extrinsic_snapshot,
                    self.metadata,
                    new_offsets,
                    degrees=True,
                )
                self._rebuild_raw_points_from_structured()
            self._extrinsic_sync_spins()
            self._extrinsic_on_lidar_selected()
            self._extrinsic_on_ref_combo_changed()
            self.vis_fram(updata_color_bar=False)

            lines = [summary, "", "—— 初对齐 ——"]
            for r in reports:
                if r.get("stage") != "pairwise":
                    continue
                lid_s = lidar_id_to_str(r["lidar_id"])
                if r.get("role") in ("skipped", "failed"):
                    lines.append("雷达 {}: {}".format(lid_s, r.get("message", "")))
                elif r.get("role") == "reference":
                    pass
                else:
                    lines.append(
                        "雷达 {}: RMSE={:.4f}m → {}".format(
                            lid_s, r.get("rmse", 0.0), r.get("target", "?")
                        )
                    )
            self._extrinsic_status.setText("\n".join(lines[:12]))
            QMessageBox.information(
                self,
                "ICP 完成",
                summary
                + "\n\n已启用跨帧外参；播放时将自动套用。可继续微调或点「应用校正」。",
            )
        except Exception as e:
            QMessageBox.critical(self, "ICP 失败", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def _extrinsic_visible_ids(self):
        return [
            lid
            for lid in self._extrinsic_lidar_ids
            if self._extrinsic_visibility.get(lid) is None
            or self._extrinsic_visibility[lid].isChecked()
        ]

    def _extrinsic_build_pos_rgba(self, keep_inside_mask=None):
        """按原始点序生成 pos/rgba，与 raw_points 行序一致，便于与掩码/框选共存。"""
        if self._extrinsic_snapshot is None:
            return None, None
        pts_full = structured_to_points_array(self._extrinsic_snapshot, self.metadata)
        n = len(pts_full)
        if n == 0:
            return np.empty((0, 3)), np.empty((0, 4))

        visible = set(self._extrinsic_visible_ids())
        show_mask = np.zeros(n, dtype=bool)
        for lid in self._extrinsic_lidar_ids:
            m = mask_lidar(pts_full, lid)
            if lid in visible:
                show_mask |= m
                xyz_rpy = self._extrinsic_offsets.get(lid, [0.0] * 6)
                if any(abs(v) > 1e-12 for v in xyz_rpy):
                    pts_full[m] = move_pcd_with_xyzrpy(pts_full[m], xyz_rpy, degrees=True)

        pos = pts_full[:, :3].astype(np.float64)
        rgba = np.zeros((n, 4), dtype=np.float32)
        for lid in self._extrinsic_lidar_ids:
            m = mask_lidar(pts_full, lid)
            if lid in visible:
                rgba[m] = self._extrinsic_lidar_colors.get(lid, default_lidar_color(0))

        combined = show_mask
        if keep_inside_mask is not None and len(keep_inside_mask) == n:
            combined = combined & np.asarray(keep_inside_mask, dtype=bool)
        pos = pos[combined]
        rgba = rgba[combined]
        return pos, rgba

    def _extrinsic_apply_commit(self):
        if self._extrinsic_snapshot is None:
            return
        try:
            corrected = apply_offsets_to_structured(
                self._extrinsic_snapshot,
                self.metadata,
                self._extrinsic_offsets,
                degrees=True,
            )
        except Exception as e:
            QMessageBox.critical(self, "应用失败", str(e))
            return
        self._session_extrinsic_offsets = self._copy_extrinsic_offsets(
            self._extrinsic_offsets
        )
        self.structured_points = corrected
        self._rebuild_raw_points_from_structured()
        self._points_rect_select_mask = None
        if self._point_select_dock is not None:
            self._reset_point_select_table_ui()
        self.vis_fram(updata_color_bar=False)
        QMessageBox.information(
            self,
            "外参标定",
            "已应用校正到当前帧；播放/切帧时将自动套用此外参。",
        )
        self._extrinsic_status.setText(
            "已应用校正（播放时自动保持）；可继续微调或导出 JSON"
        )

    def _rebuild_raw_points_from_structured(self):
        fields = self.metadata
        valid = [f for f in fields if f != "_"]
        if not valid:
            return
        points_ = np.asarray(self.structured_points[valid[0]], dtype=np.float64).reshape(-1, 1)
        for field in valid[1:]:
            points_ = np.hstack(
                (
                    points_,
                    np.asarray(self.structured_points[field], dtype=np.float64).reshape(-1, 1),
                )
            )
        self.raw_points = points_

    def _extrinsic_export_json(self):
        if self._extrinsic_snapshot is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            None, "保存偏移", "extrinsic_offsets.json", "JSON (*.json)"
        )
        if not path:
            return
        export_offsets_json(
            path,
            getattr(self, "pcd_file", None),
            self._extrinsic_offsets,
            self._extrinsic_lidar_ids,
            self._extrinsic_ref_lidar_id,
        )
        QMessageBox.information(self, "导出成功", path)

    def _extrinsic_save_pcd(self):
        if self._extrinsic_snapshot is None:
            return
        source_pcd = getattr(self, "pcd_file", None)
        if not source_pcd or not os.path.isfile(source_pcd):
            QMessageBox.warning(self, "保存失败", "无法定位当前帧原始 PCD 路径")
            return
        path, _ = QFileDialog.getSaveFileName(None, "保存校正 PCD", "", "PCD (*.pcd)")
        if not path:
            return
        try:
            corrected = apply_offsets_to_structured(
                self._extrinsic_snapshot,
                self.metadata,
                self._extrinsic_offsets,
                degrees=True,
            )
            save_corrected_structured_with_wata(
                corrected,
                self.metadata,
                source_pcd,
                path,
                metadata=getattr(self, "_extrinsic_pcd_metadata", None),
            )
            QMessageBox.information(self, "保存成功", path)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _extrinsic_batch_folder(self):
        if self._extrinsic_snapshot is None:
            return
        in_dir = QFileDialog.getExistingDirectory(None, "输入 PCD 目录")
        if not in_dir:
            return
        out_dir = QFileDialog.getExistingDirectory(None, "输出目录")
        if not out_dir:
            return
        files = sorted(Path(in_dir).rglob("*.pcd"))
        if not files:
            QMessageBox.warning(self, "批量校正", "输入目录下未找到 .pcd 文件")
            return

        total = len(files)
        progress = QProgressDialog("准备批量校正…", "取消", 0, total, self)
        progress.setWindowTitle("批量校正 PCD")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(420)
        progress.setValue(0)

        ok, fail = 0, 0
        cancelled = False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for i, pcd in enumerate(files):
                if progress.wasCanceled():
                    cancelled = True
                    break
                rel = pcd.relative_to(in_dir)
                progress.setLabelText(
                    "正在处理 ({}/{})\n{}".format(i + 1, total, rel)
                )
                progress.setValue(i)
                QApplication.processEvents()

                out_path = Path(out_dir) / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    structured, file_meta = load_structured_points(str(pcd))
                    ok_f, _ = validate_lidar_id_last_column(file_meta["fields"])
                    if not ok_f:
                        fail += 1
                        continue
                    corrected = apply_offsets_to_structured(
                        structured,
                        file_meta["fields"],
                        self._extrinsic_offsets,
                        degrees=True,
                    )
                    save_corrected_structured_with_wata(
                        corrected,
                        file_meta["fields"],
                        str(pcd),
                        str(out_path),
                        metadata=file_meta,
                    )
                    ok += 1
                except Exception as e:
                    print("批量失败 {}: {}".format(pcd, e))
                    fail += 1

                progress.setValue(i + 1)
                QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()
            progress.setValue(total)

        msg = "成功 {}，失败 {}".format(ok, fail)
        if cancelled:
            msg += "，已取消（未处理 {} 个）".format(total - ok - fail)
        msg += "\n输出目录:\n{}".format(out_dir)
        QMessageBox.information(self, "批量校正完成", msg)

    def _update_extrinsic_button_style(self, active: bool):
        try:
            btn = self.toolbar.widgetForAction(self._extrinsic_calib_action)
            if btn is None:
                return
            if active:
                btn.setStyleSheet(
                    "QToolButton { background-color:#3949ab; color:white; "
                    "border-radius:6px; padding:6px 10px; }"
                )
            else:
                btn.setStyleSheet(
                    "QToolButton { background-color:transparent; border-radius:6px; padding:6px 10px; }"
                )
        except Exception:
            pass
