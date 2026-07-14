#!/usr/bin/env bash
set -eu

# 使用当前环境的 Python 运行 PyInstaller（建议先 conda activate pcdview）
CONDA_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
LIBEXPAT="${CONDA_PREFIX}/lib/libexpat.so.1"
if [[ ! -f "${LIBEXPAT}" ]]; then
  echo "错误: 未找到 libexpat: ${LIBEXPAT}" >&2
  exit 1
fi

python -m PyInstaller --onefile --windowed \
  --runtime-hook hooks/pyi_rth_libexpat.py \
  --add-binary "${LIBEXPAT}:." \
  --name "PCDView" \
  --icon "icons/app.ico" \
  --add-data "icons/app.ico:icons" \
  --add-data "icons/color.svg:icons" \
  --add-data "icons/coordinate.svg:icons" \
  --add-data "icons/fengguangming.ttf:icons" \
  --add-data "icons/next.png:icons" \
  --add-data "icons/next.png:icons" \
  --add-data "icons/open.svg:icons" \
  --add-data "icons/history.svg:icons" \
  --add-data "icons/help.svg:icons" \
  --add-data "icons/open_dir.svg:icons" \
  --add-data "icons/box_selection.svg:icons" \
  --add-data "icons/cancel_box_selection.svg:icons" \
  --add-data "icons/mask.svg:icons" \
  --add-data "icons/mask_mask.svg:icons" \
  --add-data "icons/pause_pcd.png:icons" \
  --add-data "icons/play_pcd.png:icons" \
  --add-data "icons/pointsize.png:icons" \
  --add-data "icons/pointsize_decrease.png:icons" \
  --add-data "icons/pointsize_increase.png:icons" \
  --add-data "icons/prev_pcd.png:icons" \
  --add-data "icons/cluster.svg:icons" \
  --add-data "icons/load_view.svg:icons" \
  --add-data "icons/save_view.svg:icons" \
  --add-data "icons/seg.png:icons" \
  --add-data "icons/map.png:icons" \
  --add-data "icons/open_boxes_dir.svg:icons" \
  --add-data "icons/wangge.svg:icons" \
  --add-data "icons/add_bbox.svg:icons" \
  --add-data "icons/calibration.svg:icons" \
  qtvis.py

# 打包完成后在桌面创建快捷方式
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="${SCRIPT_DIR}/dist/PCDView"
ICON="${SCRIPT_DIR}/icons/app.ico"
DESKTOP_FILE=""

if [[ ! -x "${EXE}" ]]; then
  echo "错误: 未找到可执行文件 ${EXE}" >&2
  exit 1
fi

DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
if [[ -z "${DESKTOP_DIR}" || ! -d "${DESKTOP_DIR}" ]]; then
  for dir in "${HOME}/Desktop" "${HOME}/桌面"; do
    if [[ -d "${dir}" ]]; then
      DESKTOP_DIR="${dir}"
      break
    fi
  done
fi
if [[ -z "${DESKTOP_DIR}" ]]; then
  DESKTOP_DIR="${HOME}/Desktop"
  mkdir -p "${DESKTOP_DIR}"
fi

DESKTOP_FILE="${DESKTOP_DIR}/PCDView.desktop"
cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=PCDView
Name[zh_CN]=PCDView
Comment=Point Cloud Viewer
Comment[zh_CN]=点云可视化工具
Exec=${EXE}
Icon=${ICON}
Terminal=false
Categories=Graphics;Science;
StartupNotify=true
EOF

chmod +x "${DESKTOP_FILE}"
# GNOME 等桌面需标记为“信任”，否则双击无反应
if command -v gio >/dev/null 2>&1; then
  gio set "${DESKTOP_FILE}" metadata::trusted true 2>/dev/null || true
fi

echo "已在桌面创建快捷方式: ${DESKTOP_FILE}"
