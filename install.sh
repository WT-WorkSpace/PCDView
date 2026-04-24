#!/usr/bin/env bash
set -eu

# 使用 miniconda view 环境的 Python 运行 PyInstaller
/home/wt/miniconda3/envs/view/bin/python -m PyInstaller --onefile --windowed \
  --name "PCDView" \
  --icon "icons/app.ico" \
  --add-data "icons/color.svg:icons" \
  --add-data "icons/coordinate.svg:icons" \
  --add-data "icons/fengguangming.ttf:icons" \
  --add-data "icons/next.png:icons" \
  --add-data "icons/open.svg:icons" \
  --add-data "icons/open_dir.svg:icons" \
  --add-data "icons/box_selection.svg:icons" \
  --add-data "icons/cancel_box_selection.svg:icons" \
  --add-data "icons/mask.svg:icons" \
  --add-data "icons/pause_pcd.png:icons" \
  --add-data "icons/play_pcd.png:icons" \
  --add-data "icons/pointsize.png:icons" \
  --add-data "icons/pointsize_decrease.png:icons" \
  --add-data "icons/pointsize_increase.png:icons" \
  --add-data "icons/prev_pcd.png:icons" \
  --add-data "icons/cluster.svg:icons" \
  --add-data "icons/load_view.svg:icons" \
  --add-data "icons/save_view.svg:icons" \
  --add-data "icons/open_boxes_dir.svg:icons" \
  --add-data "icons/wangge.svg:icons" \
  --add-data "icons/add_bbox.svg:icons" \
  qtvis.py

# 生成桌面快捷方式（可直接在桌面看到图标）
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DESKTOP_DIR="$HOME/Desktop"
APP_BIN="$PROJECT_DIR/dist/PCDView"
ICON_PATH="$PROJECT_DIR/icons/app.ico"
DESKTOP_FILE="$DESKTOP_DIR/PCDView.desktop"

mkdir -p "$DESKTOP_DIR"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=PCDView
Comment=Point Cloud Viewer
Exec=$APP_BIN
Icon=$ICON_PATH
Terminal=false
Categories=Graphics;Development;
StartupNotify=true
EOF

chmod +x "$DESKTOP_FILE"
# GNOME: 标记为受信任的桌面启动器，避免“Untrusted Desktop File”提示
if command -v gio >/dev/null 2>&1; then
  gio set "$DESKTOP_FILE" metadata::trusted true || true
fi
echo "桌面图标已生成: $DESKTOP_FILE"