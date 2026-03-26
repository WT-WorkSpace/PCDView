# 使用 miniconda view 环境的 Python 运行 PyInstaller
/home/wt/miniconda3/envs/view/bin/python -m PyInstaller --onefile --windowed \
  --add-data "icons/color.svg:icons"\
  --add-data "icons/coordinate.svg:icons"\
  --add-data "icons/fengguangming.ttf:icons"\
  --add-data "icons/next.png:icons"\
  --add-data "icons/open.png:icons"\
  --add-data "icons/open_dir.png:icons"\
  --add-data "icons/open_folder.svg:icons"\
  --add-data "icons/pause_pcd.png:icons"\
  --add-data "icons/play_pcd.png:icons"\
  --add-data "icons/pointsize.png:icons"\
  --add-data "icons/pointsize_decrease.png:icons"\
  --add-data "icons/pointsize_increase.png:icons"\
  --add-data "icons/prev_pcd.png:icons"\
  --add-data "icons/load_view.svg:icons"\
  --add-data "icons/save_view.svg:icons"\
  --add-data "icons/open_boxes_dir.svg:icons"\
  qtvis.py

rm ~/Desktop/qtvis
mv dist/qtvis ~/Desktop/