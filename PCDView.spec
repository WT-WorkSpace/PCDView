# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['qtvis.py'],
    pathex=[],
    binaries=[('/home/wt/miniconda3/envs/pcdview/lib/libexpat.so.1', '.')],
    datas=[('icons/app.ico', 'icons'), ('icons/color.svg', 'icons'), ('icons/coordinate.svg', 'icons'), ('icons/fengguangming.ttf', 'icons'), ('icons/next.png', 'icons'), ('icons/next.png', 'icons'), ('icons/open.svg', 'icons'), ('icons/open_dir.svg', 'icons'), ('icons/box_selection.svg', 'icons'), ('icons/cancel_box_selection.svg', 'icons'), ('icons/mask.svg', 'icons'), ('icons/pause_pcd.png', 'icons'), ('icons/play_pcd.png', 'icons'), ('icons/pointsize.png', 'icons'), ('icons/pointsize_decrease.png', 'icons'), ('icons/pointsize_increase.png', 'icons'), ('icons/prev_pcd.png', 'icons'), ('icons/cluster.svg', 'icons'), ('icons/load_view.svg', 'icons'), ('icons/save_view.svg', 'icons'), ('icons/open_boxes_dir.svg', 'icons'), ('icons/wangge.svg', 'icons'), ('icons/add_bbox.svg', 'icons'), ('icons/extrinsic_calib.svg', 'icons')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hooks/pyi_rth_libexpat.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PCDView',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icons/app.ico'],
)
