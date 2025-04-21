# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['qtvis.py'],
    pathex=[],
    binaries=[],
    datas=[('icons/color.png', 'icons'), ('icons/next.png', 'icons'), ('icons/open.png', 'icons'), ('icons/open_dir.png', 'icons'), ('icons/open_folder.svg', 'icons'), ('icons/pause_pcd.png', 'icons'), ('icons/play_pcd.png', 'icons'), ('icons/pointsize.png', 'icons'), ('icons/pointsize_decrease.png', 'icons'), ('icons/pointsize_increase.png', 'icons'), ('icons/prev_pcd.png', 'icons'), ('icons/fengguangming.ttf', 'icons')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
    name='qtvis',
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
)
