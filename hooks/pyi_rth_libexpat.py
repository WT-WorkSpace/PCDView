# PyInstaller runtime hook: ensure bundled libexpat loads before pyexpat.
import os
import sys

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    meipass = sys._MEIPASS
    os.environ["LD_LIBRARY_PATH"] = meipass + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    for name in ("libexpat.so.1", "libexpat.so.1.12.0", "libexpat.so"):
        path = os.path.join(meipass, name)
        if os.path.isfile(path):
            try:
                import ctypes

                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break
