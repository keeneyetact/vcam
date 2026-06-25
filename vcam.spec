# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for a single-file vcam.exe (the GUI control panel).

Build:  pyinstaller --noconfirm vcam.spec
Output: dist/vcam.exe
"""
from PyInstaller.utils.hooks import collect_all

datas = [("models/rvm_mobilenetv3_fp32.onnx", "models")]   # bundle the RVM model
binaries = []
hiddenimports = []

# Native packages whose DLLs/data PyInstaller's default analysis can miss.
for pkg in ("onnxruntime", "pyvirtualcam", "cv2"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["vcam_ui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # MediaPipe (and its heavy deps) are not used by the RVM engine; excluding
    # them keeps the exe small and avoids MediaPipe's painful packaging.
    excludes=["mediapipe", "matplotlib", "jax", "jaxlib", "tensorflow",
              "torch", "pandas", "scipy", "IPython", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="vcam",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,          # GUI app — no console window
    disable_windowed_traceback=False,
    icon=None,
)
