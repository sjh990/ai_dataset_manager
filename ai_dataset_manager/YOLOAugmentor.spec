# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec (Windows / Ubuntu 공용). 실행: python -m PyInstaller --noconfirm --clean YOLOAugmentor.spec
import sys

block_cipher = None
is_win = sys.platform.startswith("win")

# 사용하지 않는 대형 Qt 모듈 제외 → 용량 절감
EXCLUDES = [
    "tkinter", "matplotlib", "scipy", "pandas", "IPython", "pytest",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtQuick3D",
    "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtPdf",
    "PySide6.QtBluetooth", "PySide6.QtPositioning", "PySide6.QtSql", "PySide6.QtTest",
    # 생성형(SDXL) 스택은 exe에 넣지 않음 (수 GB). SDXL은 run.bat/run.sh(venv) 실행에서 사용.
    "torch", "torchvision", "diffusers", "transformers", "accelerate", "safetensors", "triton", "nvidia",
]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=[("assets/icon.png", "assets")],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="YOLOAugmentor",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon="assets/icon.ico" if is_win else None,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name="YOLOAugmentor")
