# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path
import detectron2
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

# Get the project root directory
try:
    spec_path = Path(__file__).resolve()
except NameError:
    spec_path = Path(sys.argv[0]).resolve()

project_root = spec_path.parent
ui_path = project_root / "UI"

# Data files to include
# Models/ is NOT bundled here — it stays external next to PolyVision.exe
datas = [
    (str(ui_path / "res"), "res"),
]

# Expand individual files so PyInstaller sees concrete paths
datas += [(str(path), ".") for path in ui_path.glob("*.ui")]
datas += [(str(path), ".") for path in project_root.glob("*.txt")]
datas += [(str(path), ".") for path in project_root.glob("*.md")]
datas += [(str(path), ".") for path in ui_path.glob("*.json")]

detectron2_configs_dir = Path(detectron2.__file__).resolve().parent / "model_zoo" / "configs"
if detectron2_configs_dir.exists():
    datas.append((str(detectron2_configs_dir), "detectron2/model_zoo/configs"))

for package in ("torch", "torchvision", "detectron2", "fvcore", "iopath", "pycocotools"):
    try:
        datas += copy_metadata(package)
    except Exception as exc:
        print(f"WARNING: could not collect metadata for {package}: {exc}")

try:
    datas += collect_data_files("detectron2", includes=["model_zoo/configs/**/*"])
except Exception as exc:
    print(f"WARNING: could not collect Detectron2 data files: {exc}")

# Hidden imports for modules that PyInstaller might miss
hiddenimports = [
    'PyQt5.QtCore',
    'PyQt5.QtGui', 
    'PyQt5.QtWidgets',
    'PyQt5.QtMultimedia',
    'cv2',
    'numpy',
    'PIL',
    'PIL.Image',
    'serial',
    'serial.tools.list_ports',
    'detectron2',
    'detectron2._C',
    'detectron2.engine',
    'detectron2.config',
    'detectron2.model_zoo',
    'detectron2.data',
    'detectron2.utils.visualizer',
    'torch',
    'torchvision',
    'yaml',
    'sqlite3',
    'json',
    'requests',
    'supervision',
    'roboflow',
    'pycocotools',
    'matplotlib',
    'pandas',
    'sklearn',
    'tqdm',
]

for package in ("retraining_runtime", "detectron2", "torchvision.ops"):
    try:
        hiddenimports += collect_submodules(package)
    except Exception as exc:
        print(f"WARNING: could not collect hidden imports for {package}: {exc}")

# Native ML binaries. The post-build diagnostic is still mandatory because
# missing CUDA/Torch/Detectron2 DLLs may only surface in the frozen app.
binaries = []
for package in ("torch", "torchvision", "detectron2"):
    try:
        binaries += collect_dynamic_libs(package)
    except Exception as exc:
        print(f"WARNING: could not collect dynamic libraries for {package}: {exc}")

upx_exclude = ["*.dll", "*.pyd", "torch*", "torchvision*", "detectron2*"]

a = Analysis(
    [str(ui_path / 'PolyVisionMain.py')],  # Main script
    pathex=[str(project_root), str(ui_path)],  # Search paths
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib.tests',
        'numpy.tests', 
        'pandas.tests',
        'PIL.tests',
        'torch.test',
        'detectron2.tests'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PolyVision',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=upx_exclude,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ui_path / 'res' / 'PolyVisionLogo.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=upx_exclude,
    name='PolyVision',
)
