# PolyVision v2.0

This repository contains the software component of the C.Scope.AI system. C.Scope.AI is an automated imaging system for microplastics sample classification using convolutional neural networks.

PolyVision is the core application software for the full system. It combines:

- The desktop application and workflow management
- GRBL-based control of the automated XY imaging platform
- A retrainable AI/ML pipeline for model inference and continuous learning

## Overview

PolyVision v2.0 is a Windows desktop application for microplastic detection and analysis. It integrates image capture, hardware control, model inference, and retraining support into a single software package.

Key components:

- `PyQt5` for the graphical user interface
- `Detectron2` for object detection and inference
- `OpenCV`, `Pillow`, and custom UI modules for image capture, review, and export
- GRBL/Serial support for automated XY platform motion and measurement
- Retraining support for updating the AI model with new datasets

## Repository structure

```
.
├── ui/                  # main application code and GUI modules
├── Models/              # training scripts, dataset tools, evaluation, experiments
│                        #   (incl. Models/archive/ for earlier iterations)
├── tests/               # automated tests (e.g. GPU retraining runtime checks)
├── docs/                # packaging and GPU retraining guides
├── packaging/           # build and GPU-repair tooling
│   ├── build_exe.bat            # builds the standalone .exe via PyInstaller
│   ├── setup_build.bat          # build environment setup
│   ├── build_requirements.txt   # extra deps needed to build the .exe
│   ├── PolyVision.spec          # PyInstaller spec file
│   └── repair_gpu_env.bat/.py   # source-install GPU environment repair
└── requirements.txt     # Python dependencies for the application
```

> **Note:** `detectron2/` is not included in this repository. It must be cloned separately — see Installation.

## Prerequisites

- Windows 10 or 11
- Python 3.10.13 (64-bit)
- Git

## Installation

1. Install Python 3.10.13:
   https://www.python.org/downloads/windows/

2. Clone this repository.

3. Clone Detectron2:

   ```powershell
   git clone https://github.com/facebookresearch/detectron2.git
   ```

4. Create and activate a virtual environment:

   ```powershell
   py -3.10 -m venv venv
   .\venv\Scripts\activate
   ```

5. Install the required Python packages:

   ```powershell
   pip install -r requirements.txt
   ```

6. Install Detectron2 in editable mode:

   ```powershell
   cd detectron2
   pip install -e .
   cd ..
   ```

## Model files and assets

Large files (model weights, datasets, evaluation outputs) are not included in this repository due to their size. The application resolves them automatically from the `Models/` directory at runtime.

### Manual setup

Download the model folders from this Google Drive parent folder:

https://drive.google.com/drive/folders/1ZMB1I_3Cc35C7hiKsMExzhUH6xYldoSB?usp=drive_link

Then place the following four folders directly inside `Models/`:

- `Models/SEAMaP-Binary-Full`
- `Models/SEAMaP-Binary-Full-6`
- `Models/SEAMaP-Multi-class-100`
- `Models/SEAMaP-Multi-class-100-1`

The final layout should look like this:

```
Models/
├── SEAMaP-Binary-Full/
├── SEAMaP-Binary-Full-6/
├── SEAMaP-Multi-class-100/
└── SEAMaP-Multi-class-100-1/
```

## Running the application

Start the main application from the `ui` directory:

```powershell
cd ui
python PolyVisionMain.py
```

## GPU retraining support

PolyVision checks for a usable NVIDIA CUDA stack when retraining starts. If the
full CUDA stack (PyTorch, Torchvision, and the Detectron2 native extension)
validates, retraining and the post-training benchmark run on the GPU; otherwise
PolyVision falls back to CPU when a valid CPU stack is available.

To check the runtime on a source/venv install, run this from the repository root
(note the `ui\` path prefix, unlike the application itself, which is launched
from inside the `ui` directory):

```powershell
python ui\PolyVisionMain.py --diagnose-retraining --require-gpu
```

Exit code `0` = requested runtime ready, `1` = retraining unavailable, `2` =
available but the `--require-gpu` requirement is not met. Add `--json` for
machine-readable output.

For everything else — the packaged-user guidance, repairing a broken source
install, and the GPU packaging/distribution handoff — see the dedicated guides:

- [GPU Retraining User Guide](docs/GPU_RETRAINING_USER_GUIDE.md) — for packaged
  end-users, plus the source-install repair procedure for administrators and
  developers.
- [Packaging Guide](docs/PACKAGING_GUIDE.md) — the build and distribution flow,
  plus the GPU build prerequisites (`TORCH_CUDA_ARCH_LIST` architecture coverage
  and the packaged acceptance gate).

## Notes

- This repository is dedicated to the PolyVision v2.0 software package only.
- PolyVision is the main software component of the C.Scope.AI automated imaging system.
- If you encounter issues, verify that Detectron2 is installed correctly and that the virtual environment is active.
- Use the `UI/` folder as the primary entry point for the application.
