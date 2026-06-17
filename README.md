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

- `UI/` – main application code and GUI modules
- `Models/` – training scripts, dataset tools, evaluation code, and experiments
- `requirements.txt` – Python dependencies for the application
- `docs/` – packaging and GPU retraining guides
- `packaging/` – build and GPU-repair tooling:
  - `build_exe.bat` – builds the standalone executable via PyInstaller
  - `setup_build.bat` – environment setup script for the build process
  - `build_requirements.txt` – additional dependencies required for building the executable
  - `PolyVision.spec` – PyInstaller spec file
  - `repair_gpu_env.bat` / `repair_gpu_env.py` – source-install GPU environment repair

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

## GPU retraining support

> **Packaged application users:** Do not install CUDA Toolkit 11.8, NVCC, Python,
> or Detectron2. The packaged application must already contain a working
> retraining runtime. If GPU retraining is unavailable, continue on CPU when
> PolyVision offers that option or contact your PolyVision support person for a
> corrected application build.

PolyVision automatically checks for NVIDIA CUDA support when retraining starts.
The diagnostic validates PyTorch, Torchvision native operations, the Detectron2
native extension, and CUDA build compatibility. If the complete CUDA stack is
usable, retraining and the post-training benchmark use the GPU. CPU fallback is
allowed only when the complete CPU retraining stack passes validation.

See [GPU Retraining User Guide](docs/GPU_RETRAINING_USER_GUIDE.md) for plain-language
instructions for packaged users and a separate technical repair procedure for
administrators and developers.

See [GPU Packaging Guide](docs/GPU_PACKAGING_GUIDE.md) for the GPU-machine packaging
handoff, PyInstaller gates, and `TORCH_CUDA_ARCH_LIST` requirements.

Run the same validation used by source and packaged builds:

```powershell
python UI\PolyVisionMain.py --diagnose-retraining
python UI\PolyVisionMain.py --diagnose-retraining --require-gpu
python UI\PolyVisionMain.py --diagnose-retraining --require-gpu --json
```

The commands return exit code `0` when the requested runtime is ready, `1` when
retraining is unavailable, and `2` when retraining is available but the
`--require-gpu` requirement is not met.

For source/venv installs, a technical administrator or developer can repair the
environment when NVIDIA hardware is detected but the Python ML stack is
CPU-only or broken. This is not an end-user repair. It requires internet access,
MSVC C++ build tools, and the CUDA 11.8 toolkit/NVCC:

```powershell
.\packaging\repair_gpu_env.bat --preflight-only
.\packaging\repair_gpu_env.bat
```

The repair performs all prerequisite checks first, builds and validates a
replacement Detectron2 wheel in a temporary environment, and changes the active
venv only after that validation passes. Close PolyVision before continuing in
the repair window, then restart PolyVision after repair completes. Details are
written to `logs/repair_gpu_env.log`.

Packaged PyInstaller `.exe` builds cannot be repaired in place this way. Do not
ask packaged users to install CUDA Toolkit 11.8 or run `repair_gpu_env.bat`.
They need a corrected GPU-enabled build. A packaged build with a valid CPU stack
can continue retraining on CPU; a packaged build with a broken retraining
runtime blocks retraining.

Before handing a build to packaging, require this command to pass:

```powershell
python UI\PolyVisionMain.py --diagnose-retraining --require-gpu
```

After packaging, run the equivalent command against `PolyVision.exe` to catch
missing native extensions or DLLs before distribution. Retraining writes new
models under the external `Models` workspace, so that workspace must remain
writable in the installed application layout.

Detectron2 source builds may include kernels only for the build machine's GPU.
For a distributed GPU build, the packaging owner must set
`TORCH_CUDA_ARCH_LIST` for every supported NVIDIA GPU architecture and run the
packaged diagnostic on representative target hardware. The diagnostic executes
a Detectron2 CUDA native operation so unsupported architectures fail before
retraining starts.

## Running the application

Start the main application from the `UI` directory:

```powershell
cd UI
python PolyVisionMain.py
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
  SEAMaP-Binary-Full/
  SEAMaP-Binary-Full-6/
  SEAMaP-Multi-class-100/
  SEAMaP-Multi-class-100-1/
```

## Notes

- This repository is dedicated to the PolyVision v2.0 software package only.
- PolyVision is the main software component of the C.Scope.AI automated imaging system.
- If you encounter issues, verify that Detectron2 is installed correctly and that the virtual environment is active.
- Use the `UI/` folder as the primary entry point for the application.
