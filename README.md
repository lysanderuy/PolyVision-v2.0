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
- `build_requirements.txt` – additional dependencies required for building the executable
- `build_exe.bat` – script to build the standalone executable via PyInstaller
- `setup_build.bat` – environment setup script for the build process
- `PolyVision.spec` / `UI/PolyVisionMain.spec` – PyInstaller spec files

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

PolyVision automatically checks for NVIDIA CUDA support when retraining starts.
If CUDA-capable PyTorch and Detectron2 are installed correctly, retraining and
the post-training benchmark use the GPU. If CUDA is unavailable, PolyVision
falls back to CPU and logs the reason.

For source/venv installs, PolyVision can guide repair when NVIDIA hardware is
detected but the Python ML stack is CPU-only or broken. The repair requires
internet access and reinstalls PyTorch with CUDA 11.8 support:

```powershell
python -m pip install --upgrade --force-reinstall torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e detectron2
python repair_gpu_env.py
```

You can also launch the guided repair script from the project root:

```powershell
.\repair_gpu_env.bat
```

Close PolyVision before continuing in the repair window, then restart PolyVision
after repair completes. If a repair is interrupted, run `.\repair_gpu_env.bat`
again from the project root; it is safe to rerun and writes details to
`repair_gpu_env.log`.

Packaged PyInstaller `.exe` builds cannot be repaired in place this way; they
need a separate GPU-enabled build or an external managed environment.

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
