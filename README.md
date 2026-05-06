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

## Running the application

Start the main application from the `UI` directory:

```powershell
cd UI
python PolyVisionMain.py
```

## Model files and assets

Large files (model weights, datasets, evaluation outputs) are not included in this repository due to their size. The application resolves them automatically from the `Models/` directory at runtime.

### Assets to download

The following files should be obtained from your project's external storage (Google Drive, etc.):

| Path | Size | Purpose |
|------|------|---------|
| `Models/SEAMaP-Binary-Full/` | Large | Pre-trained binary classification model |
| `Models/SEAMaP-Binary-Full-6/` | Large | Binary model variant |
| `Models/SEAMaP-Multi-class-100/` | Large | Pre-trained multi-class model |
| `Models/SEAMaP-Multi-class-100-1/` | Large | Multi-class model variant |
| `Models/Retraining/original_datasets/` | Very Large | Baseline datasets for model retraining |

### Automatic download (recommended)

Use the provided `download_assets.py` script to automatically download all shared folders from Google Drive:

1. For each shared folder, get the folder ID from the Google Drive URL:
   - Open the shared folder
   - Copy the URL: `https://drive.google.com/drive/folders/FOLDER_ID`
   - Extract the `FOLDER_ID` part

2. Edit `assets_config.json` and replace the placeholder folder IDs:
   ```json
   {
     "assets": {
       "Models/SEAMaP-Binary-Full": {
         "folder_url": "https://drive.google.com/drive/folders/1ABC123DEF...",
         "description": "Binary microplastics detection model (main)"
       },
       ...
     }
   }
   ```

3. Run the script:
   ```powershell
   python download_assets.py
   ```

   Or download specific assets:
   ```powershell
   python download_assets.py Models/SEAMaP-Binary-Full
   ```

The script will automatically download and organize all folders into the correct directory structure.

### Manual setup

Alternatively, obtain the model files from your external storage and place them manually in the expected structure:

```
Models/
  SEAMaP-Binary-Full/
    faster_rcnn_R_50_FPN_3x/
      <timestamp>/
        model_final.pth
  SEAMaP-Multi-class-100/
    faster_rcnn_R_50_FPN_3x/
      <timestamp>/
        model_final.pth
```

## Notes

- This repository is dedicated to the PolyVision v2.0 software package only.
- PolyVision is the main software component of the C.Scope.AI automated imaging system.
- If you encounter issues, verify that Detectron2 is installed correctly and that the virtual environment is active.
- Use the `UI/` folder as the primary entry point for the application.
