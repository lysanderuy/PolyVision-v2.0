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
- `detectron2/` – Detectron2 source for model inference and training support
- `Models/` – training scripts, dataset tools, evaluation code, and experiments
- `requirements.txt` – Python dependencies for the application

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

## Model configuration

The model path is configured in the application source. Example:

```python
MODEL_PATH = "../SEAMaP-Binary-Full/faster_rcnn_R_50_FPN_3x/2025-05-30-13-16-04/model_final.pth"
```

Large model files are stored externally and are not included in this repository.

## Notes

- This repository is dedicated to the PolyVision v2.0 software package only.
- PolyVision is the main software component of the C.Scope.AI automated imaging system.
- If you encounter issues, verify that Detectron2 is installed correctly and that the virtual environment is active.
- Use the `UI/` folder as the primary entry point for the application.
