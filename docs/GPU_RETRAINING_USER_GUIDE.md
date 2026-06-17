# PolyVision GPU Retraining User Guide

## For Regular PolyVision Users

If you normally open PolyVision by double-clicking `PolyVision.exe`, you use the
packaged application.

You do **not** need to install:

- CUDA Toolkit 11.8 or NVCC
- Python or Python packages
- Detectron2
- Visual Studio build tools

These technical components must already be included and tested by the person
who prepared the PolyVision application.

The only GPU-related software a regular user may occasionally need updated is
the NVIDIA graphics driver. A graphics driver is different from CUDA Toolkit.
Ask your PolyVision support person or system administrator to check the driver;
do not install CUDA Toolkit yourself.

### When PolyVision Says GPU Retraining Is Unavailable

1. Read the message shown by PolyVision.
2. If **Continue on CPU** is available, select it to continue retraining.
   Retraining will work but may take longer.
3. If PolyVision says the retraining runtime is unavailable, close the message
   and contact your PolyVision support person.
4. Take a screenshot of the message and send it to support.

Do not download CUDA Toolkit, change application files, or run repair scripts.
Support will determine whether the NVIDIA graphics driver needs an update or
the packaged application needs a corrected build.

### Information to Send to Support

Send the following:

- A screenshot of the complete PolyVision error message
- Whether **Continue on CPU** was shown
- The computer name or workstation label
- Whether an NVIDIA graphics card is installed, if known

## For PolyVision Administrators and Developers

This section applies only to a source installation that is started from Python
and contains the project `venv` folder. It does not apply to ordinary packaged
application users.

CUDA Toolkit 11.8/NVCC is required on the technical build or repair computer
because Detectron2's native GPU extension must be compiled there. End users of
the packaged application do not compile Detectron2.

### Check Before Repairing

From the project root:

```powershell
.\packaging\repair_gpu_env.bat --preflight-only
```

The preflight makes no package changes. It checks:

- The project virtual environment
- Detectron2 source files
- A CUDA 11.8-compatible Microsoft C++ compiler
- CUDA Toolkit 11.8 and NVCC
- Internet access

Resolve every reported prerequisite before running the repair.

CUDA 11.8 requires an MSVC compiler version below `19.40`. If the preflight
reports that a newer compiler such as `19.44` is incompatible, open **Visual
Studio Installer**, modify **Build Tools 2022**, and install the individual
component **MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.39 - 17.9)**.
Keep newer toolsets installed; PolyVision will automatically prefer the
compatible `14.3x` toolset for the Detectron2 build.

### Repair a Source Installation

Close PolyVision, then run:

```powershell
.\packaging\repair_gpu_env.bat
```

The repair first builds and validates a replacement Detectron2 wheel in a
temporary environment. It changes the active virtual environment only after the
temporary GPU validation succeeds. Restart PolyVision after the repair completes.
Details are written to `logs/repair_gpu_env.log`.

### Validate Before Packaging

From the project root:

```powershell
venv\Scripts\python.exe UI\PolyVisionMain.py --diagnose-retraining --require-gpu --json
```

Do not package or distribute the application unless the command exits
successfully and reports:

- `"retraining_available": true`
- `"cpu_ready": true`
- `"gpu_ready": true`
- `"detectron2_native_available": true`

Run the equivalent diagnostic against the packaged executable before giving it
to users. If it fails, correct and rebuild the application instead of asking the
user to install CUDA Toolkit or repair Python packages.
