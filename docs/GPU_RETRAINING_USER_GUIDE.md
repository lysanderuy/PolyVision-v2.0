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

Setting up a GPU build or repair machine — installing CUDA 11.8 and the MSVC
toolset, running `repair_gpu_env.bat`, and validating the runtime before
packaging — is covered in the
[GPU Build & Repair Setup](GPU_BUILD_SETUP.md) guide. That work does not apply to
packaged-application users and is kept out of this guide on purpose.
