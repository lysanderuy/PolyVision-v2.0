# GPU Build & Repair Setup

This guide is for **developers and administrators** preparing a build or repair
machine for PolyVision's GPU retraining runtime. It applies only to a source
installation that is started from Python and contains the project `venv` folder.

It does **not** apply to ordinary packaged-application users. If you double-click
`PolyVision.exe`, see the [GPU Retraining User Guide](GPU_RETRAINING_USER_GUIDE.md)
instead.

CUDA Toolkit 11.8/NVCC is required on the build or repair computer because
Detectron2's native GPU extension must be compiled there. End users of the
packaged application never compile Detectron2.

## Build-machine prerequisites: installing the GPU toolchain

Install these on the build/repair machine, in this order, before running the
repair. The repair preflight (below) checks each one and reports what is missing.

### 1. MSVC v143 14.39 toolset

CUDA 11.8 requires an MSVC compiler version below `19.40`. If the preflight
reports a newer compiler (for example `19.43` or `19.44`) as incompatible, open
**Visual Studio Installer**, choose **Modify** on **Build Tools 2022**, and under
**Individual components** install **"MSVC v143 - VS 2022 C++ x64/x86 build tools
(v14.39 - 17.9)"**. Keep any newer toolsets installed; PolyVision automatically
prefers the compatible `14.3x` toolset for the Detectron2 build.

### 2. CUDA Toolkit 11.8

Install the **CUDA Toolkit 11.8** from NVIDIA's CUDA 11.8 download archive
(`developer.nvidia.com/cuda-11-8-0-download-archive`). The version must be 11.8 —
newer CUDA toolkits are not compatible with the shipped Detectron2 build.

### 3. Set `CUDA_HOME`

Point `CUDA_HOME` at the 11.8 installation so the build can find NVCC, for
example:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```

### 4. Verify before repairing

Confirm both tools resolve to the expected versions:

```powershell
nvcc --version   # must report release 11.8
cl               # must report a 14.39.x build (Visual Studio 2022)
```

## Check Before Repairing

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

Resolve every reported prerequisite (see the toolchain steps above) before
running the repair. The preflight must report `Preflight result: ok`.

## Repair a Source Installation

Close PolyVision, then run:

```powershell
.\packaging\repair_gpu_env.bat
```

The repair first builds and validates a replacement Detectron2 wheel in a
temporary environment. It changes the active virtual environment only after the
temporary GPU validation succeeds. Restart PolyVision after the repair completes.
Details are written to `logs/repair_gpu_env.log`.

## Detectron2 GPU Architecture Coverage

Detectron2 CUDA kernels are compiled for specific NVIDIA compute capabilities.
Set `TORCH_CUDA_ARCH_LIST` before compiling Detectron2 so the build supports the
GPUs expected in the field.

Example only:

```powershell
$env:TORCH_CUDA_ARCH_LIST = "7.5;8.6;8.9"
```

Choose the final list from the actual supported hardware fleet. If this is not
set, the build may only support the GPU installed in the build machine.

After setting the architecture list, rebuild and validate the source runtime:

```powershell
.\packaging\repair_gpu_env.bat --preflight-only
.\packaging\repair_gpu_env.bat
venv\Scripts\python.exe ui\PolyVisionMain.py --diagnose-retraining --require-gpu --json
```

## Validate Before Packaging

From the project root:

```powershell
venv\Scripts\python.exe ui\PolyVisionMain.py --diagnose-retraining --require-gpu --json
```

Do not package or distribute the application unless the command exits with code
`0` and meets the
[GPU-ready pass criteria](../README.md#gpu-ready-pass-criteria) (the single
source of truth for those fields). For what each exit code means and what to do
next, see the [exit-code table](../README.md#gpu-retraining).

Run the equivalent diagnostic against the packaged executable before giving it to
users. If it fails, correct and rebuild the application instead of asking the user
to install CUDA Toolkit or repair Python packages. The full build and
distribution flow is in the [Packaging Guide](PACKAGING_GUIDE.md).
