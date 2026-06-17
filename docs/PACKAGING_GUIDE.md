# Packaging Guide

This guide is for the developer or build agent creating the packaged
`PolyVision.exe`. It is not for regular PolyVision users.

The packaged build is produced with PyInstaller from the project `venv`. Because
PolyVision ships with a GPU-capable retraining runtime, most of the work happens
before the build: the source environment must be GPU-ready and validated. The
GPU build prerequisites below cover that; the Build, Distribution Rules, and
Packaged Acceptance Gate sections cover the general packaging flow.

## GPU build prerequisites

### Build Machine Requirement

The GPU-enabled package must be built on a Windows machine with an NVIDIA GPU.
PyInstaller freezes the current Python environment; it cannot add GPU support
to a CPU-only environment.

A single GPU-enabled package can still serve CPU-only users. At runtime,
PolyVision selects CUDA only when the complete GPU stack passes diagnostics.
Otherwise it falls back to CPU when the CPU retraining stack is valid.

### Required Source Runtime

Before packaging, the source environment must pass:

```powershell
venv\Scripts\python.exe UI\PolyVisionMain.py --diagnose-retraining --require-gpu --json
$LASTEXITCODE
```

The exit code must be `0`, and the JSON must report:

- `"status": "gpu_ready"`
- `"selected_device": "cuda"`
- `"retraining_available": true`
- `"cpu_ready": true`
- `"gpu_ready": true`
- `"detectron2_native_available": true`
- `"detectron2_cuda_available": true`

Do not package if this command fails.

### Detectron2 GPU Architecture Coverage

Detectron2 CUDA kernels are compiled for specific NVIDIA compute capabilities.
Set `TORCH_CUDA_ARCH_LIST` before compiling Detectron2 so the package supports
the GPUs expected in the field.

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
venv\Scripts\python.exe UI\PolyVisionMain.py --diagnose-retraining --require-gpu --json
```

## Build

Install PyInstaller tooling:

```powershell
.\packaging\setup_build.bat
```

Build the one-folder package:

```powershell
.\packaging\build_exe.bat
```

`build_exe.bat` now:

- blocks packaging if the source runtime is not GPU-ready;
- uses `PolyVision.spec`;
- copies `Models` next to `PolyVision.exe` when available;
- runs the packaged executable diagnostic before reporting success.

## Distribution Rules

- Use the one-folder `dist\PolyVision` output.
- Do not use a one-file executable.
- Keep `Models` next to `PolyVision.exe`.
- Install the app and `Models` in a user-writable location because retraining
  writes new model files there.
- Do not ask packaged users to install CUDA Toolkit, Python packages, or run
  `repair_gpu_env.bat`.

## Packaged Acceptance Gate

Before distribution, this must pass on the packaged executable:

```powershell
.\dist\PolyVision\PolyVision.exe --diagnose-retraining --require-gpu --json
$LASTEXITCODE
```

If it fails, inspect the PyInstaller output for missing hidden imports, native
DLLs, `detectron2._C.pyd`, Torch/Torchvision binaries, or missing CUDA
architecture coverage. Fix and rebuild instead of shipping the package.
