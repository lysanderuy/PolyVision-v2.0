# Packaging Guide

This guide is for the developer or build agent creating the packaged
`PolyVision.exe`. It is not for regular PolyVision users.

The packaged build is produced with PyInstaller from the project `venv`. Because
PolyVision ships with a GPU-capable retraining runtime, most of the work happens
before the build: the source environment must be GPU-ready and validated. The
GPU build prerequisites below cover that; the Build, Distribution Rules, and
Packaged Acceptance Gate sections cover the general packaging flow.

## Running from source vs. packaging

These are two different things with two different requirements:

- **Running PolyVision from source** does not require a GPU. If the diagnostic
  passes *without* `--require-gpu` (exit code `0`), retraining works on CPU — just
  slower. No CUDA setup needed.
- **Building a distributable package is GPU-only by design.** `build_exe.bat`
  refuses to package unless the source runtime is GPU-ready, because the shipped
  `PolyVision.exe` is always GPU-enabled — a single package serves both CPU-only
  and GPU users in the field (see *Build Machine Requirement* below).

So if you only need to run and retrain locally, you can stop here and use CPU. To
produce a package, you must satisfy the GPU build prerequisites that follow.

## GPU build prerequisites

Before packaging, the build machine must have the GPU toolchain installed and a
GPU-ready source environment. Installing CUDA 11.8 and the MSVC toolset, running
the repair, and choosing Detectron2 architecture coverage are all covered in
[GPU Build & Repair Setup](GPU_BUILD_SETUP.md). This section covers only what
packaging itself requires.

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
venv\Scripts\python.exe ui\PolyVisionMain.py --diagnose-retraining --require-gpu --json
$LASTEXITCODE
```

The exit code must be `0` and the JSON must meet the
[GPU-ready pass criteria](../README.md#gpu-ready-pass-criteria) (the single
source of truth for those fields).

If the exit code is not `0`, do not package — the build is GPU-only by design.
Repair the GPU stack first; see
[GPU Build & Repair Setup](GPU_BUILD_SETUP.md) and the
[exit-code table](../README.md#gpu-retraining). The Detectron2 architecture
coverage (`TORCH_CUDA_ARCH_LIST`) that determines which field GPUs the package
supports is also covered there.

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
