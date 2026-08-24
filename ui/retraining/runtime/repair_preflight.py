from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


EXPECTED_CUDA_VERSION = "11.8"
MINIMUM_MSVC_VERSION = (19, 10)
MAXIMUM_MSVC_VERSION_EXCLUSIVE = (19, 40)


@dataclass
class RepairPreflight:
    ready: bool
    checks: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _normalize_cuda_version(text: str) -> str:
    match = re.search(r"(\d+\.\d+)", text or "")
    return match.group(1) if match else ""


def _normalize_msvc_version(text: str) -> tuple[str, tuple[int, int] | None]:
    match = re.search(r"\bVersion\s+(\d+)\.(\d+)", text or "", re.IGNORECASE)
    if not match:
        return "", None
    major, minor = int(match.group(1)), int(match.group(2))
    return f"{major}.{minor}", (major, minor)


def _msvc_compiler_check(compiler: str) -> tuple[str, str]:
    try:
        result = subprocess.run(
            [compiler],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        version_text, version = _normalize_msvc_version(result.stdout + result.stderr)
        if version is None:
            return "", f"Could not determine the MSVC compiler version from {compiler}."
        if not (MINIMUM_MSVC_VERSION <= version < MAXIMUM_MSVC_VERSION_EXCLUSIVE):
            return "", (
                f"MSVC compiler {version_text} is incompatible with CUDA {EXPECTED_CUDA_VERSION}. "
                "Install the Visual Studio 2022 MSVC v143 14.39 (17.9) or an earlier 14.3x "
                "toolset, then rerun the repair preflight."
            )
        return f"MSVC compiler: {compiler} (version {version_text})", ""
    except Exception as exc:
        return "", f"Could not validate the MSVC compiler at {compiler}: {exc}"


def _current_torch_cuda_check() -> tuple[str, str]:
    try:
        import torch

        torch_cuda = _normalize_cuda_version(str(getattr(torch.version, "cuda", "") or ""))
        if torch_cuda == EXPECTED_CUDA_VERSION:
            return f"Current PyTorch CUDA build: {torch_cuda}", ""
        return (
            f"Current PyTorch CUDA build will be replaced: {torch_cuda or 'CPU-only/unavailable'}",
            "",
        )
    except Exception as exc:
        return f"Current PyTorch installation will be replaced: unavailable ({exc})", ""


def _torch_cuda_home() -> str:
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        return str(CUDA_HOME or "")
    except Exception:
        return ""


def run_repair_preflight(project_root: Path) -> RepairPreflight:
    checks: List[str] = []
    errors: List[str] = []

    if sys.prefix == sys.base_prefix:
        errors.append("The repair must run from PolyVision's virtual environment.")
    else:
        checks.append(f"Virtual environment: {sys.prefix}")

    detectron2_setup = project_root / "detectron2" / "setup.py"
    if detectron2_setup.is_file():
        checks.append(f"Detectron2 source: {detectron2_setup.parent}")
    else:
        errors.append(f"Detectron2 source was not found at {detectron2_setup.parent}.")

    compiler = shutil.which("cl.exe") or shutil.which("cl")
    if compiler:
        compiler_check, compiler_error = _msvc_compiler_check(compiler)
        if compiler_check:
            checks.append(compiler_check)
        if compiler_error:
            errors.append(compiler_error)
    else:
        errors.append(
            "MSVC cl.exe was not found on PATH. Run the repair from an x64 Native Tools "
            "Command Prompt with Visual Studio C++ build tools installed."
        )

    torch_check, torch_error = _current_torch_cuda_check()
    checks.append(torch_check)
    if torch_error:
        errors.append(torch_error)

    try:
        default_cuda_home = Path(
            rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{EXPECTED_CUDA_VERSION}"
        )
        cuda_home_value = os.getenv("CUDA_PATH") or os.getenv("CUDA_HOME") or _torch_cuda_home()
        if not cuda_home_value and default_cuda_home.is_dir():
            cuda_home_value = str(default_cuda_home)
        cuda_home = Path(cuda_home_value) if cuda_home_value else None
        nvcc = cuda_home / "bin" / "nvcc.exe" if cuda_home else None
        if not nvcc or not nvcc.is_file():
            errors.append(
                f"CUDA {EXPECTED_CUDA_VERSION} toolkit/NVCC was not found. "
                "Install the matching toolkit and ensure CUDA_HOME is configured."
            )
        else:
            result = subprocess.run(
                [str(nvcc), "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            nvcc_version = _normalize_cuda_version(result.stdout + result.stderr)
            if result.returncode != 0 or nvcc_version != EXPECTED_CUDA_VERSION:
                errors.append(
                    f"NVCC is {nvcc_version or 'unavailable'}; expected {EXPECTED_CUDA_VERSION}."
                )
            else:
                checks.append(f"CUDA toolkit/NVCC: {nvcc_version} ({cuda_home})")
    except Exception as exc:
        errors.append(f"Could not validate CUDA build prerequisites: {exc}")

    try:
        request = urllib.request.Request(
            "https://download.pytorch.org/whl/cu118/",
            headers={"User-Agent": "PolyVision-GPU-Repair"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            if int(getattr(response, "status", 200)) >= 400:
                raise RuntimeError(f"HTTP {response.status}")
        checks.append("Network access: download.pytorch.org reachable")
    except Exception as exc:
        errors.append(f"Cannot reach download.pytorch.org: {exc}")

    return RepairPreflight(ready=not errors, checks=checks, errors=errors)


def print_repair_preflight(report: RepairPreflight) -> None:
    print("PolyVision GPU repair preflight")
    print("=" * 40)
    for check in report.checks:
        print(f"OK: {check}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print("Preflight result:", "ready" if report.ready else "blocked")
