from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional


GPU_READY = "gpu_ready"
HARDWARE_PRESENT_SOFTWARE_MISSING = "hardware_present_software_missing"
HARDWARE_MISSING = "hardware_missing"
CUDA_BROKEN = "cuda_broken"


@dataclass
class GpuDiagnostic:
    status: str
    selected_device: str
    hardware_present: bool
    nvidia_gpus: List[str] = field(default_factory=list)
    torch_version: Optional[str] = None
    torch_cuda_version: Optional[str] = None
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_name: Optional[str] = None
    reason: str = ""
    repair_recommended: bool = False
    force_cpu: bool = False
    errors: List[str] = field(default_factory=list)


def _creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _detect_nvidia_gpus() -> tuple[bool, List[str], Optional[str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=_creation_flags(),
        )
    except FileNotFoundError:
        return False, [], "nvidia-smi was not found."
    except Exception as exc:
        return False, [], f"nvidia-smi check failed: {exc}"

    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "nvidia-smi returned an error."
        return False, [], error

    gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return bool(gpus), gpus, None


def diagnose_gpu_support() -> GpuDiagnostic:
    force_cpu = os.getenv("POLYVISION_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}
    hardware_present, nvidia_gpus, nvidia_error = _detect_nvidia_gpus()
    errors: List[str] = []
    if nvidia_error:
        errors.append(nvidia_error)

    try:
        import torch
    except Exception as exc:
        status = HARDWARE_PRESENT_SOFTWARE_MISSING if hardware_present else HARDWARE_MISSING
        return GpuDiagnostic(
            status=status,
            selected_device="cpu",
            hardware_present=hardware_present,
            nvidia_gpus=nvidia_gpus,
            reason=f"PyTorch could not be imported: {exc}",
            repair_recommended=hardware_present,
            force_cpu=force_cpu,
            errors=errors + [str(exc)],
        )

    torch_version = getattr(torch, "__version__", None)
    torch_cuda_version = getattr(torch.version, "cuda", None)
    cuda_available = False
    cuda_device_count = 0
    cuda_device_name = None

    try:
        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
        if cuda_available and cuda_device_count > 0:
            cuda_device_name = torch.cuda.get_device_name(0)
            probe = torch.tensor([1.0], device="cuda")
            _ = probe * 2
            torch.cuda.synchronize()
    except Exception as exc:
        return GpuDiagnostic(
            status=CUDA_BROKEN,
            selected_device="cpu",
            hardware_present=hardware_present or cuda_available,
            nvidia_gpus=nvidia_gpus,
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            cuda_device_name=cuda_device_name,
            reason=f"CUDA was detected but failed a runtime allocation test: {exc}",
            repair_recommended=hardware_present,
            force_cpu=force_cpu,
            errors=errors + [str(exc)],
        )

    if cuda_available and cuda_device_count > 0:
        selected_device = "cpu" if force_cpu else "cuda"
        reason = "CUDA runtime is available."
        if force_cpu:
            reason = "POLYVISION_FORCE_CPU is set, so CPU was selected even though CUDA is available."
        return GpuDiagnostic(
            status=GPU_READY,
            selected_device=selected_device,
            hardware_present=True,
            nvidia_gpus=nvidia_gpus or ([cuda_device_name] if cuda_device_name else []),
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            cuda_device_name=cuda_device_name,
            reason=reason,
            repair_recommended=False,
            force_cpu=force_cpu,
            errors=errors,
        )

    if hardware_present:
        if torch_cuda_version is None or "+cpu" in str(torch_version).lower():
            status = HARDWARE_PRESENT_SOFTWARE_MISSING
            reason = "NVIDIA hardware was detected, but this PyTorch install is CPU-only."
        else:
            status = CUDA_BROKEN
            reason = "NVIDIA hardware was detected, but PyTorch CUDA is not available."
        return GpuDiagnostic(
            status=status,
            selected_device="cpu",
            hardware_present=True,
            nvidia_gpus=nvidia_gpus,
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            cuda_device_name=cuda_device_name,
            reason=reason,
            repair_recommended=True,
            force_cpu=force_cpu,
            errors=errors,
        )

    return GpuDiagnostic(
        status=HARDWARE_MISSING,
        selected_device="cpu",
        hardware_present=False,
        nvidia_gpus=nvidia_gpus,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        reason="No NVIDIA CUDA-capable GPU was detected.",
        repair_recommended=False,
        force_cpu=force_cpu,
        errors=errors,
    )


def select_training_device(report: Optional[GpuDiagnostic] = None) -> str:
    report = report or diagnose_gpu_support()
    return report.selected_device


def format_diagnostic_lines(report: GpuDiagnostic) -> List[str]:
    lines = [
        "--- GPU Diagnostic ---",
        f"Status: {report.status}",
        f"Selected device: {report.selected_device}",
        f"Reason: {report.reason}",
        f"PyTorch: {report.torch_version or 'unavailable'}",
        f"PyTorch CUDA build: {report.torch_cuda_version or 'none'}",
        f"CUDA available: {report.cuda_available}",
        f"CUDA device count: {report.cuda_device_count}",
    ]
    if report.cuda_device_name:
        lines.append(f"CUDA device: {report.cuda_device_name}")
    if report.nvidia_gpus:
        lines.append(f"NVIDIA hardware: {', '.join(report.nvidia_gpus)}")
    if report.force_cpu:
        lines.append("Force CPU override: enabled")
    if report.repair_recommended:
        lines.append("Repair recommended: run repair_gpu_env.bat from the project root.")
    if report.errors:
        lines.append(f"Diagnostic notes: {' | '.join(report.errors)}")
    return lines


def should_offer_repair(report: GpuDiagnostic) -> bool:
    return report.repair_recommended and report.status in {
        HARDWARE_PRESENT_SOFTWARE_MISSING,
        CUDA_BROKEN,
    }


if __name__ == "__main__":
    for diagnostic_line in format_diagnostic_lines(diagnose_gpu_support()):
        print(diagnostic_line)
