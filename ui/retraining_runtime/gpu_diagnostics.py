from __future__ import annotations

import importlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


GPU_READY = "gpu_ready"
GPU_INSUFFICIENT_VRAM = "gpu_insufficient_vram"
HARDWARE_PRESENT_SOFTWARE_MISSING = "hardware_present_software_missing"
HARDWARE_MISSING = "hardware_missing"
CUDA_BROKEN = "cuda_broken"
RETRAINING_RUNTIME_BROKEN = "retraining_runtime_broken"

# A 4 GiB card is the practical floor for Faster R-CNN R50-FPN at the configured
# batch size and input resolution. Cards report slightly less than their nominal
# size, so admit anything at or above 3.5 GiB (covers 4 GB-class GPUs) and route
# smaller cards to CPU instead of letting training start and crash with a CUDA
# out-of-memory error. Set POLYVISION_ALLOW_SMALL_GPU=1 to bypass this guard.
MINIMUM_GPU_MEMORY_BYTES = int(3.5 * 1024 ** 3)


@dataclass
class GpuDiagnostic:
    status: str
    selected_device: str
    hardware_present: bool
    retraining_available: bool = False
    cpu_ready: bool = False
    gpu_ready: bool = False
    nvidia_gpus: List[str] = field(default_factory=list)
    torch_version: Optional[str] = None
    torch_cuda_version: Optional[str] = None
    torchvision_version: Optional[str] = None
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_name: Optional[str] = None
    gpu_total_memory_bytes: Optional[int] = None
    detectron2_version: Optional[str] = None
    detectron2_native_available: bool = False
    detectron2_cuda_available: Optional[bool] = None
    detectron2_cuda_version: Optional[str] = None
    reason: str = ""
    repair_recommended: bool = False
    force_cpu: bool = False
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "selected_device": self.selected_device,
            "hardware_present": self.hardware_present,
            "retraining_available": self.retraining_available,
            "cpu_ready": self.cpu_ready,
            "gpu_ready": self.gpu_ready,
            "nvidia_gpus": self.nvidia_gpus,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "torchvision_version": self.torchvision_version,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_name": self.cuda_device_name,
            "gpu_total_memory_bytes": self.gpu_total_memory_bytes,
            "detectron2_version": self.detectron2_version,
            "detectron2_native_available": self.detectron2_native_available,
            "detectron2_cuda_available": self.detectron2_cuda_available,
            "detectron2_cuda_version": self.detectron2_cuda_version,
            "reason": self.reason,
            "repair_recommended": self.repair_recommended,
            "force_cpu": self.force_cpu,
            "errors": self.errors,
        }


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


def _normalize_cuda_version(version: Optional[str]) -> Optional[str]:
    if version is None:
        return None
    text = str(version).strip()
    digits = []
    current = ""
    for char in text:
        if char.isdigit() or char == ".":
            current += char
        elif current:
            digits.append(current.strip("."))
            current = ""
    if current:
        digits.append(current.strip("."))
    for candidate in digits:
        parts = [part for part in candidate.split(".") if part]
        if len(parts) >= 2:
            return ".".join(parts[:2])
        if len(parts) == 1 and len(parts[0]) >= 3:
            return f"{parts[0][:-1]}.{parts[0][-1]}"
    return text or None


def _probe_torchvision(torch_module, device: str = "cpu") -> tuple[Optional[str], Optional[str]]:
    version = None
    try:
        torchvision = importlib.import_module("torchvision")
        version = getattr(torchvision, "__version__", None)
        ops = importlib.import_module("torchvision.ops")
        boxes = torch_module.tensor(
            [[0.0, 0.0, 10.0, 10.0]],
            dtype=torch_module.float32,
            device=device,
        )
        scores = torch_module.tensor([0.9], dtype=torch_module.float32, device=device)
        result = ops.nms(boxes, scores, 0.5)
        if int(result.numel()) != 1:
            raise RuntimeError("torchvision NMS returned an unexpected result.")
        return version, None
    except Exception as exc:
        return version, f"Torchvision native operations failed on {device}: {exc}"


def _probe_detectron2() -> tuple[
    Optional[str],
    bool,
    Optional[bool],
    Optional[str],
    Optional[str],
]:
    try:
        detectron2 = importlib.import_module("detectron2")
        version = getattr(detectron2, "__version__", None)
    except Exception as exc:
        return None, False, None, None, f"Detectron2 could not be imported: {exc}"

    try:
        native = importlib.import_module("detectron2._C")
        has_cuda = bool(getattr(native, "has_cuda", lambda: False)())
        cuda_version = str(getattr(native, "get_cuda_version", lambda: "")()) or None
        return version, True, has_cuda, cuda_version, None
    except Exception as exc:
        return version, False, None, None, f"Detectron2 native extension could not be imported: {exc}"


def _probe_detectron2_native_op(torch_module, device: str = "cpu") -> Optional[str]:
    try:
        boxes = torch_module.tensor(
            [[5.0, 5.0, 4.0, 2.0, 0.0]],
            dtype=torch_module.float32,
            device=device,
        )
        scores = torch_module.tensor([0.9], dtype=torch_module.float32, device=device)
        result = torch_module.ops.detectron2.nms_rotated(boxes, scores, 0.5)
        if int(result.numel()) != 1:
            raise RuntimeError("Detectron2 rotated NMS returned an unexpected result.")
        if device == "cuda":
            torch_module.cuda.synchronize()
        return None
    except Exception as exc:
        return f"Detectron2 native operation failed on {device}: {exc}"


def diagnose_gpu_support() -> GpuDiagnostic:
    force_cpu = os.getenv("POLYVISION_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}
    allow_small_gpu = os.getenv("POLYVISION_ALLOW_SMALL_GPU", "").strip().lower() in {"1", "true", "yes", "on"}
    hardware_present, nvidia_gpus, nvidia_error = _detect_nvidia_gpus()
    errors: List[str] = []
    if nvidia_error:
        errors.append(nvidia_error)

    try:
        import torch
    except Exception as exc:
        return GpuDiagnostic(
            status=RETRAINING_RUNTIME_BROKEN,
            selected_device="unavailable",
            hardware_present=hardware_present,
            retraining_available=False,
            cpu_ready=False,
            gpu_ready=False,
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
    gpu_total_memory_bytes: Optional[int] = None

    try:
        cpu_probe = torch.tensor([1.0], device="cpu")
        _ = cpu_probe * 2
    except Exception as exc:
        return GpuDiagnostic(
            status=RETRAINING_RUNTIME_BROKEN,
            selected_device="unavailable",
            hardware_present=hardware_present,
            retraining_available=False,
            cpu_ready=False,
            gpu_ready=False,
            nvidia_gpus=nvidia_gpus,
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            reason=f"PyTorch CPU operations failed: {exc}",
            repair_recommended=hardware_present,
            force_cpu=force_cpu,
            errors=errors + [str(exc)],
        )

    torchvision_version, torchvision_error = _probe_torchvision(torch)
    detectron2_version, detectron2_native, detectron2_has_cuda, detectron2_cuda_version, detectron2_error = (
        _probe_detectron2()
    )
    if torchvision_error:
        errors.append(torchvision_error)
    if detectron2_error:
        errors.append(detectron2_error)

    detectron2_cpu_error = None
    if detectron2_native:
        detectron2_cpu_error = _probe_detectron2_native_op(torch, device="cpu")
        if detectron2_cpu_error:
            errors.append(detectron2_cpu_error)

    cpu_ready = torchvision_error is None and detectron2_native and detectron2_cpu_error is None
    cuda_error = None
    torchvision_gpu_error = None
    detectron2_gpu_error = None
    try:
        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
        if cuda_available and cuda_device_count > 0:
            cuda_device_name = torch.cuda.get_device_name(0)
            try:
                gpu_total_memory_bytes = int(torch.cuda.get_device_properties(0).total_memory)
            except Exception:
                # Fail open: if total memory can't be read, skip the VRAM guard
                # rather than blocking a GPU that might be perfectly usable.
                gpu_total_memory_bytes = None
            probe = torch.tensor([1.0], device="cuda")
            _ = probe * 2
            torch.cuda.synchronize()
            _, torchvision_gpu_error = _probe_torchvision(torch, device="cuda")
            if torchvision_gpu_error:
                errors.append(torchvision_gpu_error)
            if detectron2_native and detectron2_has_cuda:
                detectron2_gpu_error = _probe_detectron2_native_op(torch, device="cuda")
                if detectron2_gpu_error:
                    errors.append(detectron2_gpu_error)
    except Exception as exc:
        cuda_error = f"CUDA runtime allocation failed: {exc}"
        errors.append(cuda_error)

    torch_cuda_normalized = _normalize_cuda_version(torch_cuda_version)
    detectron2_cuda_normalized = _normalize_cuda_version(detectron2_cuda_version)
    cuda_versions_match = (
        torch_cuda_normalized is not None
        and detectron2_cuda_normalized is not None
        and torch_cuda_normalized == detectron2_cuda_normalized
    )
    gpu_ready = bool(
        cuda_available
        and cuda_device_count > 0
        and cuda_error is None
        and torchvision_gpu_error is None
        and detectron2_gpu_error is None
        and detectron2_has_cuda
        and cuda_versions_match
    )

    if not cpu_ready:
        return GpuDiagnostic(
            status=RETRAINING_RUNTIME_BROKEN,
            selected_device="unavailable",
            hardware_present=hardware_present or cuda_available,
            retraining_available=False,
            cpu_ready=False,
            gpu_ready=False,
            nvidia_gpus=nvidia_gpus or ([cuda_device_name] if cuda_device_name else []),
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            torchvision_version=torchvision_version,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            cuda_device_name=cuda_device_name,
            detectron2_version=detectron2_version,
            detectron2_native_available=detectron2_native,
            detectron2_cuda_available=detectron2_has_cuda,
            detectron2_cuda_version=detectron2_cuda_version,
            reason="The Detectron2 retraining runtime is incomplete or unusable.",
            repair_recommended=hardware_present or cuda_available,
            force_cpu=force_cpu,
            errors=errors,
        )

    if gpu_ready:
        gpu_too_small = (
            gpu_total_memory_bytes is not None
            and gpu_total_memory_bytes < MINIMUM_GPU_MEMORY_BYTES
            and not allow_small_gpu
            and not force_cpu
        )
        if gpu_too_small:
            have_gib = gpu_total_memory_bytes / (1024 ** 3)
            need_gib = MINIMUM_GPU_MEMORY_BYTES / (1024 ** 3)
            return GpuDiagnostic(
                status=GPU_INSUFFICIENT_VRAM,
                selected_device="cpu",
                hardware_present=True,
                retraining_available=True,
                cpu_ready=True,
                gpu_ready=False,
                nvidia_gpus=nvidia_gpus or ([cuda_device_name] if cuda_device_name else []),
                torch_version=torch_version,
                torch_cuda_version=torch_cuda_version,
                torchvision_version=torchvision_version,
                cuda_available=cuda_available,
                cuda_device_count=cuda_device_count,
                cuda_device_name=cuda_device_name,
                gpu_total_memory_bytes=gpu_total_memory_bytes,
                detectron2_version=detectron2_version,
                detectron2_native_available=detectron2_native,
                detectron2_cuda_available=detectron2_has_cuda,
                detectron2_cuda_version=detectron2_cuda_version,
                reason=(
                    f"GPU has {have_gib:.1f} GiB of memory; retraining needs about "
                    f"{need_gib:.1f} GiB, so CPU was selected."
                ),
                repair_recommended=False,
                force_cpu=force_cpu,
                errors=errors,
            )

        selected_device = "cpu" if force_cpu else "cuda"
        reason = "CUDA runtime is available."
        if force_cpu:
            reason = "POLYVISION_FORCE_CPU is set, so CPU was selected even though CUDA is available."
        return GpuDiagnostic(
            status=GPU_READY,
            selected_device=selected_device,
            hardware_present=True,
            retraining_available=True,
            cpu_ready=True,
            gpu_ready=True,
            nvidia_gpus=nvidia_gpus or ([cuda_device_name] if cuda_device_name else []),
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            torchvision_version=torchvision_version,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            cuda_device_name=cuda_device_name,
            gpu_total_memory_bytes=gpu_total_memory_bytes,
            detectron2_version=detectron2_version,
            detectron2_native_available=detectron2_native,
            detectron2_cuda_available=detectron2_has_cuda,
            detectron2_cuda_version=detectron2_cuda_version,
            reason=reason,
            repair_recommended=False,
            force_cpu=force_cpu,
            errors=errors,
        )

    if hardware_present or cuda_available:
        if cuda_error:
            status = CUDA_BROKEN
            reason = cuda_error
        elif torchvision_gpu_error:
            status = CUDA_BROKEN
            reason = torchvision_gpu_error
        elif detectron2_gpu_error:
            status = CUDA_BROKEN
            reason = detectron2_gpu_error
        elif torch_cuda_version is None or "+cpu" in str(torch_version).lower():
            status = HARDWARE_PRESENT_SOFTWARE_MISSING
            reason = "NVIDIA hardware was detected, but this PyTorch install is CPU-only."
        elif not detectron2_has_cuda:
            status = HARDWARE_PRESENT_SOFTWARE_MISSING
            reason = "NVIDIA hardware was detected, but Detectron2 was built without CUDA support."
        elif not cuda_versions_match:
            status = CUDA_BROKEN
            reason = (
                "Detectron2 and PyTorch use different CUDA builds "
                f"({detectron2_cuda_version or 'unknown'} vs {torch_cuda_version or 'unknown'})."
            )
        else:
            status = CUDA_BROKEN
            reason = "NVIDIA hardware was detected, but PyTorch CUDA is not available."
        return GpuDiagnostic(
            status=status,
            selected_device="cpu",
            hardware_present=True,
            retraining_available=True,
            cpu_ready=True,
            gpu_ready=False,
            nvidia_gpus=nvidia_gpus or ([cuda_device_name] if cuda_device_name else []),
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            torchvision_version=torchvision_version,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            cuda_device_name=cuda_device_name,
            detectron2_version=detectron2_version,
            detectron2_native_available=detectron2_native,
            detectron2_cuda_available=detectron2_has_cuda,
            detectron2_cuda_version=detectron2_cuda_version,
            reason=reason,
            repair_recommended=True,
            force_cpu=force_cpu,
            errors=errors,
        )

    return GpuDiagnostic(
        status=HARDWARE_MISSING,
        selected_device="cpu",
        hardware_present=False,
        retraining_available=True,
        cpu_ready=True,
        gpu_ready=False,
        nvidia_gpus=nvidia_gpus,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        torchvision_version=torchvision_version,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        detectron2_version=detectron2_version,
        detectron2_native_available=detectron2_native,
        detectron2_cuda_available=detectron2_has_cuda,
        detectron2_cuda_version=detectron2_cuda_version,
        reason="No NVIDIA CUDA-capable GPU was detected.",
        repair_recommended=False,
        force_cpu=force_cpu,
        errors=errors,
    )


def select_training_device(report: Optional[GpuDiagnostic] = None) -> str:
    report = report or diagnose_gpu_support()
    if not report.retraining_available:
        raise RuntimeError(f"Retraining runtime is unavailable: {report.reason}")
    return report.selected_device


def format_diagnostic_lines(report: GpuDiagnostic) -> List[str]:
    lines = [
        "--- GPU Diagnostic ---",
        f"Status: {report.status}",
        f"Selected device: {report.selected_device}",
        f"Reason: {report.reason}",
        f"Retraining available: {report.retraining_available}",
        f"CPU retraining ready: {report.cpu_ready}",
        f"GPU retraining ready: {report.gpu_ready}",
        f"PyTorch: {report.torch_version or 'unavailable'}",
        f"PyTorch CUDA build: {report.torch_cuda_version or 'none'}",
        f"Torchvision: {report.torchvision_version or 'unavailable'}",
        f"CUDA available: {report.cuda_available}",
        f"CUDA device count: {report.cuda_device_count}",
        f"Detectron2: {report.detectron2_version or 'unavailable'}",
        f"Detectron2 native extension: {'available' if report.detectron2_native_available else 'unavailable'}",
        f"Detectron2 CUDA support: {report.detectron2_cuda_available}",
        f"Detectron2 CUDA build: {report.detectron2_cuda_version or 'none'}",
    ]
    if report.cuda_device_name:
        lines.append(f"CUDA device: {report.cuda_device_name}")
    if report.gpu_total_memory_bytes:
        lines.append(f"GPU memory: {report.gpu_total_memory_bytes / (1024 ** 3):.1f} GiB")
    if report.nvidia_gpus:
        lines.append(f"NVIDIA hardware: {', '.join(report.nvidia_gpus)}")
    if report.force_cpu:
        lines.append("Force CPU override: enabled")
    if report.repair_recommended:
        lines.append(
            "Repair guidance: packaged users need a corrected application build; "
            "technical source-install administrators can run packaging\\repair_gpu_env.bat."
        )
    if report.errors:
        lines.append(f"Diagnostic notes: {' | '.join(report.errors)}")
    return lines


def should_offer_repair(report: GpuDiagnostic) -> bool:
    return report.repair_recommended and report.status in {
        HARDWARE_PRESENT_SOFTWARE_MISSING,
        CUDA_BROKEN,
        RETRAINING_RUNTIME_BROKEN,
    }


def diagnostic_exit_code(report: GpuDiagnostic, require_gpu: bool = False) -> int:
    if not report.retraining_available:
        return 1
    if require_gpu and not report.gpu_ready:
        return 2
    return 0


def print_diagnostic(report: GpuDiagnostic, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
        return
    for diagnostic_line in format_diagnostic_lines(report):
        print(diagnostic_line)
