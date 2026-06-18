from __future__ import annotations

from .gpu_diagnostics import GPU_INSUFFICIENT_VRAM, GpuDiagnostic, should_offer_repair


READY = "ready"
BLOCKED = "blocked"
CPU_FALLBACK = "cpu_fallback"
REPAIR_OR_CPU = "repair_or_cpu"


def retraining_start_policy(report: GpuDiagnostic, frozen: bool) -> str:
    if not report.retraining_available:
        return BLOCKED
    # The GPU works but is too small for training; offer CPU without a repair
    # option, since there is nothing to repair.
    if report.status == GPU_INSUFFICIENT_VRAM:
        return CPU_FALLBACK
    if report.gpu_ready or not should_offer_repair(report):
        return READY
    if frozen:
        return CPU_FALLBACK
    return REPAIR_OR_CPU
