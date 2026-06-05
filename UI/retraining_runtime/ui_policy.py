from __future__ import annotations

from .gpu_diagnostics import GpuDiagnostic, should_offer_repair


READY = "ready"
BLOCKED = "blocked"
CPU_FALLBACK = "cpu_fallback"
REPAIR_OR_CPU = "repair_or_cpu"


def retraining_start_policy(report: GpuDiagnostic, frozen: bool) -> str:
    if not report.retraining_available:
        return BLOCKED
    if report.gpu_ready or not should_offer_repair(report):
        return READY
    if frozen:
        return CPU_FALLBACK
    return REPAIR_OR_CPU
