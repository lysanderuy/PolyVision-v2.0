from __future__ import annotations

import argparse
from pathlib import Path
import sys

from UI.retraining_runtime.diagnostic_cli import main as diagnostic_main
from UI.retraining_runtime.gpu_diagnostics import diagnose_gpu_support, format_diagnostic_lines
from UI.retraining_runtime.repair_preflight import print_repair_preflight, run_repair_preflight


PROJECT_ROOT = Path(__file__).resolve().parent


def _remove_project_root_import_shadow() -> None:
    """Make diagnostics import installed packages the same way UI/PolyVisionMain.py does."""
    for entry in list(sys.path):
        try:
            resolved = Path(entry or Path.cwd()).resolve()
        except OSError:
            continue
        if resolved == PROJECT_ROOT:
            sys.path.remove(entry)


def verify_environment() -> int:
    _remove_project_root_import_shadow()
    print("PolyVision GPU environment verification")
    print("=" * 45)
    report = diagnose_gpu_support()
    for line in format_diagnostic_lines(report):
        print(line)

    try:
        import cv2
        import numpy
        from PIL.ImageQt import ImageQt
        print(f"NumPy import: OK ({numpy.__version__})")
        print(f"OpenCV import: OK ({cv2.__version__})")
        print("Pillow ImageQt import: OK")
    except Exception as exc:
        print(f"Core GUI/vision import: FAILED ({exc})")
        return 1

    if not report.retraining_available:
        print("\nRetraining runtime verification failed. Review the diagnostic above.")
        return 1

    if not report.gpu_ready:
        print("\nGPU verification did not select CUDA. Review the diagnostic above.")
        return 2

    print("\nGPU verification passed. Restart PolyVision before retraining.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--diagnose-retraining", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--preflight-repair", action="store_true")
    args, _ = parser.parse_known_args()
    if args.preflight_repair:
        report = run_repair_preflight(PROJECT_ROOT)
        print_repair_preflight(report)
        return 0 if report.ready else 3
    if args.diagnose_retraining:
        _remove_project_root_import_shadow()
        forwarded = ["--diagnose-retraining"]
        if args.require_gpu:
            forwarded.append("--require-gpu")
        if args.json:
            forwarded.append("--json")
        return diagnostic_main(forwarded)
    return verify_environment()


if __name__ == "__main__":
    raise SystemExit(main())
