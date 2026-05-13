from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RETRAINING_DIR = PROJECT_ROOT / "Models" / "Retraining"
if str(RETRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(RETRAINING_DIR))

from gpu_diagnostics import diagnose_gpu_support, format_diagnostic_lines


def main() -> int:
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

    try:
        import detectron2
        print(f"Detectron2 import: OK ({getattr(detectron2, '__version__', 'unknown version')})")
    except Exception as exc:
        print(f"Detectron2 import: FAILED ({exc})")
        return 1

    if report.selected_device != "cuda":
        print("\nGPU verification did not select CUDA. Review the diagnostic above.")
        return 2

    print("\nGPU verification passed. Restart PolyVision before retraining.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
