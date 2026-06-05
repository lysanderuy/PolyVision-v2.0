"""Compatibility wrapper for source scripts using the canonical UI runtime."""

from pathlib import Path
import sys


UI_DIR = Path(__file__).resolve().parents[2] / "UI"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from retraining_runtime.gpu_diagnostics import *  # noqa: F401,F403


if __name__ == "__main__":
    from retraining_runtime.gpu_diagnostics import diagnose_gpu_support, print_diagnostic

    print_diagnostic(diagnose_gpu_support())
