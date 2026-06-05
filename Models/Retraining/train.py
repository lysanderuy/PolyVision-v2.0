"""Compatibility wrapper for direct source execution of the retraining CLI."""

from pathlib import Path
import sys


UI_DIR = Path(__file__).resolve().parents[2] / "UI"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from retraining_runtime.train import *  # noqa: F401,F403


if __name__ == "__main__":
    from retraining_runtime.train import cli_main

    cli_main()
