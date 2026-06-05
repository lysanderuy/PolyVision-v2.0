from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .gpu_diagnostics import (
    diagnose_gpu_support,
    diagnostic_exit_code,
    print_diagnostic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the PolyVision retraining runtime.")
    parser.add_argument(
        "--diagnose-retraining",
        action="store_true",
        help="Run retraining runtime diagnostics and exit.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail unless the complete GPU retraining runtime is usable.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the diagnostic report as JSON.",
    )
    return parser


def requested(argv: Sequence[str]) -> bool:
    return "--diagnose-retraining" in argv


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = diagnose_gpu_support()
    print_diagnostic(report, as_json=args.json)
    return diagnostic_exit_code(report, require_gpu=args.require_gpu)


if __name__ == "__main__":
    raise SystemExit(main())
