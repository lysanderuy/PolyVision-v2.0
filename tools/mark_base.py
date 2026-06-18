#!/usr/bin/env python3
"""Dev tool: stamp a model folder as the protected base model.

A base model is identified by the presence of a `base_model.marker` file inside
its folder (NOT by a hardcoded timestamp). The app reads this marker to know which
model must never be deleted on deploy.

Run this once, on the dev side, after training a new base model and before
sharing/bundling it. It is safe to re-run on an already-stamped folder: it only
(re)writes the marker and never touches the model weights.

Usage:
    python tools/mark_base.py <model_dir> --type Binary
    python tools/mark_base.py <model_dir> --type Multiclass

<model_dir> is the folder that directly contains model_final.pth.
"""

import argparse
import os
import sys

# Import the shared marker logic from ui/ (kept dependency-free for this reason).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui"))
from base_marker import BASE_MODEL_MARKER, write_base_marker


def main():
    parser = argparse.ArgumentParser(
        description="Stamp a model folder as the protected base model.")
    parser.add_argument(
        "model_dir",
        help="Path to the model folder that directly contains model_final.pth")
    parser.add_argument(
        "--type", required=True, choices=["Binary", "Multiclass"],
        help="Model type this base belongs to")
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)

    if not os.path.isdir(model_dir):
        parser.error(f"Not a directory: {model_dir}")
    if not os.path.exists(os.path.join(model_dir, "model_final.pth")):
        parser.error(
            f"No model_final.pth found in: {model_dir}\n"
            f"Point this at the folder that directly contains the trained model.")

    already_stamped = os.path.exists(os.path.join(model_dir, BASE_MODEL_MARKER))

    marker_path = write_base_marker(model_dir, args.type, source="designated")

    verb = "Re-stamped" if already_stamped else "Stamped"
    print(f"{verb} base model ({args.type}): {model_dir}")
    print(f"Wrote marker: {marker_path}")


if __name__ == "__main__":
    main()
