# base_marker.py
#
# Single source of truth for base-model identity. The base model is the
# protected, never-deleted model. It is identified by the presence of a marker
# file inside its folder, NOT by a hardcoded timestamp, so a freshly trained base
# (which has a different timestamp) is still recognized.
#
# This module intentionally has no heavy dependencies (no PyQt/cv2/detectron2) so
# it can be imported both by the app (ui/retrain.py) and by the standalone dev
# tool (tools/mark_base.py).

import os
import json
from datetime import datetime

BASE_MODEL_MARKER = "base_model.marker"

# Frozen backstop: the original v1 base model folder names. Kept ONLY so installs
# that predate the marker convention still recognize their base (and self-heal a
# marker). Do NOT add entries here -- new bases are stamped by tools/mark_base.py.
LEGACY_BASES = {
    "Binary": "2025-10-01-03-07-35",
    "Multiclass": "2025-10-01-03-54-34",
}


def write_base_marker(model_dir, model_type, source):
    """Write the base-model marker into model_dir.

    Existence of the file is authoritative; the JSON payload is advisory metadata
    only and is never parsed to make a protection decision.
    """
    marker_path = os.path.join(model_dir, BASE_MODEL_MARKER)
    payload = {
        "model_type": model_type,
        "source": source,
        "original_timestamp": os.path.basename(model_dir),
        "marked_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(marker_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return marker_path


def is_base_model_dir(model_dir, model_type):
    """Single source of truth for 'is this folder a protected base model?'.

    Resolution order, existence-authoritative (never parse-to-decide):
      1. base_model.marker present -> base.
      2. folder name matches the frozen LEGACY_BASES backstop -> base, and
         self-heal by writing the marker so future checks use the primary path.
      3. otherwise -> not a base.
    """
    if not os.path.isdir(model_dir):
        return False
    if os.path.exists(os.path.join(model_dir, BASE_MODEL_MARKER)):
        return True
    if os.path.basename(model_dir) == LEGACY_BASES.get(model_type):
        try:
            write_base_marker(model_dir, model_type, source="migrated-legacy")
        except OSError:
            pass  # Protection still holds via the legacy name for this run.
        return True
    return False
