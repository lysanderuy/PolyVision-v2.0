from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app_paths import resource_path, user_settings_path


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _deep_merge(defaults: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(defaults)
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def default_settings() -> dict[str, Any]:
    return _load_json(resource_path("default_settings.json"))


def normalize_settings(settings: dict[str, Any]) -> dict[str, Any]:
    defaults = default_settings()
    normalized = _deep_merge(defaults, settings)

    image = normalized.setdefault("image_settings", {})
    default_image = defaults.get("image_settings", {})
    image["image_quality"] = image.get("image_quality") or default_image.get("image_quality", "Medium")
    image["image_sharpness"] = _as_int(
        image.get("image_sharpness"),
        _as_int(default_image.get("image_sharpness"), 1),
    )
    image["image_saturation"] = _as_int(
        image.get("image_saturation"),
        _as_int(default_image.get("image_saturation"), 0),
    )
    image["calibration"] = _as_float(
        image.get("calibration"),
        _as_float(default_image.get("calibration"), 0.0084),
    )

    grbl = normalized.setdefault("grbl_settings", {})
    default_grbl = defaults.get("grbl_settings", {})
    grbl["steps_per_mm"] = _as_float(
        grbl.get("steps_per_mm"),
        _as_float(default_grbl.get("steps_per_mm"), 1.0),
    )
    grbl["max_feedrate"] = _as_float(
        grbl.get("max_feedrate"),
        _as_float(default_grbl.get("max_feedrate"), 1000.0),
    )
    grbl["area_scan"] = _as_bool(
        grbl.get("area_scan"),
        _as_bool(default_grbl.get("area_scan"), True),
    )

    general = normalized.setdefault("general_features", {})
    default_general = defaults.get("general_features", {})
    if general.get("model") not in {"Binary", "Multiclass"}:
        general["model"] = default_general.get("model", "Binary")
    general["sound"] = _as_bool(
        general.get("sound"),
        _as_bool(default_general.get("sound"), False),
    )

    return normalized


def save_settings(settings: dict[str, Any], path: str | None = None) -> dict[str, Any]:
    target = Path(path or user_settings_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_settings(settings)

    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(target.parent),
            delete=False,
            prefix=f".{target.name}.",
            suffix=".tmp",
        ) as temp_file:
            temp_name = temp_file.name
            json.dump(normalized, temp_file, indent=4)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, target)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)

    return normalized


def load_settings(path: str | None = None, *, repair: bool = True) -> dict[str, Any]:
    settings_path = path or user_settings_path()
    try:
        raw_settings = _load_json(settings_path)
    except (OSError, json.JSONDecodeError):
        raw_settings = {}

    normalized = normalize_settings(raw_settings)
    if repair and normalized != raw_settings:
        save_settings(normalized, settings_path)
    return normalized
