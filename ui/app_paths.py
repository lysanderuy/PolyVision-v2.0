"""
Utilities for resolving resource and configuration file paths.

Ensures PyInstaller onefile bundles can find bundled assets and that we
persist user-editable settings in a writable location.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_IS_FROZEN = getattr(sys, "frozen", False)
_BASE_PATH = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def base_path() -> Path:
    return _BASE_PATH


def resource_path(*relative_parts: str) -> str:
    """
    Return an absolute path to a bundled resource.

    Works both during development and when the app is running from a
    PyInstaller onefile bundle.
    """
    return str(_BASE_PATH.joinpath(*relative_parts))


def _user_config_dir() -> Path:
    if not _IS_FROZEN:
        return _BASE_PATH

    roaming = os.getenv("APPDATA")
    if roaming:
        config_dir = Path(roaming) / "PolyVision"
    else:
        config_dir = Path.home() / ".polyvision"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _ensure_user_file(filename: str) -> Path:
    target = _user_config_dir() / filename
    if target.exists():
        return target

    default = _BASE_PATH / filename
    if default.exists():
        target.write_bytes(default.read_bytes())
    else:
        target.touch()
    return target


def user_settings_path() -> str:
    """
    Return a writable path for user_settings.json.

    In frozen builds we copy the default template to the user's config
    directory so changes persist across launches.
    """
    return str(_ensure_user_file("user_settings.json"))


def app_storage_dir() -> Path:
    """
    Base directory for writable runtime data (databases, exports, caches).

    Mirrors the project directory during development and moves to the user's
    roaming profile when packaged via PyInstaller.
    """
    return _user_config_dir()


def ensure_storage_dir(*relative_parts: str) -> Path:
    """
    Ensure a directory exists under the storage root and return it.
    """
    path = app_storage_dir().joinpath(*relative_parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def storage_path(*relative_parts: str) -> str:
    """
    Return a writable path rooted in the storage directory.

    Parent directories are created automatically.
    """
    path = app_storage_dir().joinpath(*relative_parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _exe_dir() -> Path:
    """
    Directory that contains PolyVision.exe when frozen, or the project
    root (one level above ui/) during development.
    """
    if _IS_FROZEN:
        return Path(sys.executable).parent
    return _BASE_PATH.parent


def models_path(*relative_parts: str) -> str:
    """
    Return an absolute path inside the Models directory.

    Frozen : <dist/PolyVision>/Models/...   (sits next to the exe)
    Dev    : <project_root>/Models/...
    """
    return str(_exe_dir().joinpath("Models", *relative_parts))


def resource_exists(*relative_parts: str) -> bool:
    """
    Test whether a bundled resource exists.
    """
    return (base_path().joinpath(*relative_parts)).exists()
