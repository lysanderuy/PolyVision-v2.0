from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = PROJECT_ROOT / "ui"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from base_marker import (
    BASE_MODEL_MARKER,
    LEGACY_BASES,
    is_base_model_dir,
    write_base_marker,
)


class IsBaseModelDirTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _make_dir(self, name):
        path = os.path.join(self.root, name)
        os.makedirs(path)
        return path

    def test_marker_present_is_base(self):
        model_dir = self._make_dir("any-folder-name")
        write_base_marker(model_dir, "Binary", source="designated")
        self.assertTrue(is_base_model_dir(model_dir, "Binary"))

    def test_legacy_name_is_base_and_self_heals(self):
        model_dir = self._make_dir(LEGACY_BASES["Binary"])
        # No marker yet: recognized via the legacy backstop...
        self.assertTrue(is_base_model_dir(model_dir, "Binary"))
        # ...and the marker is written so future checks use the primary path.
        self.assertTrue(os.path.exists(os.path.join(model_dir, BASE_MODEL_MARKER)))

    def test_legacy_name_does_not_match_other_type(self):
        # A Binary legacy timestamp checked as Multiclass must NOT count as base.
        model_dir = self._make_dir(LEGACY_BASES["Binary"])
        self.assertFalse(is_base_model_dir(model_dir, "Multiclass"))

    def test_unstamped_fresh_base_is_not_base(self):
        # A freshly trained, unstamped base reads as "not a base" until stamped.
        # This is exactly why deploy_model has its fail-safe halt.
        model_dir = self._make_dir("2026-07-12-09-30-00")
        self.assertFalse(is_base_model_dir(model_dir, "Binary"))

    def test_missing_directory_is_not_base(self):
        missing = os.path.join(self.root, "does-not-exist")
        self.assertFalse(is_base_model_dir(missing, "Binary"))


class WriteBaseMarkerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_writes_valid_json_with_expected_keys(self):
        model_dir = os.path.join(self.root, "2026-01-01-00-00-00")
        os.makedirs(model_dir)

        marker_path = write_base_marker(model_dir, "Multiclass", source="designated")

        self.assertTrue(os.path.exists(marker_path))
        with open(marker_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertEqual(payload["model_type"], "Multiclass")
        self.assertEqual(payload["source"], "designated")
        self.assertEqual(payload["original_timestamp"], "2026-01-01-00-00-00")
        self.assertIn("marked_at", payload)


if __name__ == "__main__":
    unittest.main()
