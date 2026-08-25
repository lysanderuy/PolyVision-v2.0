import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


UI_DIR = Path(__file__).resolve().parents[1] / "ui"
sys.path.insert(0, str(UI_DIR))

import settings_store


class SettingsStoreTests(unittest.TestCase):
    def test_load_settings_repairs_missing_calibration_and_grbl_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = Path(temp_dir) / "user_settings.json"
            settings_path.write_text(json.dumps({"general_features": {"model": "Multiclass"}}))

            repaired = settings_store.load_settings(str(settings_path))

            self.assertEqual(repaired["general_features"]["model"], "Multiclass")
            self.assertIn("calibration", repaired["image_settings"])
            self.assertIn("steps_per_mm", repaired["grbl_settings"])
            self.assertIn("max_feedrate", repaired["grbl_settings"])
            self.assertIn("area_scan", repaired["grbl_settings"])

            persisted = json.loads(settings_path.read_text())
            self.assertEqual(persisted, repaired)

    def test_save_settings_is_atomic_and_normalizes_types(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = Path(temp_dir) / "user_settings.json"

            saved = settings_store.save_settings(
                {
                    "image_settings": {"calibration": "0.01"},
                    "grbl_settings": {
                        "steps_per_mm": "2.5",
                        "max_feedrate": "1200",
                        "area_scan": "false",
                    },
                    "general_features": {"model": "Binary", "sound": "true"},
                },
                str(settings_path),
            )

            self.assertEqual(saved["image_settings"]["calibration"], 0.01)
            self.assertEqual(saved["grbl_settings"]["steps_per_mm"], 2.5)
            self.assertEqual(saved["grbl_settings"]["max_feedrate"], 1200.0)
            self.assertFalse(saved["grbl_settings"]["area_scan"])
            self.assertTrue(saved["general_features"]["sound"])
            self.assertFalse(any(name.endswith(".tmp") for name in os.listdir(temp_dir)))

    def test_invalid_json_falls_back_to_defaults_and_repairs_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = Path(temp_dir) / "user_settings.json"
            settings_path.write_text("{")

            repaired = settings_store.load_settings(str(settings_path))

            self.assertIn("image_settings", repaired)
            self.assertEqual(json.loads(settings_path.read_text()), repaired)


if __name__ == "__main__":
    unittest.main()
