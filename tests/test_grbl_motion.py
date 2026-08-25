import sys
import unittest
from pathlib import Path


UI_DIR = Path(__file__).resolve().parents[1] / "ui"
sys.path.insert(0, str(UI_DIR))

from grbl_motion import axis_word, directed_axis_value


class GrblMotionTests(unittest.TestCase):
    def test_default_autoscan_orientation_uses_expected_signs(self):
        self.assertEqual(axis_word("X", 5, inverted_feed=True), "X-5")
        self.assertEqual(axis_word("Y", 3, inverted_feed=True), "Y3")

    def test_unchecked_feed_flips_autoscan_signs(self):
        self.assertEqual(axis_word("X", 5, inverted_feed=False), "X5")
        self.assertEqual(axis_word("Y", 3, inverted_feed=False), "Y-3")

    def test_negative_scan_distance_flips_relative_to_axis_rule(self):
        self.assertEqual(directed_axis_value("X", -5, inverted_feed=True), 5)
        self.assertEqual(directed_axis_value("Y", -3, inverted_feed=False), 3)


if __name__ == "__main__":
    unittest.main()
