import unittest

from wideband_recognition.anchor_preflight import best_possible_recall


class AnchorPreflightTests(unittest.TestCase):
    def test_pinned_yolov5n_default_small_anchor_fits_minimum_raster_box(self):
        minimum_width = 8192 / 999_936 * 640
        minimum_height = 0.008 * 640
        self.assertEqual(best_possible_recall((minimum_width, minimum_height)), 1.0)
