import unittest

from wideband_recognition.geometry import SignalBox, box_to_pixels, pixels_to_box, to_yolo


class GeometryTests(unittest.TestCase):
    def test_time_frequency_round_trip_uses_covered_samples(self):
        covered = 1953 * 512
        box = SignalBox(12345, 765432, -0.123, 0.077, "PAM")
        pixels = box_to_pixels(box, covered)
        recovered = pixels_to_box(pixels, covered, class_name="PAM")
        self.assertAlmostEqual(recovered.start, box.start, places=9)
        self.assertAlmostEqual(recovered.end, box.end, places=9)
        self.assertAlmostEqual(recovered.f_low, box.f_low, places=12)
        self.assertAlmostEqual(recovered.f_high, box.f_high, places=12)
        self.assertEqual(1_000_000 - covered, 64)

    def test_yolo_values_are_normalized_and_positive(self):
        box = SignalBox(0, 8192, -0.004, 0.004, "PAM")
        label = to_yolo(box, 5, 999936)
        self.assertEqual(label[0], 5)
        self.assertTrue(all(0 <= value <= 1 for value in label[1:]))
        self.assertGreater(label[3], 0)
        self.assertGreater(label[4], 0)

    def test_invalid_box_is_rejected(self):
        with self.assertRaises(ValueError):
            box_to_pixels(SignalBox(5, 4, -0.1, 0.1), 100)
