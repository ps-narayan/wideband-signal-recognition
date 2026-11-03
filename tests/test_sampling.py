import unittest

import numpy as np

from wideband_recognition.sampling import cap_frames_per_class


class FrameSamplingTests(unittest.TestCase):
    def test_class_cap_is_balanced_and_reproducible(self):
        frames = np.arange(30).reshape(10, 3)
        labels = np.array([0] * 6 + [1] * 4)
        rows = [{"index": i} for i in range(len(labels))]
        x, y, kept = cap_frames_per_class(frames, labels, rows, 3, seed=17)
        x2, y2, kept2 = cap_frames_per_class(frames, labels, rows, 3, seed=17)
        self.assertEqual(len(x), 6)
        self.assertEqual(np.bincount(y).tolist(), [3, 3])
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_array_equal(y, y2)
        self.assertEqual(kept, kept2)

    def test_none_keeps_every_frame(self):
        frames = np.arange(6).reshape(3, 2)
        labels = np.array([0, 1, 1])
        rows = [{"i": i} for i in range(3)]
        x, y, kept = cap_frames_per_class(frames, labels, rows, None)
        np.testing.assert_array_equal(x, frames)
        np.testing.assert_array_equal(y, labels)
        self.assertEqual(kept, rows)

    def test_invalid_cap_is_rejected(self):
        with self.assertRaises(ValueError):
            cap_frames_per_class(np.zeros((1, 2)), np.array([0]), [{}], 0)
