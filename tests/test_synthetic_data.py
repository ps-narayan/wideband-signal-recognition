import unittest

import numpy as np

from wideband_recognition.dataset import grouped_split
from wideband_recognition.geometry import SignalBox
from wideband_recognition.layout import pair_remains_resolved_after_edge_jitter, unresolved_pairs
from wideband_recognition.extraction import perturb_box
from wideband_recognition.synthetic_data import sample_specs


class SyntheticDataTests(unittest.TestCase):
    def test_layouts_are_replayable_and_guard_safe(self):
        first, snr1 = sample_specs(451, profile="train")
        second, snr2 = sample_specs(451, profile="train")
        self.assertEqual(first, second)
        self.assertEqual(snr1, snr2)
        boxes = [
            SignalBox(item.start, item.end, item.f_low, item.f_high, item.detector_class)
            for item in first
        ]
        self.assertEqual(unresolved_pairs(boxes), [])
        self.assertEqual({item.detector_class for item in first}, {"AM-DSB", "FM", "OFDM", "2-FSK", "OOK", "PAM"})
        self.assertTrue(all(item.end <= 999_936 for item in first))
        self.assertTrue(all(
            pair_remains_resolved_after_edge_jitter(boxes[i], boxes[j])
            for i in range(len(boxes)) for j in range(i + 1, len(boxes))
        ))
        for jitter_seed in range(20):
            rng = np.random.default_rng(jitter_seed)
            perturbed = [perturb_box(box, 999_936, rng) for box in boxes]
            self.assertEqual(unresolved_pairs(perturbed), [])

    def test_shifted_profile_changes_width_and_snr_ranges(self):
        train, train_snr = sample_specs(33, profile="train")
        shift, shift_snr = sample_specs(33, profile="shifted")
        self.assertTrue(all(0.008 <= item.bandwidth <= 0.08 for item in train))
        self.assertTrue(all(0.02 <= item.bandwidth <= 0.12 for item in shift))
        self.assertTrue(-5 <= train_snr <= 20)
        self.assertTrue(-5 <= shift_snr <= 5)

    def test_recordings_never_cross_splits(self):
        split = grouped_split()
        self.assertEqual(len(split), 90)
        self.assertEqual({split[f"main-{i:03d}"] for i in range(90)}.issubset({"train", "val", "test"}), True)
        self.assertEqual(sum(value == "train" for value in split.values()), 60)
        self.assertEqual(sum(value == "val" for value in split.values()), 10)
        self.assertEqual(sum(value == "test" for value in split.values()), 20)
