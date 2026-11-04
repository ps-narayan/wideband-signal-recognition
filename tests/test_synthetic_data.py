import unittest
from collections import Counter

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
        from wideband_recognition.signals import DETECTOR_CLASSES
        self.assertEqual({item.detector_class for item in first}, set(DETECTOR_CLASSES))
        self.assertEqual(len(first), 12)
        self.assertEqual(len({item.modulation for item in first if item.detector_class == "PAM"}), 1)
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
        self.assertTrue(all(0.025 <= item.bandwidth <= 0.04 for item in train))
        self.assertTrue(all(0.03 <= item.bandwidth <= 0.05 for item in shift))
        self.assertTrue(-5 <= train_snr <= 20)
        self.assertTrue(-5 <= shift_snr <= 5)

    def test_larger_box_constraints_are_enforced_and_replayable(self):
        first, snr1 = sample_specs(2026, profile="train", min_bandwidth=0.03, min_duration=120_000)
        second, snr2 = sample_specs(2026, profile="train", min_bandwidth=0.03, min_duration=120_000)
        self.assertEqual(first, second)
        self.assertEqual(snr1, snr2)
        self.assertTrue(all(item.bandwidth >= 0.03 for item in first))
        self.assertTrue(all(item.duration >= 120_000 for item in first))

    def test_invalid_generator_minimums_are_rejected(self):
        with self.assertRaises(ValueError):
            sample_specs(1, min_bandwidth=0.081)
        with self.assertRaises(ValueError):
            sample_specs(1, min_duration=250001)

    def test_recordings_never_cross_splits(self):
        split = grouped_split()
        self.assertEqual(len(split), 90)
        self.assertEqual({split[f"main-{i:03d}"] for i in range(90)}.issubset({"train", "val", "test"}), True)
        self.assertEqual(sum(value == "train" for value in split.values()), 60)
        self.assertEqual(sum(value == "val" for value in split.values()), 10)
        self.assertEqual(sum(value == "test" for value in split.values()), 20)

    def test_main_manifest_has_frozen_snr_strata_and_balanced_pam_subtypes(self):
        from scripts.run_milestone2 import _main_manifest

        rows, split = _main_manifest(
            120, 1, 2026, train_fraction=0.4, validation_fraction=0.1,
            test_snr_strata=(
                {"global_snr_db": 0.5, "recordings": 30},
                {"global_snr_db": 15.0, "recordings": 30},
            ),
        )
        self.assertEqual(Counter(split.values()), {"train": 48, "val": 12, "test": 60})
        by_split = {name: {row["recording_id"] for row in rows if row["split"] == name}
                    for name in ("train", "val", "test")}
        self.assertFalse(by_split["train"] & by_split["val"])
        self.assertFalse(by_split["train"] & by_split["test"])
        self.assertFalse(by_split["val"] & by_split["test"])
        test = [row for row in rows if row["split"] == "test"]
        low = [row for row in test if row["global_snr_db_override"] == 0.5]
        high = [row for row in test if row["global_snr_db_override"] == 15.0]
        self.assertEqual((len(low), len(high)), (30, 30))
        self.assertEqual(Counter(row["pam_subtype_index"] for row in low), Counter({i: 5 for i in range(6)}))
        self.assertEqual(Counter(row["pam_subtype_index"] for row in high), Counter({i: 5 for i in range(6)}))
