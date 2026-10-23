import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wideband_recognition.geometry import SignalBox
from wideband_recognition.layout import unresolved_pairs
from wideband_recognition.signals import (
    SignalSpec,
    generate_recording,
    make_visibility_specs,
    save_generator_metadata,
)


class SignalTests(unittest.TestCase):
    def test_seed_replays_iq_exactly(self):
        specs = make_visibility_specs(seed=11, n_samples=500_000)
        first = generate_recording(specs, 500_000, seed=12, global_snr_db=-5)
        second = generate_recording(specs, 500_000, seed=12, global_snr_db=-5)
        np.testing.assert_array_equal(first.iq, second.iq)
        self.assertAlmostEqual(first.measured_global_snr_db, -5.0, delta=0.1)

    def test_visibility_layout_has_minimum_signal_and_no_unresolved_pairs(self):
        specs = make_visibility_specs(seed=2)
        self.assertEqual(min(s.bandwidth for s in specs), 0.008)
        self.assertEqual(min(s.duration for s in specs), 8192)
        boxes = [SignalBox(s.start, s.end, s.f_low, s.f_high, s.detector_class) for s in specs]
        self.assertEqual(unresolved_pairs(boxes), [])
        pam = next(s for s in specs if s.detector_class == "PAM")
        self.assertGreaterEqual(pam.rolloff, 0.2)
        self.assertLessEqual(pam.rolloff, 0.5)

    def test_close_layout_is_rejected_unless_stress_flag_is_set(self):
        specs = [
            SignalSpec("2-PSK", 0, 8192, 0.0, 0.02),
            SignalSpec("4-PSK", 0, 8192, 0.005, 0.02),
        ]
        with self.assertRaises(ValueError):
            generate_recording(specs, 16384, seed=1, global_snr_db=-5)


    def test_metadata_saves_seed_parameters_and_unresolved_status(self):
        specs = make_visibility_specs(seed=3, n_samples=500_000)
        generated = generate_recording(specs, 500_000, seed=4, global_snr_db=-5)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "segment.json"
            save_generator_metadata(path, specs, 500_000, 4, -5, generated)
            metadata = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["seed"], 4)
        self.assertEqual(metadata["n_samples"], 500_000)
        self.assertEqual(len(metadata["signals"]), 6)
        self.assertTrue(all(status == "resolved" for status in metadata["resolution_status"]))
