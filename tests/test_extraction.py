import tempfile
import unittest
from pathlib import Path

import numpy as np

from wideband_recognition.extraction import (
    design_extraction_filter,
    extract_frames,
    measure_filter_response,
    perturb_box,
    read_complex_iq,
)
from wideband_recognition.geometry import SignalBox
from wideband_recognition.signals import SignalSpec, generate_recording


class ExtractionTests(unittest.TestCase):
    def test_perturbed_boxes_stay_valid_and_within_edge_limits(self):
        box = SignalBox(20_000, 100_000, -0.02, 0.02, "PAM")
        for seed in range(100):
            moved = perturb_box(box, 200_000, np.random.default_rng(seed))
            self.assertLess(moved.start, moved.end)
            self.assertLess(moved.f_low, moved.f_high)
            self.assertLessEqual(abs(moved.start - box.start), box.duration * 0.05)
            self.assertLessEqual(abs(moved.f_low - box.f_low), box.bandwidth * 0.05)

    def test_minimum_bandwidth_filter_meets_response_limits(self):
        response = design_extraction_filter(0.008)
        ripple, stop_db = measure_filter_response(response)
        self.assertLessEqual(ripple, 0.1)
        self.assertLessEqual(stop_db, -60.0)
        self.assertEqual(response.group_delay, 555)

    def test_seed_and_iq_file_paths_produce_identical_frames(self):
        specs = [SignalSpec("2-PSK", 0, 16_384, 0.1, 0.04, carrier_phase=0.7)]
        generated = generate_recording(specs, 16_384, seed=17, global_snr_db=10)
        box = SignalBox(0, 16_384, 0.08, 0.12, "PAM")
        replayed = generate_recording(specs, 16_384, seed=17, global_snr_db=10)
        direct = extract_frames(generated.iq, box, n_frames=2, seed=9)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "sample.npy"
            np.save(path, replayed.iq, allow_pickle=False)
            loaded = read_complex_iq(path)
            from_file = extract_frames(loaded, box, n_frames=2, seed=9)
        np.testing.assert_array_equal(direct, from_file)

    def test_unresolved_box_is_not_extracted(self):
        box = SignalBox(0, 2048, -0.02, 0.02, "PAM")
        with self.assertRaises(ValueError):
            extract_frames(np.ones(2048, dtype=np.complex64), box, unresolved=True)
