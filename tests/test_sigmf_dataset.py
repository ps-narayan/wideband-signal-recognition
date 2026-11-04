import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wideband_recognition.sigmf_dataset import (
    _canonical_label,
    convert_sigmf_recording,
    load_sigmf_boxes,
    write_sigmf_dataset_yaml,
)
from wideband_recognition.signals import PAPER_DETECTOR_CLASSES


class SigMFDatasetTests(unittest.TestCase):
    def test_paper_label_mapping_merges_psk_qam_and_apsk_into_pam(self):
        for name in ("2-PSK", "64-QAM", "16 APSK", "M-PSK", "BPSK", "QPSK", "PAM"):
            self.assertEqual(_canonical_label(name), "PAM")
        self.assertEqual(_canonical_label("2-CPFSK"), "2-CPFSK")
        self.assertEqual(_canonical_label("GMSK-1"), "GMSK1")
        with self.assertRaisesRegex(ValueError, "generic GMSK is ambiguous"):
            _canonical_label("GMSK")

    def test_sigmf_edges_are_converted_from_rf_hz_to_normalized_baseband(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.sigmf-meta"
            path.write_text(json.dumps({
                "global": {"core:sample_rate": 1_000_000, "core:frequency": 915_000_000},
                "captures": [],
                "annotations": [{
                    "core:sample_start": 100,
                    "core:sample_count": 800,
                    "core:freq_lower_edge": 914_900_000,
                    "core:freq_upper_edge": 915_100_000,
                    "core:label": "16-QAM",
                }],
            }), encoding="utf-8")
            boxes = load_sigmf_boxes(path, n_samples=1024)
        self.assertEqual(boxes[0].class_name, "PAM")
        self.assertEqual((boxes[0].start, boxes[0].end), (100.0, 900.0))
        self.assertAlmostEqual(boxes[0].f_low, -0.1)
        self.assertAlmostEqual(boxes[0].f_high, 0.1)

    def test_missing_sample_count_uses_file_end_and_ambiguous_gmsk_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.sigmf-meta"
            common = {"global": {"core:sample_rate": 1.0}, "captures": []}
            path.write_text(json.dumps({**common, "annotations": [{
                "core:sample_start": 10,
                "freq_low_normalized": -0.1,
                "freq_high_normalized": 0.1,
                "core:label": "FM",
            }]}), encoding="utf-8")
            boxes = load_sigmf_boxes(path, n_samples=100)
            self.assertEqual((boxes[0].start, boxes[0].end), (10.0, 100.0))
            path.write_text(json.dumps({**common, "annotations": [{
                "core:sample_start": 0,
                "core:sample_count": 100,
                "freq_low_normalized": -0.1,
                "freq_high_normalized": 0.1,
                "core:label": "GMSK",
            }]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "assign GMSK1 or GMSK2"):
                load_sigmf_boxes(path, n_samples=100)

    def test_missing_sample_count_stops_at_the_active_capture_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.sigmf-meta"
            path.write_text(json.dumps({
                "global": {"core:sample_rate": 1.0},
                "captures": [
                    {"core:sample_start": 0},
                    {"core:sample_start": 60},
                ],
                "annotations": [{
                    "core:sample_start": 10,
                    "freq_low_normalized": -0.1,
                    "freq_high_normalized": 0.1,
                    "core:label": "FM",
                }],
            }), encoding="utf-8")
            boxes = load_sigmf_boxes(path, n_samples=100)
        self.assertEqual((boxes[0].start, boxes[0].end), (10.0, 60.0))

    def test_sigmf_recording_streams_to_12_class_images_labels_and_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = np.exp(2j * np.pi * 0.15 * np.arange(1536)).astype(np.complex64)
            interleaved = np.column_stack((samples.real, samples.imag)).astype("<f4")
            data_path = root / "capture.sigmf-data"
            interleaved.tofile(data_path)
            meta_path = root / "capture.sigmf-meta"
            meta_path.write_text(json.dumps({
                "global": {
                    "core:datatype": "cf32_le",
                    "core:sample_rate": 1.0,
                    "core:dataset": data_path.name,
                },
                "captures": [],
                "annotations": [{
                    "core:sample_start": 900,
                    "core:sample_count": 300,
                    "freq_low_normalized": 0.1,
                    "freq_high_normalized": 0.2,
                    "core:label": "2-PSK",
                }],
            }), encoding="utf-8")

            report = convert_sigmf_recording(
                meta_path, root / "processed", recording_id="record-01", split="train",
                segment_samples=1024, image_size=64,
            )
            self.assertEqual(report, {
                "segments": 2, "labeled_segments": 2, "signals": 1, "samples": 1536
            })
            labels = root / "processed" / "labels" / "train"
            rows0 = (labels / "record-01-seg-00000.txt").read_text(encoding="utf-8").splitlines()
            rows1 = (labels / "record-01-seg-00001.txt").read_text(encoding="utf-8").splitlines()
            self.assertTrue(rows0[0].startswith("3 "))
            self.assertTrue(rows1[0].startswith("3 "))
            manifest = json.loads((
                root / "processed" / "metadata" / "train" / "record-01-seg-00001.json"
            ).read_text(encoding="utf-8"))
            self.assertEqual(manifest["recording_id"], "record-01")
            self.assertEqual(manifest["boxes"][0]["start"], 0.0)
            self.assertEqual(manifest["boxes"][0]["end"], 176.0)

            yaml_path = write_sigmf_dataset_yaml(root / "processed", root / "processed" / "dataset.yaml")
            self.assertIn(f"nc: {len(PAPER_DETECTOR_CLASSES)}", yaml_path.read_text(encoding="utf-8"))
            self.assertEqual(
                json.loads((root / "processed" / "class_names.json").read_text(encoding="utf-8"))["names"],
                list(PAPER_DETECTOR_CLASSES),
            )


if __name__ == "__main__":
    unittest.main()
