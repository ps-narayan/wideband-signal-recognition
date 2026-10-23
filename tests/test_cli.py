import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from wideband_recognition.cli import (
    _frequency_output,
    _read_yolo_labels,
    extract_cli_frames,
    make_segments,
    read_iq_input,
    run_inference,
)
from wideband_recognition.extraction import extract_frames
from wideband_recognition.geometry import SignalBox


class CliInputTests(unittest.TestCase):
    def test_sigmf_cf32_metadata_supplies_sample_rate_and_center(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            iq = np.array([0.25 + 0.5j, -0.25 - 0.5j], dtype=np.complex64)
            data_path = root / "capture.sigmf-data"
            iq.tofile(data_path)
            metadata = {
                "global": {
                    "core:datatype": "cf32_le",
                    "core:sample_rate": 2_000_000,
                    "core:frequency": 915_000_000,
                    "core:dataset": data_path.name,
                },
                "captures": [],
                "annotations": [],
            }
            meta_path = root / "capture.sigmf-meta"
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")
            actual, rate, info = read_iq_input(meta_path)
            np.testing.assert_array_equal(actual, iq)
            self.assertEqual(rate, 2_000_000)
            self.assertEqual(info["center_frequency_hz"], 915_000_000)

    def test_sigmf_rejects_unimplemented_datatype(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "capture.sigmf-data"
            data_path.write_bytes(b"\x00\x00")
            meta_path = root / "capture.sigmf-meta"
            meta_path.write_text(json.dumps({
                "global": {"core:datatype": "ri16_le"},
                "captures": [],
                "annotations": [],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported SigMF"):
                read_iq_input(meta_path)

    def test_sigmf_rejects_multiple_channels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "capture.sigmf-data"
            np.array([0.0, 0.0], dtype=np.float32).tofile(data_path)
            meta_path = root / "capture.sigmf-meta"
            meta_path.write_text(json.dumps({
                "global": {"core:datatype": "cf32_le", "core:num_channels": 2},
                "captures": [],
                "annotations": [],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "multi-channel"):
                read_iq_input(meta_path)
    def test_deployment_and_training_evaluation_extraction_match_without_seed(self):
        rng = np.random.default_rng(5)
        iq = (rng.standard_normal(16_384) + 1j * rng.standard_normal(16_384)).astype(np.complex64)
        box = SignalBox(0, 16_384, 0.08, 0.12, "PAM")
        expected = extract_frames(iq, box, n_frames=2, jitter=False)
        actual = extract_cli_frames(iq, box)
        np.testing.assert_array_equal(actual, expected)

    def test_file_input_path_feeds_the_same_deployment_extraction(self):
        rng = np.random.default_rng(11)
        iq = (rng.standard_normal(16_384) + 1j * rng.standard_normal(16_384)).astype(np.complex64)
        box = SignalBox(0, 16_384, 0.08, 0.12, "PAM")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.npy"
            np.save(path, iq, allow_pickle=False)
            from_file, _, _ = read_iq_input(path)
        np.testing.assert_array_equal(extract_cli_frames(from_file, box), extract_frames(iq, box, jitter=False))

    def test_yolo_labels_are_mapped_to_named_detector_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            labels = Path(temporary) / "labels.txt"
            labels.write_text("5 0.5 0.5 0.25 0.5 0.9\n", encoding="utf-8")
            result = _read_yolo_labels(labels)
        self.assertEqual(result[0]["class"], "PAM")
        self.assertAlmostEqual(result[0]["confidence"], 0.9)
        self.assertEqual(result[0]["xyxy"], (240.0, 160.0, 400.0, 480.0))

    def test_hz_output_is_baseband_offset_and_absolute_when_center_is_known(self):
        box = SignalBox(0, 1024, -0.1, 0.2)
        result = _frequency_output(box, 1_000_000, 915_000_000)
        self.assertEqual(result["normalized_cycles_per_sample"], [-0.1, 0.2])
        self.assertEqual(result["offset_hz_from_baseband_center"], [-100_000.0, 200_000.0])
        self.assertEqual(result["absolute_hz"], [914_900_000.0, 915_200_000.0])

    def test_segmenting_skips_only_short_tail(self):
        iq = np.ones(1000, dtype=np.complex64)
        result = make_segments(iq, 600)
        self.assertEqual([start for start, _ in result], [0])
        self.assertEqual(result[0][1].size, 600)
        with self.assertRaises(ValueError):
            make_segments(iq[:511], 512)

    def test_cli_mocked_detector_writes_json_and_annotated_image(self):
        rng = np.random.default_rng(3)
        iq = (rng.standard_normal(8199) + 1j * rng.standard_normal(8199)).astype(np.complex64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "capture.npy"
            np.save(input_path, iq, allow_pickle=False)
            weights = root / "detector.pt"
            weights.write_bytes(b"mock")
            yolo_dir = root / "yolov5"
            yolo_dir.mkdir()
            (yolo_dir / "detect.py").write_text("# pinned CLI placeholder", encoding="utf-8")
            output_dir = root / "outputs"

            def fake_detect(command, **kwargs):
                project = Path(command[command.index("--project") + 1])
                name = command[command.index("--name") + 1]
                labels = project / name / "labels"
                labels.mkdir(parents=True)
                (labels / "segment_00000.txt").write_text(
                    "5 0.5 0.5 0.25 0.25 0.9\n", encoding="utf-8"
                )

            args = Namespace(
                input=str(input_path), output_dir=str(output_dir), yolo_dir=str(yolo_dir),
                detector_weights=str(weights), confidence_threshold=0.25, device="cpu",
                sample_rate_hz=None, raw_dtype="complex64", segment_samples=8199,
                pam_weights=None,
            )
            with patch("wideband_recognition.cli.subprocess.check_output", return_value="f9578796b8a64889aa9e66dc06914d19f40d338e"), patch(
                "wideband_recognition.cli.subprocess.run", side_effect=fake_detect
            ):
                report = run_inference(args)

            self.assertFalse(report["seed_dependent"])
            self.assertEqual(report["samples_processed"], 8192)
            self.assertEqual(report["samples_not_processed"], 7)
            self.assertEqual(report["detections"][0]["detector_class"], "PAM")
            self.assertEqual(len(report["detections"][0]["frequency_range"]["normalized_cycles_per_sample"]), 2)
            self.assertTrue((output_dir / "detections.json").is_file())
            self.assertTrue((output_dir / "annotated_segment_00000.png").is_file())
            saved = json.loads((output_dir / "detections.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["detections"]), 1)


if __name__ == "__main__":
    unittest.main()

