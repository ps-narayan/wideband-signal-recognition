import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from wideband_recognition.cli import (
    _isolated_yolo_environment,
    _frequency_output,
    _load_detector_classes,
    _read_yolo_labels,
    build_parser,
    extract_cli_frames,
    make_segments,
    read_iq_input,
    run_inference,
)
from wideband_recognition.extraction import extract_frames
from wideband_recognition.geometry import SignalBox


class CliInputTests(unittest.TestCase):
    def test_yolo_subprocess_gets_isolated_cross_platform_checkpoint_shim(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime"
            with patch.dict("os.environ", {"PYTHONPATH": "existing-path"}):
                environment = _isolated_yolo_environment(runtime)
            shim = runtime / "sitecustomize.py"
            self.assertTrue(shim.is_file())
            self.assertIn("pathlib.PosixPath = pathlib.WindowsPath", shim.read_text(encoding="utf-8"))
            self.assertEqual(environment["PYTHONPATH"].split(os.pathsep)[0], str(runtime))

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
            labels.write_text("3 0.5 0.5 0.25 0.5 0.9\n", encoding="utf-8")
            result = _read_yolo_labels(labels)
        self.assertEqual(result[0]["class"], "PAM")
        self.assertAlmostEqual(result[0]["confidence"], 0.9)
        self.assertEqual(result[0]["xyxy"], (240.0, 160.0, 400.0, 480.0))

    def test_yolo_label_pixels_follow_training_image_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            labels = Path(temporary) / "labels.txt"
            labels.write_text("3 0.5 0.5 0.25 0.5 0.9\n", encoding="utf-8")
            result = _read_yolo_labels(labels, image_size=1280)
        self.assertEqual(result[0]["xyxy"], (480.0, 320.0, 800.0, 960.0))

    def test_yolo_label_rounding_overshoot_is_clamped_at_image_edge(self):
        with tempfile.TemporaryDirectory() as temporary:
            labels = Path(temporary) / "labels.txt"
            labels.write_text("3 0.937891 0.5 0.124219 0.1 0.9\n", encoding="utf-8")
            result = _read_yolo_labels(labels, image_size=1280)
        self.assertEqual(result[0]["xyxy"][2], 1280.0)

    def test_yolo_label_rejects_materially_out_of_frame_box(self):
        with tempfile.TemporaryDirectory() as temporary:
            labels = Path(temporary) / "labels.txt"
            labels.write_text("3 0.95 0.5 0.2 0.1 0.9\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "falls outside image"):
                _read_yolo_labels(labels, image_size=1280)

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
                runtime = Path(kwargs["env"]["APPDATA"]).parent
                self.assertTrue((runtime / "appdata").is_dir())
                self.assertTrue((runtime / "ultralytics-config" / "Arial.ttf").is_file())
                self.assertTrue((runtime / "mpl-config").is_dir())
                self.assertTrue(kwargs["capture_output"])
                self.assertEqual(kwargs["encoding"], "utf-8")
                self.assertEqual(kwargs["errors"], "replace")
                self.assertEqual(command[command.index("--conf-thres") + 1], "0.25")
                project = Path(command[command.index("--project") + 1])
                name = command[command.index("--name") + 1]
                labels = project / name / "labels"
                labels.mkdir(parents=True)
                (labels / "segment_00000.txt").write_text(
                    "3 0.5 0.5 0.25 0.25 0.9\n", encoding="utf-8"
                )
                return SimpleNamespace(returncode=0, stdout="detect ok\n", stderr="")

            args = Namespace(
                input=str(input_path), output_dir=str(output_dir), yolo_dir=str(yolo_dir),
                detector_weights=str(weights), confidence_threshold=0.25, device="cpu",
                sample_rate_hz=None, raw_dtype="complex64", segment_samples=8199,
                pam_weights=None,
                detector_only=True,
            )
            with patch("wideband_recognition.cli.subprocess.check_output", return_value="f9578796b8a64889aa9e66dc06914d19f40d338e"), patch(
                "wideband_recognition.cli.subprocess.run", side_effect=fake_detect
            ):
                report = run_inference(args)

            self.assertFalse(report["seed_dependent"])
            self.assertEqual(report["samples_processed"], 8192)
            self.assertEqual(report["samples_not_processed"], 7)
            self.assertEqual(report["detections"][0]["detector_class"], "PAM")
            self.assertGreaterEqual(report["processing_runtime_seconds"]["total"], 0.0)
            self.assertEqual(len(report["detections"][0]["frequency_range"]["normalized_cycles_per_sample"]), 2)
            self.assertTrue((output_dir / "detections.json").is_file())
            self.assertTrue((output_dir / "annotated_segment_00000.png").is_file())
            saved = json.loads((output_dir / "detections.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["detections"]), 1)

    def test_default_cli_path_is_yolo_plus_pam_and_requires_pam_checkpoint(self):
        args = build_parser().parse_args([
            "capture.sigmf-meta", "--detector-weights", "detector.pt", "--yolo-dir", "yolov5"
        ])
        self.assertEqual(args.backend, "yolo")
        self.assertFalse(args.detector_only)
        self.assertIsNone(args.pam_weights)
        with self.assertRaisesRegex(ValueError, "requires --pam-weights"):
            run_inference(args)

    def test_detector_class_map_can_name_all_twelve_reference_classes(self):
        from wideband_recognition.signals import PAPER_DETECTOR_CLASSES

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "class_names.json"
            path.write_text(json.dumps({"names": list(PAPER_DETECTOR_CLASSES)}), encoding="utf-8")
            self.assertEqual(_load_detector_classes(path), PAPER_DETECTOR_CLASSES)
            labels = Path(temporary) / "labels.txt"
            labels.write_text("3 0.5 0.5 0.25 0.25 0.9\n", encoding="utf-8")
            result = _read_yolo_labels(labels, class_names=PAPER_DETECTOR_CLASSES)
        self.assertEqual(result[0]["class"], "PAM")
        self.assertEqual(PAPER_DETECTOR_CLASSES[11], "GMSK2")


if __name__ == "__main__":
    unittest.main()

