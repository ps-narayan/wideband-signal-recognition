import tempfile
import unittest
from pathlib import Path

from wideband_recognition.yolo_schedule import temporary_yolov5_schedule


class TemporaryYoloScheduleTests(unittest.TestCase):
    def test_overrides_are_applied_then_original_source_is_restored(self):
        original = (
            "nbs = 64  # nominal batch size\n"
            'nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)\n'
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            train_script = Path(directory) / "train.py"
            train_script.write_bytes(original)
            with self.assertRaisesRegex(RuntimeError, "simulated training failure"):
                with temporary_yolov5_schedule(
                    train_script, nominal_batch_size=4, warmup_min_iterations=10
                ):
                    patched = train_script.read_text(encoding="utf-8")
                    self.assertIn("nbs = 4  # temporary nominal batch size", patched)
                    self.assertIn("), 10)  # temporary warmup floor", patched)
                    raise RuntimeError("simulated training failure")
            self.assertEqual(train_script.read_bytes(), original)

    def test_marker_mismatch_fails_without_mutating_source(self):
        original = b"changed pinned source\n"
        with tempfile.TemporaryDirectory() as directory:
            train_script = Path(directory) / "train.py"
            train_script.write_bytes(original)
            with self.assertRaisesRegex(RuntimeError, "marker changed"):
                with temporary_yolov5_schedule(
                    train_script, nominal_batch_size=4, warmup_min_iterations=10
                ):
                    self.fail("the source marker mismatch should stop before training")
            self.assertEqual(train_script.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
