import unittest

from scripts.publish_reduced_results import normalize_summary, normalize_zero_detection_precision


class ReportingNormalizationTests(unittest.TestCase):
    def test_zero_detection_precision_and_ci_are_unmeasured(self):
        metrics = {
            "main_test": {
                "end_to_end": {
                    "predictions": 0,
                    "precision": 0.0,
                    "precision_ci95": [0.0, 0.0],
                    "by_global_snr_with_recording_bootstrap": {
                        "[0,5)": {
                            "precision": {"estimate": 0.0, "ci95": [0.0, 0.0]},
                        }
                    },
                },
                "yolo": {
                    "by_global_snr_with_recording_bootstrap": {
                        "[0,5)": {"precision": {"estimate": None, "ci95": [None, None]}},
                    }
                },
            },
            "shifted_test": {
                "end_to_end": {
                    "predictions": 0,
                    "precision": 0.0,
                    "precision_ci95": [0.0, 0.0],
                },
                "end_to_end_by_global_snr_with_recording_bootstrap": {
                    "[0,5)": {
                        "precision": {"estimate": 0.0, "ci95": [0.0, 0.0]},
                    }
                },
                "yolo_by_global_snr_with_recording_bootstrap": {
                    "[0,5)": {"precision": {"estimate": None, "ci95": [None, None]}},
                },
            },
        }
        normalized = normalize_zero_detection_precision(metrics)
        for split in ("main_test", "shifted_test"):
            e2e = normalized[split]["end_to_end"]
            self.assertIsNone(e2e["precision"])
            self.assertEqual(e2e["precision_ci95"], [None, None])
        self.assertIsNone(
            normalized["main_test"]["end_to_end"]["by_global_snr_with_recording_bootstrap"]["[0,5)"]["precision"]["estimate"]
        )
        self.assertEqual(
            normalized["shifted_test"]["end_to_end_by_global_snr_with_recording_bootstrap"]["[0,5)"]["precision"]["ci95"],
            [None, None],
        )

    def test_summary_marks_zero_detection_precision_and_pam_chance(self):
        metrics = {
            "main_test": {
                "end_to_end": {"predictions": 0},
                "yolo": {
                    "by_global_snr_with_recording_bootstrap": {
                        "[0,5)": {"precision": {"estimate": None}},
                    }
                },
            },
            "shifted_test": {"end_to_end": {"predictions": 0}},
        }
        summary = "\n".join([
            "Oracle PAM accuracy at global SNR [0,5) dB: 0.0000; CI95=[0.0000, 0.0000]",
            "End-to-end precision: 0.0000; CI95=[0.0000, 0.0000]",
            "End-to-end precision at global SNR [0,5) dB: 0.0000; CI95=[0.0000, 0.0000]",
            "Shifted end-to-end precision: 0.0000; CI95=[0.0000, 0.0000]",
            "YOLO precision: n/a; CI95=[n/a, n/a]",
            "AM-DSB: P=n/a CI95=[None, None]",
            "GMSK2: P=not measured CI95=[None, None]; R=0.0000 CI95=[0.0000, 0.0000]; IoU=not measured CI95=[None, None]",
            "All-signal classifier accuracy for Others: 0.1250; CI95=[0.0000, 0.2500]",
        ])
        result = normalize_summary(summary, metrics)
        self.assertIn("Oracle PAM accuracy at global SNR [0,5) dB (PAM chance level 1/7 = 14.29%)", result)
        self.assertIn("End-to-end precision: not measured; CI95=[not measured, not measured]", result)
        self.assertIn("End-to-end precision at global SNR [0,5) dB: not measured; CI95=[not measured, not measured]", result)
        self.assertIn("Shifted end-to-end precision: not measured; CI95=[not measured, not measured]", result)
        self.assertIn("YOLO precision: not measured; CI95=[not measured, not measured]", result)
        self.assertIn("P=not measured CI95=[not measured, not measured]", result)
        self.assertIn("IoU=not measured CI95=[not measured, not measured]", result)
        self.assertIn("All-signal classifier accuracy for Others (PAM chance level 1/7 = 14.29%)", result)

    def test_measured_zero_precision_with_predictions_is_preserved(self):
        metrics = {
            "main_test": {
                "end_to_end": {"predictions": 3, "precision": 0.0, "precision_ci95": [0.0, 0.0]},
                "yolo": {},
            },
            "shifted_test": {},
        }
        normalize_zero_detection_precision(metrics)
        self.assertEqual(metrics["main_test"]["end_to_end"]["precision"], 0.0)
        self.assertEqual(metrics["main_test"]["end_to_end"]["precision_ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
