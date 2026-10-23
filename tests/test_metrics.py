import unittest

from wideband_recognition.metrics import calibrate_confidence, detection_stats, iou_xyxy, match_image


class MetricTests(unittest.TestCase):
    def test_iou_and_one_to_one_matching(self):
        box = [0, 0, 10, 10]
        self.assertEqual(iou_xyxy(box, box), 1.0)
        gt = [{"class": "PAM", "xyxy": box}]
        predictions = [
            {"class": "PAM", "xyxy": box, "confidence": 0.9},
            {"class": "FM", "xyxy": box, "confidence": 0.8},
        ]
        matches, missed, extras = match_image(gt, predictions, class_agnostic=True)
        self.assertEqual(len(matches), 1)
        self.assertEqual(missed, [])
        self.assertEqual(extras, [1])

    def test_class_aware_matching_keeps_wrong_class_as_error(self):
        items = [{
            "gt": [{"class": "PAM", "xyxy": [0, 0, 10, 10]}],
            "pred": [{"class": "FM", "xyxy": [0, 0, 10, 10], "confidence": 0.9}],
        }]
        stats = detection_stats(items, confidence_threshold=0.5, class_agnostic=False)
        self.assertEqual((stats["tp"], stats["fp"], stats["fn"]), (0, 1, 1))

    def test_validation_calibration_is_bounded_and_reports_target_attainment(self):
        items = [{
            "gt": [{"class": "PAM", "xyxy": [0, 0, 10, 10]}],
            "pred": [
                {"class": "PAM", "xyxy": [0, 0, 10, 10], "confidence": 0.9},
                {"class": "FM", "xyxy": [20, 20, 30, 30], "confidence": 0.1},
            ],
        }]
        selected = calibrate_confidence(items, target_precision=0.8)
        self.assertTrue(selected["attained_target"])
        self.assertEqual(selected["precision"], 1.0)
        self.assertEqual(selected["recall"], 1.0)
