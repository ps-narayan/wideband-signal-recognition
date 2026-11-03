# Clean max-detection-cap sensitivity

These are same-image diagnostic passes on the eight training recordings (48 signals). No validation or test recordings were used. Before every `detect.py` call, the script removed the entire prior prediction directory; each JSON records `prediction_directory_cleared_before_inference: true`. The cap is per image, confidence threshold is 0.001, and class mAP@0.5 is calculated over the predictions retained at that cap.

The 1280-pixel checkpoint and the 640-pixel enlarged-box and YOLOv5s checkpoints use the earlier pinned-default optimizer schedule. These runs isolate the inference cap only. Their failure to memorize is not a controlled comparison of model size or box size because the schedule allowed only 25 optimizer updates in 100 epochs.

| Checkpoint | Max detections / image | Predictions saved | Class-agnostic recall | Class mAP@0.5 | Prediction directory cleared |
|---|---:|---:|---:|---:|---|
| YOLOv5n, 1280 px | 100 | 800 | 6.25% | 2.60% | yes |
| YOLOv5n, 1280 px | 300 | 2,400 | 6.25% | 2.60% | yes |
| YOLOv5n, 1280 px | 1,000 | 8,000 | 10.42% | 2.84% | yes |
| YOLOv5n, 1280 px | 3,000 | 23,363 | 27.08% | 2.85% | yes |
| YOLOv5n, enlarged boxes, 640 px | 100 | 800 | 2.08% | 0.00% | yes |
| YOLOv5n, enlarged boxes, 640 px | 300 | 2,400 | 2.08% | 0.00% | yes |
| YOLOv5n, enlarged boxes, 640 px | 1,000 | 4,873 | 2.08% | 0.00% | yes |
| YOLOv5n, enlarged boxes, 640 px | 3,000 | 4,873 | 2.08% | 0.00% | yes |
| YOLOv5s, 640 px | 100 | 800 | 4.17% | 0.00% | yes |
| YOLOv5s, 640 px | 300 | 2,400 | 4.17% | 0.00% | yes |
| YOLOv5s, 640 px | 1,000 | 8,000 | 10.42% | 0.00% | yes |
| YOLOv5s, 640 px | 3,000 | 8,673 | 12.50% | 0.00% | yes |

The cap reduced recall when it truncated candidates. Raising it from 1,000 to 3,000 improved the 1280-pixel checkpoint's recall while changing mAP by only 0.01 percentage points. For the enlarged-box and YOLOv5s checkpoints, the cap did not explain their zero mAP. The complete per-class and IoU counts are in [cap_sensitivity_summary.json](../cap_sensitivity_summary.json), with one full metrics JSON per run in each `maxdet-*` directory.
