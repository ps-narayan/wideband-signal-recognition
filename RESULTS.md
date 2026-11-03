# Reduced CPU Run Results

Run label: reduced run: CPU, small data, not comparable to target
Comparison: reduced run: CPU, small data, not comparable to target. Target statuses are not comparable; this run does not establish whether the portfolio targets are met or missed.
Device: cpu
YOLOv5 source commit: f9578796b8a64889aa9e66dc06914d19f40d338e
Detector training status: completed; checkpoint: best.pt; requested epochs: 135.
PAM training status: completed; epochs completed: 23 of 125.
Recording support: main test 4 recordings / 4 segments; shifted test 4 recordings / 4 segments.
Recording-ID leakage audit: train=12, validation=2, test=4; main splits are pairwise disjoint; shifted test IDs are separate. See [split_audit.json](results/reduced_cpu/split_audit.json) for IDs.
SNR definition: 10*log10(total clean signal power / total added noise power over the full I/Q segment).
Confidence intervals use recording-level bootstrap resampling. Cells with fewer than 30 signals are marked LOW-SUPPORT.

## Acceptance targets

Measured values are shown below. The reduced protocol makes performance targets not comparable to the full-scale acceptance criteria. The owner-machine timing remains not measured.

| Target | Threshold | Measured | Recording-level 95% CI | Support | Status |
|---|---:|---:|---:|---|---|
| Detector recall, global SNR >= 5 dB | >= 80% | 0.00% | [0.00%, 0.00%] | 3 recordings / 18 signals; LOW-SUPPORT | not comparable |
| Mean IoU, global SNR >= 5 dB | >= 0.75 | not measured | [not measured, not measured] | 3 recordings / 18 signals; LOW-SUPPORT | not comparable |
| Class mAP@0.5, global SNR >= 5 dB | >= 60% | 20.50% | [18.60%, 40.39%] | 3 recordings / 18 signals; LOW-SUPPORT | not comparable |
| PAM oracle-box accuracy (chance 1/7 = 14.29%), global SNR >= 10 dB | >= 80% | 0.00% | [0.00%, 0.00%] | 3 recordings / 3 signals; LOW-SUPPORT | not comparable |
| PAM oracle-box accuracy (chance 1/7 = 14.29%), global SNR [0, 1) dB | > 50% | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT | not comparable |
| Processing time per 1M samples on owner's machine | < 5 seconds | not measured | not applicable | not applicable | not comparable |

Validation-only operating-point calibration:
YOLO: target precision=80.00%; attained target=True; threshold=0.313669; validation precision=100.00%.
Energy baseline: target precision=80.00%; attained target=True; mask threshold=1.5; confidence threshold=3.0; validation precision=81.82%.

## Main recording-disjoint test

### Detector and geometry baseline

| Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| YOLOv5n: geometry precision | 100.00% | [100.00%, 100.00%] | 4 recordings / 24 signals; LOW-SUPPORT |
| YOLOv5n: class-agnostic recall | 4.17% | [0.00%, 12.50%] | 4 recordings / 24 signals; LOW-SUPPORT |
| YOLOv5n: mean IoU of true positives | 62.06% | [62.06%, 62.06%] | 4 recordings / 24 signals; LOW-SUPPORT |
| Energy baseline: geometry precision | 50.00% | [21.43%, 83.33%] | 4 recordings / 24 signals; LOW-SUPPORT |
| Energy baseline: class-agnostic recall | 54.17% | [25.00%, 83.33%] | 4 recordings / 24 signals; LOW-SUPPORT |
| Energy baseline: mean IoU of true positives | 69.24% | [63.36%, 71.85%] | 4 recordings / 24 signals; LOW-SUPPORT |
| YOLOv5n class mAP@0.5, all SNR | 19.14% | [18.61%, 32.73%] | 4 recordings / 24 signals; LOW-SUPPORT |
| YOLOv5n class mAP@0.5, global SNR >= 5 dB | 20.50% | [18.60%, 40.39%] | 3 recordings / 18 signals; LOW-SUPPORT |

### Per-class detector metrics

| Class metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| AM-DSB: precision | 100.00% | [100.00%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| AM-DSB: recall | 25.00% | [0.00%, 75.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| AM-DSB: mean IoU | 62.06% | [62.06%, 62.06%] | 4 recordings / 4 signals; LOW-SUPPORT |
| AM-DSB: AP@0.5 | 45.28% | [11.11%, 77.78%] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM: AP@0.5 | 3.84% | [0.00%, 28.12%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM: AP@0.5 | 23.53% | [22.22%, 28.57%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK: AP@0.5 | 16.67% | [0.00%, 85.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK: AP@0.5 | 3.00% | [0.48%, 8.40%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM: AP@0.5 | 22.50% | [5.00%, 50.00%] | 4 recordings / 4 signals; LOW-SUPPORT |

### Main test by global SNR

| Global SNR bin | YOLO recall | 95% CI | Support | YOLO mean IoU | 95% CI | Support | PAM oracle accuracy (chance 1/7 = 14.29%) | 95% CI | Support |
|---|---:|---|---|---:|---|---|---:|---|---|
| [0,5) dB | 16.67% | [16.67%, 16.67%] | 1 recordings / 6 signals; LOW-SUPPORT | 62.06% | [62.06%, 62.06%] | 1 recordings / 6 signals; LOW-SUPPORT | 0.00% (chance 1/7 = 14.29%) | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| [10,15) dB | 0.00% | [0.00%, 0.00%] | 1 recordings / 6 signals; LOW-SUPPORT | not measured | [not measured, not measured] | 1 recordings / 6 signals; LOW-SUPPORT | 0.00% (chance 1/7 = 14.29%) | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| [15,20) dB | 0.00% | [0.00%, 0.00%] | 2 recordings / 12 signals; LOW-SUPPORT | not measured | [not measured, not measured] | 2 recordings / 12 signals; LOW-SUPPORT | 0.00% (chance 1/7 = 14.29%) | [0.00%, 0.00%] | 2 recordings / 2 signals; LOW-SUPPORT |

### Per-SNR metrics for detector, baseline, PAM, and end-to-end pipeline

Each row is a measured metric with its own recording-bootstrap interval and support.

| SNR bin | Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---|---:|---:|---|
| [0,5) dB | YOLO precision | 100.00% | [100.00%, 100.00%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | YOLO recall | 16.67% | [16.67%, 16.67%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | YOLO mean_iou_true_positives | 62.06% | [62.06%, 62.06%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline precision | 83.33% | [83.33%, 83.33%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline recall | 83.33% | [83.33%, 83.33%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline mean_iou_true_positives | 71.89% | [71.89%, 71.89%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | PAM oracle-box accuracy (chance 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| [0,5) dB | Non-PAM detections classified as Others | 100.00% | [100.00%, 100.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| [0,5) dB | End-to-end precision | not measured | [100.00%, 100.00%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | End-to-end recall | not measured | [16.67%, 16.67%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | YOLO precision | not measured | [not measured, not measured] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | YOLO recall | 0.00% | [0.00%, 0.00%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | YOLO mean_iou_true_positives | not measured | [not measured, not measured] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | Energy baseline precision | 28.57% | [28.57%, 28.57%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | Energy baseline recall | 33.33% | [33.33%, 33.33%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | Energy baseline mean_iou_true_positives | 71.75% | [71.75%, 71.75%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | PAM oracle-box accuracy (chance 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| [10,15) dB | End-to-end precision | not measured | [not measured, not measured] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | End-to-end recall | not measured | [0.00%, 0.00%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [15,20) dB | YOLO precision | not measured | [not measured, not measured] | 2 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | YOLO recall | 0.00% | [0.00%, 0.00%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | YOLO mean_iou_true_positives | not measured | [not measured, not measured] | 2 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | Energy baseline precision | 46.15% | [14.29%, 83.33%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | Energy baseline recall | 50.00% | [16.67%, 83.33%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | Energy baseline mean_iou_true_positives | 66.19% | [54.87%, 68.45%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | PAM oracle-box accuracy (chance 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 2 recordings / 2 signals; LOW-SUPPORT |
| [15,20) dB | End-to-end precision | not measured | [not measured, not measured] | 2 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | End-to-end recall | not measured | [0.00%, 0.00%] | 2 recordings / 12 signals; LOW-SUPPORT |

### PAM accuracy and false-class results by class

PAM classifier chance accuracy is 1/7 = 14.29%; every PAM accuracy below carries this reference.
| Class / metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| 2-PSK: oracle-box accuracy (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| 2-PSK: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 4-PSK: oracle-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 4-PSK: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 8-PSK: oracle-box accuracy (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| 8-PSK: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 16-QAM: oracle-box accuracy (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| 16-QAM: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 64-QAM: oracle-box accuracy (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| 64-QAM: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 256-QAM: oracle-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 256-QAM: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| All-signal classifier: 2-PSK (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| All-signal classifier: 4-PSK (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| All-signal classifier: 8-PSK (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| All-signal classifier: 16-QAM (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| All-signal classifier: 64-QAM (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| All-signal classifier: 256-QAM (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| All-signal classifier: Others (chance level: 1/7 = 14.29%) | 43.75% | [18.75%, 68.75%] | 4 recordings / 16 signals; LOW-SUPPORT |
| False-class AM-DSB detections classified as Others | 100.00% | [100.00%, 100.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| False-class FM detections classified as Others | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| False-class 2-FSK detections classified as Others | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| False-class OOK detections classified as Others | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |

### PAM and end-to-end metrics

| Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| PAM accuracy with oracle boxes (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM accuracy with detector boxes (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| PAM CNN all-signal accuracy incl. Others (chance level: 1/7 = 14.29%) | 35.00% | [15.00%, 55.00%] | 4 recordings / 20 signals; LOW-SUPPORT |
| Non-PAM detections predicted as Others | 100.00% | [100.00%, 100.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| End-to-end precision | 100.00% | [100.00%, 100.00%] | 4 recordings / 24 signals; LOW-SUPPORT |
| End-to-end recall | 4.17% | [0.00%, 12.50%] | 4 recordings / 24 signals; LOW-SUPPORT |

Approximate PAM in-band SNR (dB): {"count": 4, "max": 44.03210932857731, "mean": 35.81053724124328, "min": 26.057143535041902}.

### Confusion matrices (measured counts)

Detector classes, main test:

| Actual \\ Predicted | AM-DSB | FM | OFDM | 2-FSK | OOK | PAM | missed | background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AM-DSB | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| FM | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| OFDM | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| 2-FSK | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| OOK | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| PAM | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| missed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| background | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

PAM subtype, oracle boxes:

| Actual \\ Predicted | 2-PSK | 4-PSK | 8-PSK | 16-QAM | 64-QAM | 256-QAM | Others |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-PSK | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 4-PSK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8-PSK | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 16-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 64-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 256-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Others | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

All-signal CNN classes:

| Actual \\ Predicted | 2-PSK | 4-PSK | 8-PSK | 16-QAM | 64-QAM | 256-QAM | Others |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-PSK | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 4-PSK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8-PSK | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 16-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 64-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 256-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Others | 0 | 0 | 1 | 6 | 0 | 2 | 7 |

## Shifted-parameter test

This distribution-shift test was not used for calibration.

| Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| YOLO precision | 100.00% | [100.00%, 100.00%] | 4 recordings / 24 signals; LOW-SUPPORT |
| YOLO recall | 12.50% | [0.00%, 25.00%] | 4 recordings / 24 signals; LOW-SUPPORT |
| YOLO mean IoU | 76.07% | [71.41%, 85.38%] | 4 recordings / 24 signals; LOW-SUPPORT |
| YOLO mAP@0.5 | 43.58% | [30.47%, 62.40%] | 4 recordings / 24 signals; LOW-SUPPORT |
| PAM oracle-box accuracy (chance 1/7 = 14.29%) | 50.00% | [0.00%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM detector-box accuracy (chance 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |

| Shifted detector class | Precision (95% CI) | Recall (95% CI) | Mean IoU (95% CI) | AP@0.5 (95% CI) | Support |
|---|---:|---:|---:|---:|---|
| AM-DSB | 50.00% [0.00%, 100.00%] | 25.00% [0.00%, 75.00%] | 85.38% [85.38%, 85.38%] | 52.06% [23.33%, 89.29%] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 37.75% [5.56%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 33.33% [33.33%, 66.67%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 25.00% [4.17%, 44.44%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK | 100.00% [100.00%, 100.00%] | 25.00% [0.00%, 75.00%] | 66.16% [66.16%, 66.16%] | 45.48% [25.00%, 81.67%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 67.86% [39.29%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |

### Shifted-test metrics by global SNR

| SNR bin | Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---|---:|---:|---|
| [-5,0) dB | YOLO precision | 100.00% | [100.00%, 100.00%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | YOLO recall | 8.33% | [0.00%, 16.67%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | YOLO mean_iou_true_positives | 85.38% | [85.38%, 85.38%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | Energy baseline precision | 52.94% | [44.44%, 62.50%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | Energy baseline recall | 75.00% | [66.67%, 83.33%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | Energy baseline mean_iou_true_positives | 88.47% | [87.95%, 89.13%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | PAM oracle-box accuracy (chance 1/7 = 14.29%) | 100.00% | [100.00%, 100.00%] | 2 recordings / 2 signals; LOW-SUPPORT |
| [-5,0) dB | Non-PAM detections classified as Others | 100.00% | [100.00%, 100.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| [-5,0) dB | End-to-end precision | 100.00% | [100.00%, 100.00%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | End-to-end recall | 8.33% | [0.00%, 16.67%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | YOLO precision | 100.00% | [100.00%, 100.00%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | YOLO recall | 16.67% | [0.00%, 33.33%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | YOLO mean_iou_true_positives | 71.41% | [71.41%, 71.41%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline precision | 62.50% | [62.50%, 62.50%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline recall | 83.33% | [83.33%, 83.33%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline mean_iou_true_positives | 87.32% | [85.43%, 89.21%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | PAM oracle-box accuracy (chance 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 2 recordings / 2 signals; LOW-SUPPORT |
| [0,5) dB | Non-PAM detections classified as Others | 0.00% | [0.00%, 0.00%] | 1 recordings / 2 signals; LOW-SUPPORT |
| [0,5) dB | End-to-end precision | 50.00% | [50.00%, 50.00%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | End-to-end recall | 8.33% | [0.00%, 16.67%] | 2 recordings / 12 signals; LOW-SUPPORT |

### Shifted-test confusion matrices

Detector classes:

| Actual \\ Predicted | AM-DSB | FM | OFDM | 2-FSK | OOK | PAM | missed | background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AM-DSB | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| FM | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| OFDM | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| 2-FSK | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| OOK | 0 | 0 | 0 | 0 | 1 | 0 | 3 | 0 |
| PAM | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| missed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| background | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

PAM subtype, oracle boxes:

| Actual \\ Predicted | 2-PSK | 4-PSK | 8-PSK | 16-QAM | 64-QAM | 256-QAM | Others |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-PSK | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4-PSK | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 8-PSK | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 64-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 256-QAM | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| Others | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Plots and measured artifacts

The metrics file preserves measured values, CIs, supports, validation thresholds, and confusion counts.
- [detector_confusion.png](results/reduced_cpu/plots/detector_confusion.png)
- [detector_confusion_shifted.png](results/reduced_cpu/plots/detector_confusion_shifted.png)
- [energy_baseline_metrics_vs_snr.png](results/reduced_cpu/plots/energy_baseline_metrics_vs_snr.png)
- [energy_baseline_metrics_vs_snr_shifted.png](results/reduced_cpu/plots/energy_baseline_metrics_vs_snr_shifted.png)
- [pam_accuracy_vs_snr.png](results/reduced_cpu/plots/pam_accuracy_vs_snr.png)
- [pam_accuracy_vs_snr_shifted.png](results/reduced_cpu/plots/pam_accuracy_vs_snr_shifted.png)
- [pam_all_classes_confusion.png](results/reduced_cpu/plots/pam_all_classes_confusion.png)
- [pam_subtype_confusion.png](results/reduced_cpu/plots/pam_subtype_confusion.png)
- [pam_subtype_confusion_shifted.png](results/reduced_cpu/plots/pam_subtype_confusion_shifted.png)
- [pam_training_curves.png](results/reduced_cpu/plots/pam_training_curves.png)
- [yolo_metrics_vs_snr.png](results/reduced_cpu/plots/yolo_metrics_vs_snr.png)
- [yolo_metrics_vs_snr_shifted.png](results/reduced_cpu/plots/yolo_metrics_vs_snr_shifted.png)

- [Complete measured metrics JSON](results/reduced_cpu/metrics.json)
- [Run summary](results/reduced_cpu/summary.txt)

## Runtime

Measured total pipeline runtime on hosted CPU (seconds): 1278.999718988.
Measured data generation runtime (seconds): 10.82237568699999.
YOLO timing probe: 22.309515293999993 seconds for 3 training batches. See [timing_probe.json](results/reduced_cpu/timing_probe.json) when present.
Detector training status: completed; evaluation completed: True.
The under-five-second target is specific to the owner's local machine and remains not measured there; hosted run time is not comparable.

## Interpretation

Use this reduced CPU execution to inspect the software path and measured outputs. It does not establish full-scale target performance. Investigate misses using validation data only; keep both test sets frozen.
