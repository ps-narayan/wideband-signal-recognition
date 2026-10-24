# Reduced CPU Run Results

Run label: reduced run: CPU, small data, not comparable to target
Comparison: reduced run: CPU, small data, not comparable to target. Target statuses are not comparable; this run does not establish whether the portfolio targets are met or missed.
Device: cpu
YOLOv5 source commit: f9578796b8a64889aa9e66dc06914d19f40d338e
Detector training status: completed; checkpoint: best.pt; requested epochs: 3.
PAM training status: completed; epochs completed: 5 of 5.
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
| Class mAP@0.5, global SNR >= 5 dB | >= 60% | 0.00% | [0.00%, 0.00%] | 3 recordings / 18 signals; LOW-SUPPORT | not comparable |
| PAM oracle-box accuracy, global SNR >= 10 dB | >= 80% | 33.33% | [0.00%, 100.00%] | 3 recordings / 3 signals; LOW-SUPPORT | not comparable |
| PAM oracle-box accuracy, global SNR [0, 1) dB | > 50% | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT | not comparable |
| Processing time per 1M samples on owner's machine | < 5 seconds | not measured | not applicable | not applicable | not comparable |

Validation-only operating-point calibration:
YOLO: target precision=80.00%; attained target=False; threshold=1.01; validation precision=0.00%.
Energy baseline: target precision=80.00%; attained target=True; mask threshold=1.75; confidence threshold=1.01; validation precision=91.67%.

## Main recording-disjoint test

### Detector and geometry baseline

| Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| YOLOv5n: geometry precision | not measured | [not measured, not measured] | 4 recordings / 24 signals; LOW-SUPPORT |
| YOLOv5n: class-agnostic recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 24 signals; LOW-SUPPORT |
| YOLOv5n: mean IoU of true positives | not measured | [not measured, not measured] | 4 recordings / 24 signals; LOW-SUPPORT |
| Energy baseline: geometry precision | 60.61% | [50.00%, 73.33%] | 4 recordings / 24 signals; LOW-SUPPORT |
| Energy baseline: class-agnostic recall | 83.33% | [70.83%, 95.83%] | 4 recordings / 24 signals; LOW-SUPPORT |
| Energy baseline: mean IoU of true positives | 72.38% | [67.84%, 77.23%] | 4 recordings / 24 signals; LOW-SUPPORT |
| YOLOv5n class mAP@0.5, all SNR | 0.00% | [0.00%, 0.00%] | 4 recordings / 24 signals; LOW-SUPPORT |
| YOLOv5n class mAP@0.5, global SNR >= 5 dB | 0.00% | [0.00%, 0.00%] | 3 recordings / 18 signals; LOW-SUPPORT |

### Per-class detector metrics

| Class metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| AM-DSB: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| AM-DSB: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| AM-DSB: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| AM-DSB: AP@0.5 | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM: AP@0.5 | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM: AP@0.5 | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK: AP@0.5 | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK: AP@0.5 | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM: AP@0.5 | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |

### Main test by global SNR

| Global SNR bin | YOLO recall | 95% CI | Support | YOLO mean IoU | 95% CI | Support | PAM oracle accuracy | 95% CI | Support |
|---|---:|---|---|---:|---|---|---:|---|---|
| [0,5) dB | 0.00% | [0.00%, 0.00%] | 1 recordings / 6 signals; LOW-SUPPORT | not measured | [not measured, not measured] | 1 recordings / 6 signals; LOW-SUPPORT | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| [10,15) dB | 0.00% | [0.00%, 0.00%] | 1 recordings / 6 signals; LOW-SUPPORT | not measured | [not measured, not measured] | 1 recordings / 6 signals; LOW-SUPPORT | 100.00% | [100.00%, 100.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| [15,20) dB | 0.00% | [0.00%, 0.00%] | 2 recordings / 12 signals; LOW-SUPPORT | not measured | [not measured, not measured] | 2 recordings / 12 signals; LOW-SUPPORT | 0.00% | [0.00%, 0.00%] | 2 recordings / 2 signals; LOW-SUPPORT |

### Per-SNR metrics for detector, baseline, PAM, and end-to-end pipeline

Each row is a measured metric with its own recording-bootstrap interval and support.

| SNR bin | Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---|---:|---:|---|
| [0,5) dB | YOLO precision | not measured | [not measured, not measured] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | YOLO recall | 0.00% | [0.00%, 0.00%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | YOLO mean_iou_true_positives | not measured | [not measured, not measured] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline precision | 83.33% | [83.33%, 83.33%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline recall | 83.33% | [83.33%, 83.33%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline mean_iou_true_positives | 75.29% | [75.29%, 75.29%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | PAM oracle-box accuracy | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| [0,5) dB | End-to-end precision | not measured | [0.00%, 0.00%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | End-to-end recall | not measured | [0.00%, 0.00%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | YOLO precision | not measured | [not measured, not measured] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | YOLO recall | 0.00% | [0.00%, 0.00%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | YOLO mean_iou_true_positives | not measured | [not measured, not measured] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | Energy baseline precision | 50.00% | [50.00%, 50.00%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | Energy baseline recall | 83.33% | [83.33%, 83.33%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | Energy baseline mean_iou_true_positives | 71.69% | [71.69%, 71.69%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | PAM oracle-box accuracy | 100.00% | [100.00%, 100.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| [10,15) dB | End-to-end precision | not measured | [0.00%, 0.00%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [10,15) dB | End-to-end recall | not measured | [0.00%, 0.00%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [15,20) dB | YOLO precision | not measured | [not measured, not measured] | 2 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | YOLO recall | 0.00% | [0.00%, 0.00%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | YOLO mean_iou_true_positives | not measured | [not measured, not measured] | 2 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | Energy baseline precision | 58.82% | [50.00%, 66.67%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | Energy baseline recall | 83.33% | [66.67%, 100.00%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | Energy baseline mean_iou_true_positives | 71.28% | [65.77%, 79.54%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | PAM oracle-box accuracy | 0.00% | [0.00%, 0.00%] | 2 recordings / 2 signals; LOW-SUPPORT |
| [15,20) dB | End-to-end precision | not measured | [0.00%, 0.00%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | End-to-end recall | not measured | [0.00%, 0.00%] | 2 recordings / 12 signals; LOW-SUPPORT |

### PAM accuracy and false-class results by class

| Class / metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| 2-PSK: oracle-box accuracy | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| 2-PSK: detector-box accuracy | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 4-PSK: oracle-box accuracy | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 4-PSK: detector-box accuracy | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 8-PSK: oracle-box accuracy | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| 8-PSK: detector-box accuracy | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 16-QAM: oracle-box accuracy | 100.00% | [100.00%, 100.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| 16-QAM: detector-box accuracy | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 64-QAM: oracle-box accuracy | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| 64-QAM: detector-box accuracy | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 256-QAM: oracle-box accuracy | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 256-QAM: detector-box accuracy | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| All-signal classifier: 2-PSK | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| All-signal classifier: 4-PSK | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| All-signal classifier: 8-PSK | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| All-signal classifier: 16-QAM | 100.00% | [100.00%, 100.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| All-signal classifier: 64-QAM | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| All-signal classifier: 256-QAM | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| All-signal classifier: Others | 12.50% | [0.00%, 25.00%] | 4 recordings / 16 signals; LOW-SUPPORT |
| False-class AM-DSB detections classified as Others | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| False-class FM detections classified as Others | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| False-class 2-FSK detections classified as Others | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| False-class OOK detections classified as Others | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |

### PAM and end-to-end metrics

| Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| PAM accuracy with oracle boxes | 25.00% | [0.00%, 75.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM accuracy with detector boxes | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| PAM CNN all-signal accuracy incl. Others | 15.00% | [5.00%, 20.00%] | 4 recordings / 20 signals; LOW-SUPPORT |
| Non-PAM detections predicted as Others | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| End-to-end precision | 0.00% | [0.00%, 0.00%] | 4 recordings / 24 signals; LOW-SUPPORT |
| End-to-end recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 24 signals; LOW-SUPPORT |

Approximate PAM in-band SNR (dB): {"count": 4, "max": 44.03210932857731, "mean": 35.81053724124328, "min": 26.057143535041902}.

### Confusion matrices (measured counts)

Detector classes, main test:

| Actual \\ Predicted | AM-DSB | FM | OFDM | 2-FSK | OOK | PAM | missed | background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AM-DSB | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
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
| 2-PSK | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 4-PSK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8-PSK | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 16-QAM | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 64-QAM | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 256-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Others | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

All-signal CNN classes:

| Actual \\ Predicted | 2-PSK | 4-PSK | 8-PSK | 16-QAM | 64-QAM | 256-QAM | Others |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-PSK | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 4-PSK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8-PSK | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 16-QAM | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 64-QAM | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 256-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Others | 3 | 3 | 0 | 7 | 0 | 1 | 2 |

## Shifted-parameter test

This distribution-shift test was not used for calibration.

| Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| YOLO precision | not measured | [not measured, not measured] | 4 recordings / 24 signals; LOW-SUPPORT |
| YOLO recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 24 signals; LOW-SUPPORT |
| YOLO mean IoU | not measured | [not measured, not measured] | 4 recordings / 24 signals; LOW-SUPPORT |
| YOLO mAP@0.5 | 0.00% | [0.00%, 0.00%] | 4 recordings / 24 signals; LOW-SUPPORT |
| PAM oracle-box accuracy | 25.00% | [0.00%, 75.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM detector-box accuracy | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |

| Shifted detector class | Precision (95% CI) | Recall (95% CI) | Mean IoU (95% CI) | AP@0.5 (95% CI) | Support |
|---|---:|---:|---:|---:|---|
| AM-DSB | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |

### Shifted-test metrics by global SNR

| SNR bin | Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---|---:|---:|---|
| [-5,0) dB | YOLO precision | not measured | [not measured, not measured] | 2 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | YOLO recall | 0.00% | [0.00%, 0.00%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | YOLO mean_iou_true_positives | not measured | [not measured, not measured] | 2 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | Energy baseline precision | 47.62% | [41.67%, 55.56%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | Energy baseline recall | 83.33% | [83.33%, 83.33%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | Energy baseline mean_iou_true_positives | 81.62% | [78.57%, 84.66%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | PAM oracle-box accuracy | 50.00% | [0.00%, 100.00%] | 2 recordings / 2 signals; LOW-SUPPORT |
| [-5,0) dB | End-to-end precision | 0.00% | [0.00%, 0.00%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | End-to-end recall | 0.00% | [0.00%, 0.00%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | YOLO precision | not measured | [not measured, not measured] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | YOLO recall | 0.00% | [0.00%, 0.00%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | YOLO mean_iou_true_positives | not measured | [not measured, not measured] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline precision | 62.50% | [62.50%, 62.50%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline recall | 83.33% | [83.33%, 83.33%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline mean_iou_true_positives | 86.77% | [86.74%, 86.80%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | PAM oracle-box accuracy | 0.00% | [0.00%, 0.00%] | 2 recordings / 2 signals; LOW-SUPPORT |
| [0,5) dB | End-to-end precision | 0.00% | [0.00%, 0.00%] | 2 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | End-to-end recall | 0.00% | [0.00%, 0.00%] | 2 recordings / 12 signals; LOW-SUPPORT |

### Shifted-test confusion matrices

Detector classes:

| Actual \\ Predicted | AM-DSB | FM | OFDM | 2-FSK | OOK | PAM | missed | background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AM-DSB | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
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
| 2-PSK | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 4-PSK | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 8-PSK | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 64-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 256-QAM | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
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

Measured total pipeline runtime on hosted CPU (seconds): 56.166612220000005.
Measured data generation runtime (seconds): 9.13046796799999.
Detector training status: completed; evaluation completed: True.
The under-five-second target is specific to the owner's local machine and remains not measured there; hosted run time is not comparable.

## Interpretation

Use this reduced CPU execution to inspect the software path and measured outputs. It does not establish full-scale target performance. Investigate misses using validation data only; keep both test sets frozen.
