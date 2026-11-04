# Reduced CPU Run Results

Run label: reduced run: CPU, small data, not comparable to target
Comparison: This reduced run is not comparable to the full-scale acceptance protocol.
Data source: self-contained reproducible synthetic SigMF/IQ generator. The 53.55 GiB Zenodo/DeepSig download requirement is dropped; no external recordings were used.
Device: cpu
YOLOv5 source commit: f9578796b8a64889aa9e66dc06914d19f40d338e
Detector training status: completed; checkpoint: best.pt; requested epochs: 135.
PAM training status: completed; epochs completed: 58 of 125.
Recording support: main test 4 recordings / 4 segments; shifted test 4 recordings / 4 segments.
Recording-ID leakage audit: train=12, validation=2, test=4; main splits are pairwise disjoint; shifted test IDs are separate. See [split_audit.json](results/reduced_cpu/split_audit.json) for IDs.
SNR definition: 10*log10(total clean signal power / total added noise power over the full I/Q segment).
Confidence intervals use recording-level bootstrap resampling. Cells with fewer than 30 signals are marked LOW-SUPPORT.

## Acceptance targets

Measured values are shown below. A missing value is marked not measured.

| Target | Threshold | Measured | Recording-level 95% CI | Support | Status |
|---|---:|---:|---:|---|---|
| Detector recall, global SNR >= 5 dB | >= 80% | 66.67% | [58.33%, 75.00%] | 2 recordings / 24 signals; LOW-SUPPORT | not comparable |
| Mean IoU, global SNR >= 5 dB | >= 0.75 | 86.17% | [84.65%, 87.34%] | 2 recordings / 24 signals; LOW-SUPPORT | not comparable |
| Class mAP@0.5, global SNR >= 5 dB | >= 60% | 53.63% | [43.68%, 69.52%] | 2 recordings / 24 signals; LOW-SUPPORT | not comparable |
| PAM oracle-box accuracy (chance 1/7 = 14.29%), global SNR >= 10 dB | >= 80% | 50.00% | [0.00%, 100.00%] | 2 recordings / 2 signals; LOW-SUPPORT | not comparable |
| PAM oracle-box accuracy (chance 1/7 = 14.29%), global SNR [0, 1) dB | > 50% | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT | not comparable |
| Local CPU processing time per 1M samples | < 5 seconds | not measured | not applicable | not measured | not comparable |

Validation-only operating-point calibration:
YOLO: target precision=80.00%; attained target=True; threshold=0.22852968000000004; validation precision=81.82%.
Energy baseline: target precision=80.00%; attained target=True; mask threshold=1.25; confidence threshold=2.55294132232666; validation precision=80.00%.

## Main recording-disjoint test

### Detector and geometry baseline

| Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| YOLOv5n: geometry precision | 82.86% | [75.86%, 88.24%] | 4 recordings / 48 signals |
| YOLOv5n: class-agnostic recall | 60.42% | [47.92%, 70.83%] | 4 recordings / 48 signals |
| YOLOv5n: mean IoU of true positives | 83.46% | [78.68%, 86.30%] | 4 recordings / 48 signals |
| Energy baseline: geometry precision | 60.00% | [55.56%, 66.04%] | 4 recordings / 48 signals |
| Energy baseline: class-agnostic recall | 62.50% | [54.17%, 70.83%] | 4 recordings / 48 signals |
| Energy baseline: mean IoU of true positives | 81.84% | [81.27%, 82.17%] | 4 recordings / 48 signals |
| YOLOv5n class mAP@0.5, all SNR | 44.96% | [38.80%, 62.39%] | 4 recordings / 48 signals |
| YOLOv5n class mAP@0.5, global SNR >= 5 dB | 53.63% | [43.68%, 69.52%] | 2 recordings / 24 signals; LOW-SUPPORT |

### Per-class detector metrics

| Class metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| AM-DSB: precision | 50.00% | [33.33%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| AM-DSB: recall | 75.00% | [25.00%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| AM-DSB: mean IoU | 73.02% | [53.25%, 86.72%] | 4 recordings / 4 signals; LOW-SUPPORT |
| AM-DSB: AP@0.5 | 65.00% | [37.50%, 95.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| AM-SSB: precision | 57.14% | [50.00%, 80.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| AM-SSB: recall | 100.00% | [100.00%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| AM-SSB: mean IoU | 79.32% | [69.98%, 88.66%] | 4 recordings / 4 signals; LOW-SUPPORT |
| AM-SSB: AP@0.5 | 95.00% | [84.12%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM: precision | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM: AP@0.5 | 39.17% | [24.12%, 87.50%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM: AP@0.5 | 12.50% | [0.00%, 75.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM: precision | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM: AP@0.5 | 17.79% | [13.33%, 80.25%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK: precision | 50.00% | [50.00%, 50.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK: recall | 100.00% | [100.00%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK: mean IoU | 85.21% | [77.56%, 92.86%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK: AP@0.5 | 75.00% | [50.00%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-CPFSK: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-CPFSK: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-CPFSK: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-CPFSK: AP@0.5 | 68.18% | [33.33%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 4-FSK: precision | 100.00% | [100.00%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 4-FSK: recall | 75.00% | [25.00%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 4-FSK: mean IoU | 88.70% | [86.63%, 91.01%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 4-FSK: AP@0.5 | 79.35% | [40.00%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 4-CPFSK: precision | 16.67% | [0.00%, 25.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 4-CPFSK: recall | 25.00% | [0.00%, 75.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 4-CPFSK: mean IoU | 90.85% | [90.85%, 90.85%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 4-CPFSK: AP@0.5 | 11.11% | [0.00%, 25.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK: AP@0.5 | 12.78% | [3.81%, 33.33%] | 4 recordings / 4 signals; LOW-SUPPORT |
| GMSK1: precision | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| GMSK1: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| GMSK1: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| GMSK1: AP@0.5 | 23.13% | [14.81%, 66.67%] | 4 recordings / 4 signals; LOW-SUPPORT |
| GMSK2: precision | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| GMSK2: recall | 0.00% | [0.00%, 0.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| GMSK2: mean IoU | not measured | [not measured, not measured] | 4 recordings / 4 signals; LOW-SUPPORT |
| GMSK2: AP@0.5 | 40.48% | [27.08%, 82.69%] | 4 recordings / 4 signals; LOW-SUPPORT |

### Main test by global SNR

| Global SNR bin | YOLO recall | 95% CI | Support | YOLO mean IoU | 95% CI | Support | PAM oracle accuracy (chance 1/7 = 14.29%) | 95% CI | Support |
|---|---:|---|---|---:|---|---|---:|---|---|
| [0,5) dB | 54.17% | [41.67%, 66.67%] | 2 recordings / 24 signals; LOW-SUPPORT | 80.13% | [75.89%, 82.79%] | 2 recordings / 24 signals; LOW-SUPPORT | 0.00% (chance 1/7 = 14.29%) | [0.00%, 0.00%] | 2 recordings / 2 signals; LOW-SUPPORT |
| [15,20) dB | 66.67% | [58.33%, 75.00%] | 2 recordings / 24 signals; LOW-SUPPORT | 86.17% | [84.65%, 87.34%] | 2 recordings / 24 signals; LOW-SUPPORT | 50.00% (chance 1/7 = 14.29%) | [0.00%, 100.00%] | 2 recordings / 2 signals; LOW-SUPPORT |

### Per-SNR metrics for detector, baseline, PAM, and end-to-end pipeline

Each row is a measured metric with its own recording-bootstrap interval and support.

| SNR bin | Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---|---:|---:|---|
| [0,5) dB | YOLO precision | 81.25% | [71.43%, 88.89%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [0,5) dB | YOLO recall | 54.17% | [41.67%, 66.67%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [0,5) dB | YOLO mean_iou_true_positives | 80.13% | [75.89%, 82.79%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline precision | 62.96% | [57.14%, 69.23%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline recall | 70.83% | [66.67%, 75.00%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline mean_iou_true_positives | 81.62% | [81.00%, 82.17%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [0,5) dB | PAM oracle-box accuracy (chance 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 2 recordings / 2 signals; LOW-SUPPORT |
| [0,5) dB | Non-PAM detections classified as Others | 10.00% | [0.00%, 14.29%] | 2 recordings / 10 signals; LOW-SUPPORT |
| [0,5) dB | End-to-end precision | not measured | [28.57%, 33.33%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [0,5) dB | End-to-end recall | not measured | [16.67%, 25.00%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [15,20) dB | YOLO precision | 84.21% | [81.82%, 87.50%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [15,20) dB | YOLO recall | 66.67% | [58.33%, 75.00%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [15,20) dB | YOLO mean_iou_true_positives | 86.17% | [84.65%, 87.34%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [15,20) dB | Energy baseline precision | 56.52% | [54.55%, 58.33%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [15,20) dB | Energy baseline recall | 54.17% | [50.00%, 58.33%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [15,20) dB | Energy baseline mean_iou_true_positives | 82.14% | [82.06%, 82.20%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [15,20) dB | PAM oracle-box accuracy (chance 1/7 = 14.29%) | 50.00% | [0.00%, 100.00%] | 2 recordings / 2 signals; LOW-SUPPORT |
| [15,20) dB | Non-PAM detections classified as Others | 30.77% | [28.57%, 33.33%] | 2 recordings / 13 signals; LOW-SUPPORT |
| [15,20) dB | End-to-end precision | not measured | [25.00%, 36.36%] | 2 recordings / 24 signals; LOW-SUPPORT |
| [15,20) dB | End-to-end recall | not measured | [16.67%, 33.33%] | 2 recordings / 24 signals; LOW-SUPPORT |

### PAM accuracy and false-class results by class

PAM classifier chance accuracy is 1/7 = 14.29%; every PAM accuracy below carries this reference.
| Class / metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| 2-PSK: oracle-box accuracy (chance level: 1/7 = 14.29%) | 100.00% | [100.00%, 100.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| 2-PSK: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 4-PSK: oracle-box accuracy (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| 4-PSK: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 8-PSK: oracle-box accuracy (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| 8-PSK: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 16-QAM: oracle-box accuracy (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| 16-QAM: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 64-QAM: oracle-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 64-QAM: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 256-QAM: oracle-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 256-QAM: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| All-signal classifier: 2-PSK (chance level: 1/7 = 14.29%) | 100.00% | [100.00%, 100.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| All-signal classifier: 4-PSK (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| All-signal classifier: 8-PSK (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| All-signal classifier: 16-QAM (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| All-signal classifier: 64-QAM (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| All-signal classifier: 256-QAM (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| All-signal classifier: Others (chance level: 1/7 = 14.29%) | 38.64% | [31.82%, 45.45%] | 4 recordings / 44 signals |
| False-class AM-DSB detections classified as Others | 0.00% | [0.00%, 0.00%] | 3 recordings / 5 signals; LOW-SUPPORT |
| False-class AM-SSB detections classified as Others | 50.00% | [0.00%, 100.00%] | 2 recordings / 2 signals; LOW-SUPPORT |
| False-class FM detections classified as Others | 0.00% | [0.00%, 0.00%] | 1 recordings / 2 signals; LOW-SUPPORT |
| False-class OFDM detections classified as Others | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| False-class 2-FSK detections classified as Others | 40.00% | [0.00%, 75.00%] | 4 recordings / 5 signals; LOW-SUPPORT |
| False-class 2-CPFSK detections classified as Others | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| False-class 4-FSK detections classified as Others | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| False-class 4-CPFSK detections classified as Others | 33.33% | [0.00%, 100.00%] | 3 recordings / 6 signals; LOW-SUPPORT |
| False-class OOK detections classified as Others | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| False-class GMSK1 detections classified as Others | 0.00% | [0.00%, 0.00%] | 2 recordings / 2 signals; LOW-SUPPORT |
| False-class GMSK2 detections classified as Others | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |

### PAM and end-to-end metrics

| Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| PAM accuracy with oracle boxes (chance level: 1/7 = 14.29%) | 25.00% | [0.00%, 75.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM accuracy with detector boxes (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| PAM CNN all-signal accuracy incl. Others (chance level: 1/7 = 14.29%) | 37.50% | [29.17%, 45.83%] | 4 recordings / 48 signals |
| Non-PAM detections predicted as Others | 21.74% | [10.00%, 30.77%] | 4 recordings / 23 signals; LOW-SUPPORT |
| End-to-end precision | 31.43% | [27.27%, 35.00%] | 4 recordings / 48 signals |
| End-to-end recall | 22.92% | [16.67%, 29.17%] | 4 recordings / 48 signals |

Approximate PAM in-band SNR (dB): {"count": 4, "max": 32.24664274599653, "mean": 23.246932055466278, "min": 11.33406840584006}.

### Confusion matrices (measured counts)

Detector classes, main test:

| Actual \\ Predicted | AM-DSB | AM-SSB | FM | PAM | OFDM | 2-FSK | 2-CPFSK | 4-FSK | 4-CPFSK | OOK | GMSK1 | GMSK2 | missed | background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AM-DSB | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| AM-SSB | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| PAM | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| OFDM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 3 | 0 |
| 2-FSK | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2-CPFSK | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 |
| 4-FSK | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4-CPFSK | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 0 |
| OOK | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 0 |
| GMSK1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| GMSK2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 1 | 0 |
| missed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| background | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |

PAM subtype, oracle boxes:

| Actual \\ Predicted | 2-PSK | 4-PSK | 8-PSK | 16-QAM | 64-QAM | 256-QAM | Others |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-PSK | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4-PSK | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 8-PSK | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16-QAM | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 64-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 256-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Others | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

All-signal CNN classes:

| Actual \\ Predicted | 2-PSK | 4-PSK | 8-PSK | 16-QAM | 64-QAM | 256-QAM | Others |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-PSK | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4-PSK | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 8-PSK | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16-QAM | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 64-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 256-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Others | 2 | 13 | 1 | 8 | 3 | 0 | 17 |

## Shifted-parameter test

This distribution-shift test was not used for calibration.

| Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| YOLO precision | 81.48% | [65.00%, 92.59%] | 4 recordings / 48 signals |
| YOLO recall | 45.83% | [27.08%, 58.33%] | 4 recordings / 48 signals |
| YOLO mean IoU | 71.77% | [66.42%, 76.52%] | 4 recordings / 48 signals |
| YOLO mAP@0.5 | 26.11% | [23.34%, 39.77%] | 4 recordings / 48 signals |
| PAM oracle-box accuracy (chance 1/7 = 14.29%) | 25.00% | [0.00%, 75.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM detector-box accuracy (chance 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |

| Shifted detector class | Precision (95% CI) | Recall (95% CI) | Mean IoU (95% CI) | AP@0.5 (95% CI) | Support |
|---|---:|---:|---:|---:|---|
| AM-DSB | 66.67% [50.00%, 100.00%] | 50.00% [0.00%, 100.00%] | 84.90% [81.39%, 88.40%] | 62.50% [25.00%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| AM-SSB | 44.44% [28.57%, 80.00%] | 100.00% [100.00%, 100.00%] | 67.73% [60.24%, 75.23%] | 58.33% [48.61%, 85.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| FM | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 34.38% [12.50%, 83.33%] | 4 recordings / 4 signals; LOW-SUPPORT |
| PAM | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 6.67% [0.00%, 22.22%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OFDM | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 13.24% [1.92%, 40.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-FSK | 9.09% [0.00%, 25.00%] | 25.00% [0.00%, 75.00%] | 94.13% [94.13%, 94.13%] | 16.45% [5.00%, 75.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 2-CPFSK | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 4.17% [0.00%, 21.88%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 4-FSK | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 4.53% [0.89%, 15.95%] | 4 recordings / 4 signals; LOW-SUPPORT |
| 4-CPFSK | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 29.38% [16.67%, 85.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| OOK | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 58.33% [40.00%, 87.50%] | 4 recordings / 4 signals; LOW-SUPPORT |
| GMSK1 | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 12.88% [0.00%, 31.41%] | 4 recordings / 4 signals; LOW-SUPPORT |
| GMSK2 | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 12.50% [1.39%, 39.29%] | 4 recordings / 4 signals; LOW-SUPPORT |

### Shifted-test metrics by global SNR

| SNR bin | Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---|---:|---:|---|
| [-5,0) dB | YOLO precision | 78.95% | [50.00%, 100.00%] | 3 recordings / 36 signals |
| [-5,0) dB | YOLO recall | 41.67% | [16.67%, 58.33%] | 3 recordings / 36 signals |
| [-5,0) dB | YOLO mean_iou_true_positives | 68.76% | [61.34%, 71.44%] | 3 recordings / 36 signals |
| [-5,0) dB | Energy baseline precision | 77.14% | [69.23%, 83.33%] | 3 recordings / 36 signals |
| [-5,0) dB | Energy baseline recall | 75.00% | [66.67%, 83.33%] | 3 recordings / 36 signals |
| [-5,0) dB | Energy baseline mean_iou_true_positives | 84.66% | [81.91%, 87.20%] | 3 recordings / 36 signals |
| [-5,0) dB | PAM oracle-box accuracy (chance 1/7 = 14.29%) | 33.33% | [0.00%, 100.00%] | 3 recordings / 3 signals; LOW-SUPPORT |
| [-5,0) dB | Non-PAM detections classified as Others | 0.00% | [0.00%, 0.00%] | 3 recordings / 12 signals; LOW-SUPPORT |
| [-5,0) dB | End-to-end precision | 15.79% | [11.11%, 25.00%] | 3 recordings / 36 signals |
| [-5,0) dB | End-to-end recall | 8.33% | [8.33%, 8.33%] | 3 recordings / 36 signals |
| [0,5) dB | YOLO precision | 87.50% | [87.50%, 87.50%] | 1 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | YOLO recall | 58.33% | [58.33%, 58.33%] | 1 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | YOLO mean_iou_true_positives | 78.22% | [78.22%, 78.22%] | 1 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline precision | 57.14% | [57.14%, 57.14%] | 1 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline recall | 66.67% | [66.67%, 66.67%] | 1 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | Energy baseline mean_iou_true_positives | 86.68% | [86.68%, 86.68%] | 1 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | PAM oracle-box accuracy (chance 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 1 recordings / 1 signals; LOW-SUPPORT |
| [0,5) dB | Non-PAM detections classified as Others | 33.33% | [33.33%, 33.33%] | 1 recordings / 6 signals; LOW-SUPPORT |
| [0,5) dB | End-to-end precision | 37.50% | [37.50%, 37.50%] | 1 recordings / 12 signals; LOW-SUPPORT |
| [0,5) dB | End-to-end recall | 25.00% | [25.00%, 25.00%] | 1 recordings / 12 signals; LOW-SUPPORT |

### Shifted-test confusion matrices

Detector classes:

| Actual \\ Predicted | AM-DSB | AM-SSB | FM | PAM | OFDM | 2-FSK | 2-CPFSK | 4-FSK | 4-CPFSK | OOK | GMSK1 | GMSK2 | missed | background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AM-DSB | 1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| AM-SSB | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 3 | 0 |
| PAM | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 |
| OFDM | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| 2-FSK | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| 2-CPFSK | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 4-FSK | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 4-CPFSK | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| OOK | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| GMSK1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| GMSK2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| missed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| background | 1 | 2 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

PAM subtype, oracle boxes:

| Actual \\ Predicted | 2-PSK | 4-PSK | 8-PSK | 16-QAM | 64-QAM | 256-QAM | Others |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-PSK | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| 4-PSK | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 8-PSK | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 16-QAM | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 64-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 256-QAM | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
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

Measured total pipeline runtime on the configured device (seconds): 1262.371373245.
Measured data generation runtime (seconds): 13.928601621999974.
YOLO timing probe: 22.414368083 seconds for 3 training batches. See [results/reduced_cpu/timing_probe.json](results/reduced_cpu/timing_probe.json) when present.
Detector training status: completed; evaluation completed: True.
The local per-segment timing target is not yet measured; see the acceptance table above.

## Interpretation

Use this reduced CPU execution to inspect the software path and measured outputs. It does not establish full-scale target performance. Investigate misses using validation data only; keep both test sets frozen.
