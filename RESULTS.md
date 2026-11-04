# Synthetic 12-Class CPU Run Results

Run label: synthetic-only 12-class major project CPU run
Comparison: Target status uses each measured point estimate; recording-level confidence intervals show uncertainty.
Data source: self-contained reproducible synthetic SigMF/IQ generator. The 53.55 GiB Zenodo/DeepSig download requirement is dropped; no external recordings were used.
Device: cpu
YOLOv5 source commit: f9578796b8a64889aa9e66dc06914d19f40d338e
Detector training status: completed; checkpoint: best.pt; requested epochs: 80.
PAM training status: completed; epochs completed: 42 of 60.
Recording support: main test 60 recordings / 60 segments; shifted test 12 recordings / 12 segments.
Recording-ID leakage audit: train=48, validation=12, test=60; main splits are pairwise disjoint; shifted test IDs are separate. See [split_audit.json](results/major_project_cpu/split_audit.json) for IDs.
SNR definition: 10*log10(total clean signal power / total added noise power over the full I/Q segment).
Confidence intervals use recording-level bootstrap resampling. Cells with fewer than 30 signals are marked LOW-SUPPORT.

## Acceptance targets

Measured values are shown below. A missing value is marked not measured.

| Target | Threshold | Measured | Recording-level 95% CI | Support | Status |
|---|---:|---:|---:|---|---|
| Detector recall, global SNR >= 5 dB | >= 80% | 63.06% | [59.30%, 66.94%] | 30 recordings / 360 signals | missed |
| Mean IoU, global SNR >= 5 dB | >= 0.75 | 87.89% | [87.18%, 88.68%] | 30 recordings / 360 signals | met |
| Class mAP@0.5, global SNR >= 5 dB | >= 60% | 41.27% | [40.95%, 45.11%] | 30 recordings / 360 signals | missed |
| PAM oracle-box accuracy (chance 1/7 = 14.29%), global SNR >= 10 dB | >= 80% | 46.67% | [30.00%, 63.33%] | 30 recordings / 30 signals | missed |
| PAM oracle-box accuracy (chance 1/7 = 14.29%), global SNR [0, 1) dB | > 50% | 36.67% | [20.00%, 53.33%] | 30 recordings / 30 signals | missed |
| Local CPU processing time per 1M samples | < 5 seconds | 29.891 s | not applicable | Maximum measured CLI internal wall time across two local CPU synthetic SigMF examples; includes input read, spectrogram, YOLO, extraction/classification, and annotated PNG write; excludes JSON serialization. | missed |

Validation-only operating-point calibration:
YOLO: target precision=80.00%; attained target=True; threshold=0.21875284000000003; validation precision=81.82%.
Energy baseline: target precision=80.00%; attained target=True; mask threshold=1.5; confidence threshold=2.301176729202271; validation precision=81.56%.

## Main recording-disjoint test

### Detector and geometry baseline

| Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| YOLOv5n: geometry precision | 79.65% | [77.20%, 82.27%] | 60 recordings / 720 signals |
| YOLOv5n: class-agnostic recall | 56.53% | [53.47%, 59.58%] | 60 recordings / 720 signals |
| YOLOv5n: mean IoU of true positives | 87.57% | [87.00%, 88.12%] | 60 recordings / 720 signals |
| Energy baseline: geometry precision | 80.61% | [78.46%, 82.63%] | 60 recordings / 720 signals |
| Energy baseline: class-agnostic recall | 81.39% | [79.31%, 83.33%] | 60 recordings / 720 signals |
| Energy baseline: mean IoU of true positives | 82.01% | [81.60%, 82.42%] | 60 recordings / 720 signals |
| YOLOv5n class mAP@0.5, all SNR | 39.34% | [38.66%, 42.41%] | 60 recordings / 720 signals |
| YOLOv5n class mAP@0.5, global SNR >= 5 dB | 41.27% | [40.95%, 45.11%] | 30 recordings / 360 signals |

### Per-class detector metrics

| Class metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| AM-DSB: precision | 47.06% | [38.21%, 57.28%] | 60 recordings / 60 signals |
| AM-DSB: recall | 53.33% | [41.67%, 66.67%] | 60 recordings / 60 signals |
| AM-DSB: mean IoU | 87.99% | [86.22%, 89.86%] | 60 recordings / 60 signals |
| AM-DSB: AP@0.5 | 55.65% | [47.80%, 66.67%] | 60 recordings / 60 signals |
| AM-SSB: precision | 98.33% | [94.30%, 100.00%] | 60 recordings / 60 signals |
| AM-SSB: recall | 98.33% | [95.00%, 100.00%] | 60 recordings / 60 signals |
| AM-SSB: mean IoU | 85.48% | [84.00%, 86.95%] | 60 recordings / 60 signals |
| AM-SSB: AP@0.5 | 98.33% | [95.00%, 100.00%] | 60 recordings / 60 signals |
| FM: precision | 53.49% | [42.26%, 64.44%] | 60 recordings / 60 signals |
| FM: recall | 38.33% | [26.67%, 51.67%] | 60 recordings / 60 signals |
| FM: mean IoU | 90.20% | [87.84%, 91.81%] | 60 recordings / 60 signals |
| FM: AP@0.5 | 57.09% | [47.92%, 69.09%] | 60 recordings / 60 signals |
| PAM: precision | not measured | [not measured, not measured] | 60 recordings / 60 signals |
| PAM: recall | 0.00% | [0.00%, 0.00%] | 60 recordings / 60 signals |
| PAM: mean IoU | not measured | [not measured, not measured] | 60 recordings / 60 signals |
| PAM: AP@0.5 | 9.50% | [5.43%, 18.70%] | 60 recordings / 60 signals |
| OFDM: precision | 36.36% | [3.39%, 66.67%] | 60 recordings / 60 signals |
| OFDM: recall | 6.67% | [0.79%, 13.33%] | 60 recordings / 60 signals |
| OFDM: mean IoU | 90.82% | [87.69%, 94.06%] | 60 recordings / 60 signals |
| OFDM: AP@0.5 | 24.89% | [19.34%, 36.02%] | 60 recordings / 60 signals |
| 2-FSK: precision | 49.58% | [46.48%, 52.68%] | 60 recordings / 60 signals |
| 2-FSK: recall | 98.33% | [95.00%, 100.00%] | 60 recordings / 60 signals |
| 2-FSK: mean IoU | 84.75% | [83.22%, 86.30%] | 60 recordings / 60 signals |
| 2-FSK: AP@0.5 | 70.89% | [64.70%, 79.70%] | 60 recordings / 60 signals |
| 2-CPFSK: precision | 22.58% | [14.52%, 33.11%] | 60 recordings / 60 signals |
| 2-CPFSK: recall | 23.33% | [15.00%, 35.87%] | 60 recordings / 60 signals |
| 2-CPFSK: mean IoU | 87.06% | [83.62%, 90.11%] | 60 recordings / 60 signals |
| 2-CPFSK: AP@0.5 | 23.28% | [19.43%, 30.36%] | 60 recordings / 60 signals |
| 4-FSK: precision | 41.30% | [35.54%, 46.70%] | 60 recordings / 60 signals |
| 4-FSK: recall | 63.33% | [49.12%, 75.00%] | 60 recordings / 60 signals |
| 4-FSK: mean IoU | 84.99% | [82.82%, 86.93%] | 60 recordings / 60 signals |
| 4-FSK: AP@0.5 | 49.89% | [43.02%, 60.03%] | 60 recordings / 60 signals |
| 4-CPFSK: precision | 15.79% | [0.00%, 30.77%] | 60 recordings / 60 signals |
| 4-CPFSK: recall | 5.00% | [0.00%, 11.67%] | 60 recordings / 60 signals |
| 4-CPFSK: mean IoU | 86.02% | [82.61%, 89.31%] | 60 recordings / 60 signals |
| 4-CPFSK: AP@0.5 | 22.67% | [19.19%, 29.49%] | 60 recordings / 60 signals |
| OOK: precision | not measured | [not measured, not measured] | 60 recordings / 60 signals |
| OOK: recall | 0.00% | [0.00%, 0.00%] | 60 recordings / 60 signals |
| OOK: mean IoU | not measured | [not measured, not measured] | 60 recordings / 60 signals |
| OOK: AP@0.5 | 19.59% | [16.07%, 28.71%] | 60 recordings / 60 signals |
| GMSK1: precision | 13.89% | [3.77%, 24.32%] | 60 recordings / 60 signals |
| GMSK1: recall | 8.33% | [1.67%, 15.00%] | 60 recordings / 60 signals |
| GMSK1: mean IoU | 95.88% | [94.61%, 97.39%] | 60 recordings / 60 signals |
| GMSK1: AP@0.5 | 23.25% | [18.34%, 30.93%] | 60 recordings / 60 signals |
| GMSK2: precision | 0.00% | [0.00%, 0.00%] | 60 recordings / 60 signals |
| GMSK2: recall | 0.00% | [0.00%, 0.00%] | 60 recordings / 60 signals |
| GMSK2: mean IoU | not measured | [not measured, not measured] | 60 recordings / 60 signals |
| GMSK2: AP@0.5 | 17.10% | [14.45%, 24.47%] | 60 recordings / 60 signals |

### Main test by global SNR

| Global SNR bin | YOLO recall | 95% CI | Support | YOLO mean IoU | 95% CI | Support | PAM oracle accuracy (chance 1/7 = 14.29%) | 95% CI | Support |
|---|---:|---|---|---:|---|---|---:|---|---|
| [0,5) dB | 50.00% | [46.39%, 53.48%] | 30 recordings / 360 signals | 87.18% | [86.28%, 88.12%] | 30 recordings / 360 signals | 36.67% (chance 1/7 = 14.29%) | [20.00%, 53.33%] | 30 recordings / 30 signals |
| [10,15) dB | 62.50% | [56.25%, 70.14%] | 12 recordings / 144 signals | 87.81% | [86.80%, 88.74%] | 12 recordings / 144 signals | 41.67% (chance 1/7 = 14.29%) | [16.67%, 66.67%] | 12 recordings / 12 signals; LOW-SUPPORT |
| [15,20) dB | 63.43% | [58.80%, 68.06%] | 18 recordings / 216 signals | 87.94% | [87.00%, 88.90%] | 18 recordings / 216 signals | 50.00% (chance 1/7 = 14.29%) | [27.78%, 72.22%] | 18 recordings / 18 signals; LOW-SUPPORT |

### Per-SNR metrics for detector, baseline, PAM, and end-to-end pipeline

Each row is a measured metric with its own recording-bootstrap interval and support.

| SNR bin | Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---|---:|---:|---|
| [0,5) dB | YOLO precision | 79.65% | [75.75%, 84.06%] | 30 recordings / 360 signals |
| [0,5) dB | YOLO recall | 50.00% | [46.39%, 53.48%] | 30 recordings / 360 signals |
| [0,5) dB | YOLO mean_iou_true_positives | 87.18% | [86.28%, 88.12%] | 30 recordings / 360 signals |
| [0,5) dB | Energy baseline precision | 86.18% | [83.62%, 88.72%] | 30 recordings / 360 signals |
| [0,5) dB | Energy baseline recall | 88.33% | [86.67%, 89.87%] | 30 recordings / 360 signals |
| [0,5) dB | Energy baseline mean_iou_true_positives | 81.37% | [80.83%, 81.88%] | 30 recordings / 360 signals |
| [0,5) dB | PAM oracle-box accuracy (chance 1/7 = 14.29%) | 36.67% | [20.00%, 53.33%] | 30 recordings / 30 signals |
| [0,5) dB | Non-PAM detections classified as Others | 63.64% | [57.25%, 70.04%] | 30 recordings / 132 signals |
| [0,5) dB | End-to-end precision | not measured | [36.09%, 49.45%] | 30 recordings / 360 signals |
| [0,5) dB | End-to-end recall | not measured | [23.06%, 30.29%] | 30 recordings / 360 signals |
| [10,15) dB | YOLO precision | 81.08% | [75.86%, 86.43%] | 12 recordings / 144 signals |
| [10,15) dB | YOLO recall | 62.50% | [56.25%, 70.14%] | 12 recordings / 144 signals |
| [10,15) dB | YOLO mean_iou_true_positives | 87.81% | [86.80%, 88.74%] | 12 recordings / 144 signals |
| [10,15) dB | Energy baseline precision | 75.00% | [75.00%, 75.00%] | 12 recordings / 144 signals |
| [10,15) dB | Energy baseline recall | 75.00% | [75.00%, 75.00%] | 12 recordings / 144 signals |
| [10,15) dB | Energy baseline mean_iou_true_positives | 82.29% | [81.56%, 83.09%] | 12 recordings / 144 signals |
| [10,15) dB | PAM oracle-box accuracy (chance 1/7 = 14.29%) | 41.67% | [16.67%, 66.67%] | 12 recordings / 12 signals; LOW-SUPPORT |
| [10,15) dB | Non-PAM detections classified as Others | 69.01% | [59.09%, 78.52%] | 12 recordings / 71 signals |
| [10,15) dB | End-to-end precision | not measured | [34.21%, 41.76%] | 12 recordings / 144 signals |
| [10,15) dB | End-to-end recall | not measured | [25.69%, 33.33%] | 12 recordings / 144 signals |
| [15,20) dB | YOLO precision | 78.74% | [75.77%, 81.32%] | 18 recordings / 216 signals |
| [15,20) dB | YOLO recall | 63.43% | [58.80%, 68.06%] | 18 recordings / 216 signals |
| [15,20) dB | YOLO mean_iou_true_positives | 87.94% | [87.00%, 88.90%] | 18 recordings / 216 signals |
| [15,20) dB | Energy baseline precision | 74.77% | [72.38%, 76.42%] | 18 recordings / 216 signals |
| [15,20) dB | Energy baseline recall | 74.07% | [72.22%, 75.00%] | 18 recordings / 216 signals |
| [15,20) dB | Energy baseline mean_iou_true_positives | 83.09% | [82.40%, 83.72%] | 18 recordings / 216 signals |
| [15,20) dB | PAM oracle-box accuracy (chance 1/7 = 14.29%) | 50.00% | [27.78%, 72.22%] | 18 recordings / 18 signals; LOW-SUPPORT |
| [15,20) dB | Non-PAM detections classified as Others | 71.00% | [63.23%, 77.72%] | 18 recordings / 100 signals |
| [15,20) dB | End-to-end precision | not measured | [33.05%, 43.16%] | 18 recordings / 216 signals |
| [15,20) dB | End-to-end recall | not measured | [26.85%, 33.80%] | 18 recordings / 216 signals |

### PAM accuracy and false-class results by class

PAM classifier chance accuracy is 1/7 = 14.29%; every PAM accuracy below carries this reference.
| Class / metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| 2-PSK: oracle-box accuracy (chance level: 1/7 = 14.29%) | 100.00% | [100.00%, 100.00%] | 10 recordings / 10 signals; LOW-SUPPORT |
| 2-PSK: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 4-PSK: oracle-box accuracy (chance level: 1/7 = 14.29%) | 30.00% | [4.75%, 60.00%] | 10 recordings / 10 signals; LOW-SUPPORT |
| 4-PSK: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 8-PSK: oracle-box accuracy (chance level: 1/7 = 14.29%) | 60.00% | [30.00%, 90.00%] | 10 recordings / 10 signals; LOW-SUPPORT |
| 8-PSK: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 16-QAM: oracle-box accuracy (chance level: 1/7 = 14.29%) | 20.00% | [0.00%, 50.00%] | 10 recordings / 10 signals; LOW-SUPPORT |
| 16-QAM: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 64-QAM: oracle-box accuracy (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 10 recordings / 10 signals; LOW-SUPPORT |
| 64-QAM: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| 256-QAM: oracle-box accuracy (chance level: 1/7 = 14.29%) | 40.00% | [10.00%, 70.00%] | 10 recordings / 10 signals; LOW-SUPPORT |
| 256-QAM: detector-box accuracy (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| All-signal classifier: 2-PSK (chance level: 1/7 = 14.29%) | 100.00% | [100.00%, 100.00%] | 10 recordings / 10 signals; LOW-SUPPORT |
| All-signal classifier: 4-PSK (chance level: 1/7 = 14.29%) | 30.00% | [4.75%, 60.00%] | 10 recordings / 10 signals; LOW-SUPPORT |
| All-signal classifier: 8-PSK (chance level: 1/7 = 14.29%) | 60.00% | [30.00%, 90.00%] | 10 recordings / 10 signals; LOW-SUPPORT |
| All-signal classifier: 16-QAM (chance level: 1/7 = 14.29%) | 20.00% | [0.00%, 50.00%] | 10 recordings / 10 signals; LOW-SUPPORT |
| All-signal classifier: 64-QAM (chance level: 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 10 recordings / 10 signals; LOW-SUPPORT |
| All-signal classifier: 256-QAM (chance level: 1/7 = 14.29%) | 40.00% | [10.00%, 70.00%] | 10 recordings / 10 signals; LOW-SUPPORT |
| All-signal classifier: Others (chance level: 1/7 = 14.29%) | 68.03% | [65.98%, 70.30%] | 60 recordings / 660 signals |
| False-class AM-DSB detections classified as Others | 39.29% | [25.49%, 50.82%] | 40 recordings / 56 signals |
| False-class AM-SSB detections classified as Others | 96.55% | [91.38%, 100.00%] | 58 recordings / 58 signals |
| False-class FM detections classified as Others | 57.14% | [40.00%, 73.72%] | 23 recordings / 28 signals; LOW-SUPPORT |
| False-class OFDM detections classified as Others | 50.00% | [16.67%, 83.33%] | 6 recordings / 6 signals; LOW-SUPPORT |
| False-class 2-FSK detections classified as Others | 75.00% | [62.00%, 87.21%] | 30 recordings / 40 signals |
| False-class 2-CPFSK detections classified as Others | 67.31% | [54.90%, 78.10%] | 33 recordings / 52 signals |
| False-class 4-FSK detections classified as Others | 50.00% | [20.00%, 80.00%] | 10 recordings / 10 signals; LOW-SUPPORT |
| False-class 4-CPFSK detections classified as Others | 73.68% | [52.63%, 94.44%] | 13 recordings / 19 signals; LOW-SUPPORT |
| False-class OOK detections classified as Others | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| False-class GMSK1 detections classified as Others | 66.67% | [51.72%, 81.02%] | 24 recordings / 33 signals |
| False-class GMSK2 detections classified as Others | 100.00% | [100.00%, 100.00%] | 1 recordings / 1 signals; LOW-SUPPORT |

### PAM and end-to-end metrics

| Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| PAM accuracy with oracle boxes (chance level: 1/7 = 14.29%) | 41.67% | [30.00%, 55.00%] | 60 recordings / 60 signals |
| PAM accuracy with detector boxes (chance level: 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |
| PAM CNN all-signal accuracy incl. Others (chance level: 1/7 = 14.29%) | 65.83% | [63.75%, 67.92%] | 60 recordings / 720 signals |
| Non-PAM detections predicted as Others | 67.33% | [62.46%, 71.61%] | 60 recordings / 303 signals |
| End-to-end precision | 39.92% | [36.56%, 43.28%] | 60 recordings / 720 signals |
| End-to-end recall | 28.33% | [26.25%, 30.69%] | 60 recordings / 720 signals |

Approximate PAM in-band SNR (dB): {"count": 60, "max": 28.04858853358853, "mean": 19.807321419015715, "min": 11.243811959412808}.

### Confusion matrices (measured counts)

Detector classes, main test:

| Actual \\ Predicted | AM-DSB | AM-SSB | FM | PAM | OFDM | 2-FSK | 2-CPFSK | 4-FSK | 4-CPFSK | OOK | GMSK1 | GMSK2 | missed | background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AM-DSB | 32 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 | 0 |
| AM-SSB | 0 | 59 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| FM | 16 | 0 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 14 | 0 |
| PAM | 1 | 0 | 1 | 0 | 3 | 0 | 4 | 0 | 3 | 0 | 4 | 0 | 44 | 0 |
| OFDM | 0 | 0 | 0 | 0 | 3 | 1 | 17 | 2 | 0 | 0 | 0 | 0 | 37 | 0 |
| 2-FSK | 0 | 0 | 0 | 0 | 0 | 52 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2-CPFSK | 0 | 0 | 0 | 0 | 0 | 2 | 11 | 10 | 1 | 0 | 0 | 0 | 36 | 0 |
| 4-FSK | 0 | 0 | 0 | 0 | 0 | 40 | 0 | 18 | 0 | 0 | 0 | 0 | 2 | 0 |
| 4-CPFSK | 0 | 0 | 0 | 0 | 0 | 0 | 13 | 0 | 3 | 0 | 0 | 1 | 43 | 0 |
| OOK | 5 | 0 | 4 | 0 | 0 | 0 | 1 | 0 | 3 | 0 | 13 | 0 | 34 | 0 |
| GMSK1 | 6 | 0 | 7 | 0 | 0 | 0 | 1 | 0 | 6 | 0 | 5 | 0 | 35 | 0 |
| GMSK2 | 0 | 0 | 0 | 0 | 1 | 0 | 10 | 0 | 3 | 0 | 3 | 0 | 43 | 0 |
| missed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| background | 8 | 1 | 6 | 0 | 4 | 24 | 5 | 54 | 0 | 0 | 2 | 0 | 0 | 0 |

PAM subtype, oracle boxes:

| Actual \\ Predicted | 2-PSK | 4-PSK | 8-PSK | 16-QAM | 64-QAM | 256-QAM | Others |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-PSK | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4-PSK | 0 | 3 | 2 | 0 | 0 | 0 | 5 |
| 8-PSK | 0 | 3 | 6 | 0 | 0 | 0 | 1 |
| 16-QAM | 0 | 0 | 1 | 2 | 1 | 5 | 1 |
| 64-QAM | 0 | 0 | 0 | 6 | 0 | 4 | 0 |
| 256-QAM | 0 | 0 | 1 | 3 | 1 | 4 | 1 |
| Others | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

All-signal CNN classes:

| Actual \\ Predicted | 2-PSK | 4-PSK | 8-PSK | 16-QAM | 64-QAM | 256-QAM | Others |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-PSK | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4-PSK | 0 | 3 | 2 | 0 | 0 | 0 | 5 |
| 8-PSK | 0 | 3 | 6 | 0 | 0 | 0 | 1 |
| 16-QAM | 0 | 0 | 1 | 2 | 1 | 5 | 1 |
| 64-QAM | 0 | 0 | 0 | 6 | 0 | 4 | 0 |
| 256-QAM | 0 | 0 | 1 | 3 | 1 | 4 | 1 |
| Others | 4 | 29 | 34 | 44 | 33 | 67 | 449 |

## Shifted-parameter test

This distribution-shift test was not used for calibration.

| Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---:|---:|---|
| YOLO precision | 74.19% | [67.74%, 81.32%] | 12 recordings / 144 signals |
| YOLO recall | 47.92% | [44.44%, 52.08%] | 12 recordings / 144 signals |
| YOLO mean IoU | 80.51% | [78.79%, 82.32%] | 12 recordings / 144 signals |
| YOLO mAP@0.5 | 32.21% | [30.54%, 39.51%] | 12 recordings / 144 signals |
| PAM oracle-box accuracy (chance 1/7 = 14.29%) | 16.67% | [0.00%, 41.67%] | 12 recordings / 12 signals; LOW-SUPPORT |
| PAM detector-box accuracy (chance 1/7 = 14.29%) | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT |

| Shifted detector class | Precision (95% CI) | Recall (95% CI) | Mean IoU (95% CI) | AP@0.5 (95% CI) | Support |
|---|---:|---:|---:|---:|---|
| AM-DSB | 33.33% [0.00%, 100.00%] | 8.33% [0.00%, 25.00%] | 86.10% [86.10%, 86.10%] | 51.84% [36.38%, 82.47%] | 12 recordings / 12 signals; LOW-SUPPORT |
| AM-SSB | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] | 82.04% [75.99%, 87.03%] | 100.00% [100.00%, 100.00%] | 12 recordings / 12 signals; LOW-SUPPORT |
| FM | 100.00% [100.00%, 100.00%] | 8.33% [0.00%, 25.00%] | 89.70% [89.70%, 89.70%] | 35.57% [21.64%, 55.48%] | 12 recordings / 12 signals; LOW-SUPPORT |
| PAM | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 13.11% [2.78%, 32.66%] | 12 recordings / 12 signals; LOW-SUPPORT |
| OFDM | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 1.85% [0.12%, 8.52%] | 12 recordings / 12 signals; LOW-SUPPORT |
| 2-FSK | 23.26% [17.44%, 28.57%] | 83.33% [66.67%, 100.00%] | 81.49% [77.40%, 85.40%] | 45.89% [31.92%, 70.92%] | 12 recordings / 12 signals; LOW-SUPPORT |
| 2-CPFSK | 14.29% [0.00%, 34.92%] | 8.33% [0.00%, 25.00%] | 75.76% [75.76%, 75.76%] | 9.68% [4.55%, 21.69%] | 12 recordings / 12 signals; LOW-SUPPORT |
| 4-FSK | 29.17% [17.39%, 42.11%] | 58.33% [33.33%, 83.33%] | 81.30% [78.12%, 85.29%] | 36.51% [23.83%, 60.14%] | 12 recordings / 12 signals; LOW-SUPPORT |
| 4-CPFSK | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 16.18% [8.97%, 32.23%] | 12 recordings / 12 signals; LOW-SUPPORT |
| OOK | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 32.95% [19.16%, 53.39%] | 12 recordings / 12 signals; LOW-SUPPORT |
| GMSK1 | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 33.80% [12.30%, 60.69%] | 12 recordings / 12 signals; LOW-SUPPORT |
| GMSK2 | not measured [not measured, not measured] | 0.00% [0.00%, 0.00%] | not measured [not measured, not measured] | 9.20% [3.89%, 17.22%] | 12 recordings / 12 signals; LOW-SUPPORT |

### Shifted-test metrics by global SNR

| SNR bin | Metric | Estimate | Recording-level 95% CI | Recordings / signals |
|---|---|---:|---:|---|
| [-5,0) dB | YOLO precision | 70.31% | [63.78%, 77.88%] | 8 recordings / 96 signals |
| [-5,0) dB | YOLO recall | 46.88% | [41.67%, 51.04%] | 8 recordings / 96 signals |
| [-5,0) dB | YOLO mean_iou_true_positives | 80.15% | [77.90%, 82.30%] | 8 recordings / 96 signals |
| [-5,0) dB | Energy baseline precision | 70.51% | [62.67%, 77.78%] | 8 recordings / 96 signals |
| [-5,0) dB | Energy baseline recall | 57.29% | [44.79%, 67.71%] | 8 recordings / 96 signals |
| [-5,0) dB | Energy baseline mean_iou_true_positives | 77.70% | [72.31%, 81.17%] | 8 recordings / 96 signals |
| [-5,0) dB | PAM oracle-box accuracy (chance 1/7 = 14.29%) | 0.00% | [0.00%, 0.00%] | 8 recordings / 8 signals; LOW-SUPPORT |
| [-5,0) dB | Non-PAM detections classified as Others | 65.38% | [45.30%, 86.75%] | 8 recordings / 26 signals; LOW-SUPPORT |
| [-5,0) dB | End-to-end precision | 28.12% | [20.97%, 34.43%] | 8 recordings / 96 signals |
| [-5,0) dB | End-to-end recall | 18.75% | [13.54%, 22.92%] | 8 recordings / 96 signals |
| [0,5) dB | YOLO precision | 82.76% | [75.00%, 96.30%] | 4 recordings / 48 signals |
| [0,5) dB | YOLO recall | 50.00% | [41.67%, 58.33%] | 4 recordings / 48 signals |
| [0,5) dB | YOLO mean_iou_true_positives | 81.20% | [78.91%, 84.41%] | 4 recordings / 48 signals |
| [0,5) dB | Energy baseline precision | 69.64% | [66.07%, 71.43%] | 4 recordings / 48 signals |
| [0,5) dB | Energy baseline recall | 81.25% | [77.08%, 83.33%] | 4 recordings / 48 signals |
| [0,5) dB | Energy baseline mean_iou_true_positives | 81.97% | [80.73%, 83.29%] | 4 recordings / 48 signals |
| [0,5) dB | PAM oracle-box accuracy (chance 1/7 = 14.29%) | 50.00% | [0.00%, 100.00%] | 4 recordings / 4 signals; LOW-SUPPORT |
| [0,5) dB | Non-PAM detections classified as Others | 63.16% | [50.00%, 80.00%] | 4 recordings / 19 signals; LOW-SUPPORT |
| [0,5) dB | End-to-end precision | 27.59% | [22.22%, 33.33%] | 4 recordings / 48 signals |
| [0,5) dB | End-to-end recall | 16.67% | [16.67%, 16.67%] | 4 recordings / 48 signals |

### Shifted-test confusion matrices

Detector classes:

| Actual \\ Predicted | AM-DSB | AM-SSB | FM | PAM | OFDM | 2-FSK | 2-CPFSK | 4-FSK | 4-CPFSK | OOK | GMSK1 | GMSK2 | missed | background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AM-DSB | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 |
| AM-SSB | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FM | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 9 | 0 |
| PAM | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 8 | 0 |
| OFDM | 0 | 0 | 0 | 0 | 0 | 7 | 1 | 3 | 0 | 0 | 0 | 0 | 1 | 0 |
| 2-FSK | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| 2-CPFSK | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 3 | 0 | 0 | 0 | 0 | 2 | 0 |
| 4-FSK | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4-CPFSK | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 11 | 0 |
| OOK | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 |
| GMSK1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 11 | 0 |
| GMSK2 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 9 | 0 |
| missed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| background | 0 | 0 | 0 | 0 | 1 | 8 | 1 | 14 | 0 | 0 | 0 | 0 | 0 | 0 |

PAM subtype, oracle boxes:

| Actual \\ Predicted | 2-PSK | 4-PSK | 8-PSK | 16-QAM | 64-QAM | 256-QAM | Others |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-PSK | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| 4-PSK | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| 8-PSK | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| 16-QAM | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| 64-QAM | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| 256-QAM | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| Others | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Plots and measured artifacts

The metrics file preserves measured values, CIs, supports, validation thresholds, and confusion counts.
- [detector_confusion.png](results/major_project_cpu/plots/detector_confusion.png)
- [detector_confusion_shifted.png](results/major_project_cpu/plots/detector_confusion_shifted.png)
- [energy_baseline_metrics_vs_snr.png](results/major_project_cpu/plots/energy_baseline_metrics_vs_snr.png)
- [energy_baseline_metrics_vs_snr_shifted.png](results/major_project_cpu/plots/energy_baseline_metrics_vs_snr_shifted.png)
- [pam_accuracy_vs_snr.png](results/major_project_cpu/plots/pam_accuracy_vs_snr.png)
- [pam_accuracy_vs_snr_shifted.png](results/major_project_cpu/plots/pam_accuracy_vs_snr_shifted.png)
- [pam_all_classes_confusion.png](results/major_project_cpu/plots/pam_all_classes_confusion.png)
- [pam_subtype_confusion.png](results/major_project_cpu/plots/pam_subtype_confusion.png)
- [pam_subtype_confusion_shifted.png](results/major_project_cpu/plots/pam_subtype_confusion_shifted.png)
- [pam_training_curves.png](results/major_project_cpu/plots/pam_training_curves.png)
- [yolo_metrics_vs_snr.png](results/major_project_cpu/plots/yolo_metrics_vs_snr.png)
- [yolo_metrics_vs_snr_shifted.png](results/major_project_cpu/plots/yolo_metrics_vs_snr_shifted.png)

- [Complete measured metrics JSON](results/major_project_cpu/metrics.json)
- [Run summary](results/major_project_cpu/summary.txt)

## Runtime

Measured total pipeline runtime on the configured device (seconds): 2477.0320000000065.
Measured data generation runtime (seconds): 322.6992998000351.
YOLO timing probe: 124.10979409998981 seconds for 6 training batches. See [results/major_project_cpu/timing_probe.json](results/major_project_cpu/timing_probe.json) when present.
Detector training status: completed; evaluation completed: True.
The local per-segment timing target is measured in this run; see the acceptance table above.

## Interpretation

Use this reduced CPU execution to inspect the software path and measured outputs. It does not establish full-scale target performance. Investigate misses using validation data only; keep both test sets frozen.

## CLI demonstration outputs (not accuracy estimates)
The runs below use a validation-selected threshold on separate generated examples; these samples are not part of either held-out evaluation set.

| Example | Ground-truth signals | Accepted boxes | Class + IoU >= 0.5 matches | PAM detections | Subtype predictions | CLI seconds / 1M-sample segment |
|---|---:|---:|---:|---:|---:|---:|
| single_16qam | 1 | 0 | 0 | 0 | 0 | 29.652594599989243 |
| mixed_12_signal | 12 | 11 | 6 | 0 | 0 | 29.89102229999844 |

Confidence threshold: 0.21875284000000003 (selected on validation).
