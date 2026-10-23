# Results and acceptance tracker

**Run summary status: not provided.** The user reports that the Colab run completed, but the supplied message contained only a placeholder. No measurements, confidence intervals, or support counts have been inserted or inferred. The paper's reported numbers are not project results. Every blank estimate/support cell below means **not yet measured** and is reserved for the actual notebook summary.

SNR means the synthetic generator's global ratio:
10 log10(total clean signal power / total added noise power across the full I/Q segment).
PAM rows also need the per-signal approximate in-band SNR when that field is available.

## Acceptance targets

| Metric | Target | Measured value | Recording-level 95% CI | Recordings / signals | Status |
|---|---:|---:|---:|---:|---|
| YOLO class-agnostic recall at global SNR >= 5 dB | >= 0.80 | not yet measured | not yet measured | not yet measured | Awaiting summary |
| YOLO mean IoU of true positives at global SNR >= 5 dB | >= 0.75 | not yet measured | not yet measured | not yet measured | Awaiting summary |
| YOLO mAP@0.5 at global SNR >= 5 dB | >= 0.60 | not yet measured | not yet measured | not yet measured | Awaiting summary |
| PAM oracle-box accuracy at global SNR >= 10 dB | >= 0.80 | not yet measured | not yet measured | not yet measured | Awaiting summary |
| PAM oracle-box accuracy in the 0-to-1 dB bin | > 0.50 | not yet measured | not yet measured | not yet measured | Awaiting summary |
| Processing time per 1M samples on owner's local machine | < 5 s | not yet measured | n/a | n/a | Owner measurement pending |

All targets are not yet measured. Once the actual run summary is available, classify each target as met or missed using the reported values. If a target misses, report it as-is and propose the next experiment using training or validation data only. Keep the test data and operating-point selection frozen.

## Main recording-disjoint test: overall

| Measurement | Estimate | Recording-level 95% CI | Recordings / signals | Support note |
|---|---:|---:|---:|---|
| YOLO precision at validation-calibrated 80% target operating point | ____ | ____ | ____ / ____ | ____ |
| YOLO class-agnostic recall at matched precision | ____ | ____ | ____ / ____ | ____ |
| YOLO mean IoU of true positives | ____ | ____ | ____ / ____ | ____ |
| YOLO class mAP@0.5, all SNR | ____ | ____ | ____ / ____ | ____ |
| Energy geometry baseline precision at matched operating point | ____ | ____ | ____ / ____ | ____ |
| Energy geometry baseline class-agnostic recall | ____ | ____ | ____ / ____ | ____ |
| Energy geometry baseline mean IoU | ____ | ____ | ____ / ____ | ____ |
| PAM subtype accuracy with oracle boxes | ____ | ____ | ____ / ____ | ____ |
| PAM subtype accuracy with detector boxes | ____ | ____ | ____ / ____ | ____ |
| Non-PAM detections labeled Others by PAM classifier | ____ | ____ | ____ / ____ | ____ |
| Joint end-to-end precision | ____ | ____ | ____ / ____ | ____ |
| Joint end-to-end recall | ____ | ____ | ____ / ____ | ____ |

Energy baseline versus YOLO comparison uses the class-agnostic recall and mean IoU at matched 80% precision. The operating thresholds come from validation. Class mAP@0.5 is reported separately.

## Main test: per detector class

Mark a row low-support when it has fewer than 30 ground-truth signals. Keep recording and signal counts beside every confidence interval.

| Detector class | Precision | Recall | Mean IoU | AP@0.5 | 95% CIs (per metric) | Recordings / signals | Support |
|---|---:|---:|---:|---:|---|---:|---|
| AM-DSB | ____ | ____ | ____ | ____ | ____ | ____ / ____ | ____ |
| FM | ____ | ____ | ____ | ____ | ____ | ____ / ____ | ____ |
| OFDM | ____ | ____ | ____ | ____ | ____ | ____ / ____ | ____ |
| 2-FSK | ____ | ____ | ____ | ____ | ____ | ____ / ____ | ____ |
| OOK | ____ | ____ | ____ | ____ | ____ | ____ / ____ | ____ |
| PAM | ____ | ____ | ____ | ____ | ____ | ____ / ____ | ____ |

## Main test: global SNR groups

Use the exact SNR-bin keys produced by metrics.json. Each cell needs its own recording bootstrap interval and support count. Mark support low when signal count is below 30.

| Global SNR bin | YOLO recall | YOLO mean IoU | PAM oracle accuracy | PAM detector-box accuracy | End-to-end precision / recall | 95% CIs | Recordings / signals | Support |
|---|---:|---:|---:|---:|---:|---|---:|---|
| [-5,0) dB | ____ | ____ | ____ | ____ | ____ / ____ | ____ | ____ / ____ | ____ |
| [0,5) dB | ____ | ____ | ____ | ____ | ____ / ____ | ____ | ____ / ____ | ____ |
| [5,10) dB | ____ | ____ | ____ | ____ | ____ / ____ | ____ | ____ / ____ | ____ |
| [10,15) dB | ____ | ____ | ____ | ____ | ____ / ____ | ____ | ____ / ____ | ____ |
| [15,20) dB | ____ | ____ | ____ | ____ | ____ / ____ | ____ | ____ / ____ | ____ |
| Outside the listed bins, if present | ____ | ____ | ____ | ____ | ____ / ____ | ____ | ____ / ____ | ____ |

## PAM subtype and classifier diagnostics

| Measurement | Estimate / matrix | Recording-level 95% CI | Recordings / signals | Support |
|---|---:|---:|---:|---|
| Oracle-box accuracy by PAM subtype | ____ | ____ | ____ / ____ | ____ |
| Detector-box accuracy by PAM subtype | ____ | ____ | ____ / ____ | ____ |
| False detector-class boxes predicted as Others, overall | ____ | ____ | ____ / ____ | ____ |
| False detector-class boxes predicted as Others, by non-PAM class | ____ | ____ | ____ / ____ | ____ |
| PAM subtype confusion matrix (oracle boxes) | ____ | ____ | ____ / ____ | ____ |
| All-signal confusion matrix with Others | ____ | ____ | ____ / ____ | ____ |
| Approximate PAM in-band SNR count / mean / min / max | ____ | n/a | ____ / ____ | ____ |

Expected confusions to inspect include OOK versus narrow AM-DSB and FM versus PAM. Do not treat these expectations as measured errors.

## Shifted-parameter test

This test has a different bandwidth distribution and global SNR placement. Keep it separate from the main test and do not use it to select thresholds.

| Measurement | Estimate | Recording-level 95% CI | Recordings / signals | Support |
|---|---:|---:|---:|---|
| YOLO precision / recall / mean IoU at main validation threshold | ____ / ____ / ____ | ____ | ____ / ____ | ____ |
| YOLO mAP@0.5 | ____ | ____ | ____ / ____ | ____ |
| PAM oracle / detector-box accuracy | ____ / ____ | ____ | ____ / ____ | ____ |
| Joint end-to-end precision / recall | ____ / ____ | ____ | ____ / ____ | ____ |
| Non-PAM-to-Others rate | ____ | ____ | ____ / ____ | ____ |
| Detector and PAM confusion matrices | ____ | ____ | ____ / ____ | ____ |

## Artifacts to retain from the run

- Metrics JSON and printed summary: ____
- Per-class and per-SNR tables with confidence intervals/support: ____
- Detector and PAM confusion matrices: ____
- Accuracy-versus-SNR plots with global SNR definition in title: ____
- Visibility check reports and worst-case images: ____
- Colab runtime and hardware record: ____
- Owner's local CLI sample run and timing: ____
