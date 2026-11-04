# Diagnosis of the synthetic-only 12-class / 7-class run

## Scope and data policy

The project now generates its own reproducible SigMF-style wideband I/Q data. It does not download the 53.55 GiB Zenodo/DeepSig data. The Zenodo record's Rights field is unverified; this project makes no license claim about it. DeepSig terms are also not verified. The full GPU run and results from external recordings are not measured.

The active detector has the full 12-label taxonomy: AM-DSB, AM-SSB, FM, PAM, OFDM, 2-FSK, 2-CPFSK, 4-FSK, 4-CPFSK, OOK, GMSK1, and GMSK2. The I/Q CNN has seven labels: 2-PSK, 4-PSK, 8-PSK, 16-QAM, 64-QAM, 256-QAM, and Others. The generator implements those six PAM subtypes; APSK subtype classification is not implemented.

## What was measured

A local CPU run used 48 training, 12 validation, and 60 main-test recordings, plus 12 separate shifted-parameter test recordings. The main test contains 720 signals, 60 for each detector class. The low-SNR target cell contains 30 recordings / 30 PAM signals at 0.5 dB. The >=10 dB PAM cell contains 30 recordings / 30 PAM signals at 15 dB. The split audit confirms train, validation, main test, and shifted test recording IDs do not overlap. Thresholds were calibrated on validation only; the test sets were not used to select them. Recording-level bootstrap intervals and support counts are in [the measured results](RESULTS.md) and [`results/major_project_cpu/`](results/major_project_cpu/).

The full pipeline ran for 2,477.032 seconds on CPU (41 minutes 17.032 seconds), slightly over the 40-minute budget. YOLOv5n completed 80/80 epochs. The PAM CNN completed 42/60 epochs before validation early stopping. Inference uses the validation-selected YOLO confidence threshold 0.21875284.

On the frozen main test at global SNR >=5 dB, YOLO recall was 63.06% (95% CI 59.30% to 66.94%, 30 recordings / 360 signals), mAP@0.5 was 41.27% (95% CI 40.95% to 45.11%, same support), and mean IoU of true positive boxes was 0.8789 (95% CI 0.8718 to 0.8868, same support). Thus recall and mAP missed their targets; mean IoU met its target. Main-test class-agnostic recall across all SNRs was 56.53% for YOLO and 81.39% for the energy geometry baseline. These are each evaluated at their own validation-calibrated operating point, not a matched-precision test comparison.

PAM oracle-box accuracy was 46.67% at global SNR >=10 dB (95% CI 30.00% to 63.33%, 30 recordings / 30 signals) and 36.67% in the 0-to-1 dB cell (95% CI 20.00% to 53.33%, 30 recordings / 30 signals). Both exceed uniform seven-class chance of 14.29% at the point estimate, but both miss the requested accuracy targets. All-signal CNN accuracy including Others was 65.83% (95% CI 63.75% to 67.92%, 60 recordings / 720 signals). Detector-box PAM subtype accuracy was not measured because there were no usable correctly matched PAM detections.

The measured local CLI processing times were 29.653 seconds per 1,000,000-sample 16-QAM segment and 29.891 seconds per 1,000,000-sample mixed segment. The <5-second target was missed. These timings include file loading, spectrogram rendering, CPU detector inference, extraction/classification, and annotated-image writing, but not JSON serialization.

## Checks of likely implementation failures

A train-only check used eight training recordings with all 12 labels. It found 96 YOLO label rows, all 12 classes present, matching checkpoint names, matching generator parameters, and maximum normalized coordinate error 4.9962e-9. Time is represented by image columns and shifted frequency by rows. Every training box exceeded YOLO's stride-8 minimum dimension, and AutoAnchor reported best possible recall 1.000. This makes a class-order, coordinate-conversion, image-orientation, or anchor-fit defect unlikely for this generated sample.

The same-image YOLO result at 640 pixels after 100 epochs was not a successful memorization check: class-agnostic recall at confidence 0.001 was 100%, but class-aware recall was 89.58%, class mAP@0.5 was 48.42%, and class-agnostic precision was 3.15% with 3,043 predictions. Geometry could be localized on the training images, but the model produced many duplicates and did not fit the class predictions well. This is evidence against a simple box-coordinate failure; it does not isolate why class learning remained weak.

The PAM CNN reached 100% on its small training-only frame set, but that does not predict held-out signal accuracy. The held-out oracle-box results above are below target, and several subtype-specific cells have only ten signals each, so those class results are low-support.

The deployed CLI completed both generated SigMF examples with the validation-selected threshold. The one-signal 16-QAM example returned zero detections. The 12-signal mixed example returned 11 boxes: 10 matched a ground-truth box at IoU >=0.5 without requiring the class to match, while six matched both class and IoU. It returned no PAM detections, so there was no CLI subtype prediction. This is a real negative example, not a successful PAM demonstration. An exploratory 0.001-threshold run also did not recover the 16-QAM signal as PAM; its best box IoU was 0.3401 and its predicted class was OOK. That exploratory output is not the calibrated example and is not used in the acceptance metrics.

## Most likely explanation and limits

The confirmed fact is that the implementation runs and the label pipeline passes direct checks, but the learned detector and PAM classifier do not meet their generalization targets. A single root cause is not established. A small synthetic training set, random detector initialization, limited within-class waveform variety, and a CPU-constrained training schedule are plausible contributors. The classifier's perfect fit on a tiny training frame set but substantially lower oracle-box test accuracy indicates a generalization gap, but does not identify which of those contributors caused it.

The detector's box geometry is substantially better than its class-aware detection performance. This pattern supports investigating class confusion and training diversity before changing coordinate code. It does not prove that simply adding more data will solve the problem. The energy detector is a geometry-only baseline and cannot replace modulation classification.

## What to try next

1. Keep both test sets frozen. Use validation recordings to inspect per-class confusion and confidence calibration.
2. Expand training variation within each class, especially pulse shaping, symbol timing, phase, signal duration/bandwidth, and SNR; regenerate a new dataset version and retain recording-disjoint splits.
3. Inspect PAM subtype confusion and extraction offsets. Compare oracle boxes with detector boxes after the detector has usable PAM recall.
4. If compute is available, run a larger GPU experiment with pretrained YOLO weights and a larger training set. Report it as a new run; do not overwrite these measurements.
5. Rework CLI latency only with measured profiling. Passing the selected threshold into NMS did not materially change the observed ~30-second segment time, so this small optimization does not meet the speed target.

Do not claim the acceptance targets were achieved. The current measured statuses are two detector misses, one geometry pass, two PAM misses, and one local speed miss. Full GPU and real-radio performance remain unmeasured.
