# Why the reduced CPU YOLOv5n detector scored zero

## Run and scope

The newest successful reduced run is [GitHub Actions run 40092609256](https://github.com/ps-narayan/wideband-signal-recognition/actions/runs/40092609256). It trained source revision 3d9dff1 and published measured files in commit 9d021b9. The newest tests run, [40092512265](https://github.com/ps-narayan/wideband-signal-recognition/actions/runs/40092512265), also passed on source revision 3d9dff1. The published-results commit changes only RESULTS.md and generated reduced-run outputs.

This diagnosis uses the detector training log, validation metrics and validation images only. The held-out test split remains frozen; none of its results were used to choose a threshold or infer the cause. The reported zero recall and zero mAP are the frozen-test outcome that prompted this investigation.

## What was measured

- The main split contained 12 training recordings, 2 validation recordings, and 4 held-out test recordings, with one 1M-sample segment per recording. The validation set had 12 signal boxes across its 2 recordings.
- YOLOv5n trained from scratch: the training command supplied an empty weights argument, 640-pixel images, batch size 2, and 3 epochs. The training log reports 12 training images and 72 box targets.
- In the validation results CSV, precision, recall, and mAP@0.5 were 0 at each of the three epochs.
- Validation inference used confidence 0.001 and logged 300 OOK predictions for each of the two validation images, reaching YOLOv5's configured max-det limit. The saved validation prediction image shows many boxes; the corresponding labels image shows the expected signal boxes.
- The validation-only 80% precision target was not attained. Calibration selected confidence threshold 1.01, with 0 true positives, 0 false positives, and 12 false negatives. That threshold therefore leaves no detections at the chosen operating point.
- The workflow's 50-seed minimum-box visibility check passed at global SNR -5 dB. The worst case was 3.992 local standard deviations above the local median. At 640 by 640, the minimum box was 5.243 by 5.120 pixels.
- YOLOv5 AutoAnchor started with best possible recall 0.958 and evolved anchors to 1.000. The anchor ratio fit passes after evolution, but a 5-pixel box is still smaller than the stride-8 feature cell.

Validation aggregate precision, recall, and mAP@0.5 stayed at zero, while low-confidence inference emitted an excessive number of OOK boxes. The evidence points to a detector that did not learn useful objectness, localization, or class separation in this run.

## Checks of likely causes

### Too little training data or too few epochs

**Evidence:** training used 12 images / 72 boxes for 3 epochs from random initialization. All three validation epochs had zero precision, recall, and mAP@0.5. The validation-only calibration also found no detections that met the requested precision target.

**Assessment:** this is the most likely explanation. It is still a hypothesis: the run is too small to distinguish an undertraining problem from other interacting limits with certainty.

### Label or coordinate conversion

**Evidence:** ground-truth boxes are created from generator parameters, then converted using the covered 999,936-sample spectrogram span. The class IDs and generated YOLO names both come from the same detector-class list. The latest CI run passed 45 tests, including the covered-sample coordinate round trip, normalized YOLO box checks, and frequency/time image-axis checks. The validation label visualization shows boxes around the generated signal regions.

**Assessment:** no label or coordinate defect was found in the code path or available validation evidence. It is not fully ruled out because the workflow artifact does not retain the raw validation label text files for a per-box numeric audit.

### Confidence threshold

**Evidence:** inference for the validation precision-recall evaluation used confidence 0.001, not the calibrated threshold 1.01. At the low threshold the model emitted up to 300 predictions per image, all named OOK. The 1.01 threshold was selected only after the validation calibration could not attain 80% precision.

**Assessment:** threshold 1.01 explains why the calibrated operating point has no detections. It does not explain the zero mAP: mAP and the per-epoch validation metrics use low-confidence candidates / a confidence sweep, and those were also zero.

### Image resize and signal visibility

**Evidence:** the 50-seed check passed its visibility rule at -5 dB, so the smallest generated signal was not erased by normalization and resize under that check. Its raster box is only about 5.24 by 5.12 pixels at 640 resolution.

**Assessment:** complete signal disappearance is not supported by the visibility test. The tiny raster box remains a plausible contributing cause because it is smaller than the detector's stride-8 feature cell; visibility does not prove that YOLO can localize or classify the box.

### Class mapping

**Evidence:** dataset labels, the YOLO dataset names, and evaluation decoding all use the same ordered detector-class tuple. The validation inference log calls the predictions OOK, consistent with the model's embedded class names; the validation label visualization shows the six expected classes.

**Assessment:** a class-order mismatch is not supported by the available evidence. The OOK-only output is more consistent with class collapse during the short training run. A hidden mismatch cannot be excluded until the raw validation prediction and label text files are retained and inspected together.

## Most likely explanation

The strongest explanation is inadequate effective training: a six-class detector was trained from scratch on only 12 recordings for 3 epochs, and validation stayed at zero throughout. The approximately 5-pixel boxes are a second plausible limitation. A high calibrated confidence threshold removed detections at the operating point, but it cannot account for the zero validation mAP observed at low confidence. No single cause is proven by this run.

## Next controlled CPU probe

The optional [10-epoch probe configuration](configs/reduced_cpu_yolo_10_epoch_probe.yaml) is now available in the manual GitHub Actions workflow. It changes only YOLO training from 3 to 10 epochs; data size, seed, split, 640-pixel images, class mapping, and 80% validation precision calibration stay the same. Push-triggered runs and the default manual option continue to use the original 3-epoch profile.

The added work is 7 epochs, or 42 additional training batches with 12 images and batch size 2. The newest complete 3-epoch pipeline took 56.17 seconds on the hosted runner. The 10-epoch wall time has not been measured, so no duration is claimed. The probe is not yet measured and does not replace the committed results. If run, it keeps threshold selection on validation and evaluates the unchanged test sets only after that selection.

If validation still has zero recall and mAP after the probe, retain the failure and investigate saved validation label/prediction text, then consider larger image resolution or larger minimum signal dimensions. The full GPU run with a larger training budget is also a sensible next experiment. Do not use either test split to select a threshold, resize, model, or epoch count.
