# Detector and PAM classifier diagnosis

## What was measured

The published reduced CPU workflow is still the only held-out pipeline result. Its source was commit [b95c885](https://github.com/ps-narayan/wideband-signal-recognition/commit/b95c8857d3ee73838ac674db99b730bced8ad4fa), evaluated by [Actions run 40094101954](https://github.com/ps-narayan/wideband-signal-recognition/actions/runs/40094101954); the companion [tests run 40094101942](https://github.com/ps-narayan/wideband-signal-recognition/actions/runs/40094101942) passed. On that frozen test set, YOLOv5n recall and mAP@0.5 were both 0%; the energy baseline recall was 83.33%, but the two detectors were not compared at matched precision. PAM accuracy was 15.00% versus seven-class uniform chance of 14.29%. These reduced-run values remain unchanged and are not comparable to the portfolio targets.

The new diagnostics use only the eight recordings already assigned to the training split: main-000, main-001, main-002, main-003, main-006, main-008, main-009, and main-011. They contain 48 generator-labelled boxes. No validation or test recording was used. No threshold was selected from validation or test data in these memorization experiments. The main and shifted test sets remain frozen.

## The direct overfit check now passes

The successful setting is YOLOv5n at 1280 x 1280, 200 epochs, SGD with learning rate 0.01, physical and nominal batch size 4, a 10-iteration warm-up floor, and augmentation disabled. Every prediction directory was removed before direct `detect.py` inference. The report is [1280-nbs4-warmup10-200/overfit_metrics.json](results/overfit/experiments/1280-nbs4-warmup10-200/overfit_metrics.json); its training log remains local and is not committed.

| Train-only measure | Measurement | Support |
|---|---:|---:|
| Class-agnostic recall at confidence 0.001 | 100.00% (48/48) | 8 recordings / 48 signals |
| Class-aware recall at confidence 0.001 | 100.00% (48/48) | 8 recordings / 48 signals |
| Mean IoU of class-agnostic true positives | 0.8994 | 48 true positives |
| Class mAP@0.5 | 100.00% | 8 signals per class, 6 classes |
| Predictions at confidence 0.001 | 2,286, including 2,238 false positives | 8 images |
| Class-agnostic precision at confidence 0.25 | 73.85% with 100.00% recall | 48 true positives / 17 false positives |
| YOLO training time | 1,100.50 seconds | 200 epochs / 400 optimizer updates |
| Direct inference time | 11.62 seconds | 8 images |

Each of the six detector classes had measured AP@0.5 of 100% on these same training images. This is evidence that the generator labels, class map, and model can fit this training sample. It is not evidence of generalization, and the 2,238 low-confidence false positives show why a validation-selected confidence threshold is still needed in the full pipeline.

The 100-epoch run with the same 1280-pixel, batch-aligned setup and 10-iteration warm-up floor measured 74.04% mAP@0.5 and 100% class-agnostic recall. Changing only the training duration from 100 to 200 epochs raised mAP to 100% and mean IoU from 0.7647 to 0.8994. The 200-epoch setting therefore passes the training-only check, although mean IoU is not perfect.

## Cause of the earlier zero-detection failure

The strongest confirmed cause was too few optimizer updates during the pinned YOLOv5 v7 warm-up and gradient-accumulation schedule. The pinned source defaults to nominal batch size 64 and a minimum warm-up of 100 iterations. Replaying its exact loop shows:

- The earlier 100-epoch check used batch 4 with two batches per epoch. It made only 25 optimizer updates; 18 occurred during warm-up.
- The old reduced workflow used 12 training images, batch 2, and 3 epochs. It made 6 optimizer updates, all during warm-up.
- The previously configured 15-epoch medium profile would have made 10 optimizer updates, all during warm-up.

Changing only nominal batch size from the pinned default to batch-aligned size 4 at 1280 pixels increased optimizer updates from 25 to 200 over 100 epochs. Same-image recall rose from 27.08% to 100.00%, and mAP@0.5 rose from 2.85% to 29.81%. With batch alignment held fixed, lowering only the warm-up floor from 100 to 10 increased mAP from 29.81% to 74.04%. Extending that corrected run from 100 to 200 epochs increased mAP to 100.00%. These controlled training-only changes support the optimizer schedule and training duration as the causes of the overfit failure. They do not establish a held-out performance improvement.

The production training path now makes nominal batch size explicit and equal to the configured physical batch, uses a 10-iteration warm-up floor, records the calculated optimizer schedule, and restores the pinned source after each training call. Exact source markers fail closed if the pinned YOLOv5 API changes. Inference also clears each prior prediction directory before validation or test predictions.

## Checks of alternative causes

| Possible cause | Evidence | Finding |
|---|---|---|
| Label coordinates or time/frequency conversion | 48 generator-derived label rows; maximum normalized coordinate discrepancy 4.97e-9 | No mismatch found. |
| Class mapping | Generator IDs, YAML order, and checkpoint class names agree | No mismatch found. |
| Image orientation | Committed ground-truth overlays show time along columns and shifted frequency along rows | No axis swap observed. See [overlay index](results/overfit/ground_truth_boxes/index.json) and [example overlay](results/overfit/ground_truth_boxes/main-000-s00.png). |
| Anchor matching | AutoAnchor improved best possible recall from 0.958 to 1.000 on the successful run | Anchors were not the remaining blocker in this training sample. |
| Loss weights | Standard pinned YOLOv5 scaling was present: box 0.05, class 0.0375, objectness 4.0 at 1280 pixels for six classes | No loss-scaling code defect found. These weights are not independently optimized. |
| Image size and stride | At 640 pixels, 13/48 boxes had a dimension below stride 8. At 1280, all 48 boxes exceeded 8 pixels; minimum width was 11.65 px and minimum height 10.93 px. | Visibility and small raster size are real risks at 640, but they were not sufficient to explain failure after the schedule was corrected at 1280. A corrected 640 overfit comparison is still unmeasured. |
| Confidence threshold | Direct checks included threshold 0.001; the successful model had 100% recall there and 73.85% precision at 0.25 on the same images | A high threshold did not cause the original zero recall. Operational confidence must be calibrated on validation data. |
| Augmentation | Augmentation was disabled for every memorization check | It did not prevent this overfit. Effects of mosaic/mixup on the full training set remain to be measured. |
| Default epochs | The first 100-epoch run failed; after fixing the update schedule, 100 epochs reached 74.04% mAP and 200 reached 100% | Three epochs alone was not the only cause; the controlled 200-epoch result confirms duration mattered after the optimizer schedule was fixed. |

### Clean max-detection-cap sensitivity

Each cap run below used a fresh prediction directory and the same confidence threshold of 0.001. These checkpoints used the old pinned-default optimizer schedule, so the tests measure cap effects only and cannot establish whether 1280 pixels, enlarged boxes, or YOLOv5s is intrinsically better. Full per-class counts and timing are in [the cap summary](results/overfit/cap_sensitivity_summary.json); every run's JSON records that the directory was cleared.

| Checkpoint | Cap / image | Predictions saved | Same-image recall | mAP@0.5 |
|---|---:|---:|---:|---:|
| YOLOv5n, 1280 px | 100 | 800 | 6.25% | 2.60% |
| YOLOv5n, 1280 px | 300 | 2,400 | 6.25% | 2.60% |
| YOLOv5n, 1280 px | 1,000 | 8,000 | 10.42% | 2.84% |
| YOLOv5n, 1280 px | 3,000 | 23,363 | 27.08% | 2.85% |
| YOLOv5n, enlarged boxes, 640 px | 100 / 300 / 1,000 / 3,000 | 800 / 2,400 / 4,873 / 4,873 | 2.08% at every cap | 0.00% at every cap |
| YOLOv5s, 640 px | 100 / 300 / 1,000 / 3,000 | 800 / 2,400 / 8,000 / 8,673 | 4.17% / 4.17% / 10.42% / 12.50% | 0.00% at every cap |

The cap can discard useful low-confidence candidates, so the old 300-detection pipeline cap was too small for these diagnostics. Raising it does not rescue the enlarged-box or YOLOv5s checkpoint's class mAP. Clean reruns reproduce the earlier 3000-cap results; the verified clean reports are the values used here.

## PAM memorization check

The PAM CNN separately reached 100% accuracy on 24 same-training frames after 122 epochs. Those frames covered 2-PSK, 4-PSK, 8-PSK, 16-QAM, 256-QAM, and Others; 64-QAM was absent from this small sample. The classifier can fit this sample, but the prior reduced held-out accuracy of 15.00% is near the seven-class chance level of 14.29%. Generalization, class coverage, and the effect of extraction error remain unresolved.

## Next run and limits

The `medium_cpu` profile now applies the overfit-compatible image size and optimizer settings: 1280-pixel images, batch size 4, nominal batch size 4, a 10-iteration warm-up floor, and max-det 3000. It keeps the same 18 main and 4 shifted recording IDs, one segment per recording, and frozen split. It uses 135 epochs, which is 405 expected optimizer updates for the 12 training images. The run includes a one-epoch timing probe and a 210-minute pipeline budget inside a 270-minute job timeout. If time expires during training, the workflow evaluates the latest saved checkpoint and marks training as partial.

The measured local overfit run took 1,100.50 seconds for 400 optimizer updates. Scaling that local rate to 405 updates gives roughly 1,114 seconds; this is an extrapolation, not a hosted-run measurement. The timing probe will measure the actual GitHub runner. PAM training remains at 125 epochs with at most 16 frames per represented class. See [configs/medium_cpu.yaml](configs/medium_cpu.yaml), [DECISIONS.md](DECISIONS.md), and [the workflow](.github/workflows/reduced_cpu_run.yml).

The hosted medium run has not yet measured its results. The 90-main / 20-shifted-recording `milestone2.yaml` full synthetic profile, the actual DeepSig recordings, and the optional full GPU run remain unmeasured. The project still uses six detector classes rather than the paper's full twelve-class detector set. The Zenodo Rights field is unverified; no external dataset is used here.
