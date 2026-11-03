# Detector and PAM classifier diagnosis

## What was measured

The latest reduced CPU pipeline ran from source commit [7d4c108](https://github.com/ps-narayan/wideband-signal-recognition/commit/7d4c108436772f255c1013869acdb8e63978b20e) in [Actions run 41566581098](https://github.com/ps-narayan/wideband-signal-recognition/actions/runs/41566581098). The companion [tests run 41566581113](https://github.com/ps-narayan/wideband-signal-recognition/actions/runs/41566581113) passed. The model pipeline took 1,278.9997 seconds; the full Actions job completed successfully and uploaded its metrics, plots, checkpoints, generated data, and logs. The run used 12 training, 2 validation, and 4 frozen main-test recordings, plus 4 separate shifted-test recordings; the recording-ID audit found no overlap.

At the validation-calibrated operating point, the frozen main test measured YOLO class-agnostic recall 4.17% (recording-bootstrap 95% CI [0.00%, 12.50%], 4 recordings / 24 signals), mean IoU 62.06% for true positives, and class mAP@0.5 19.14% overall (CI [18.61%, 32.73%]). For global SNR >= 5 dB, mAP@0.5 was 20.50% (CI [18.60%, 40.39%], 3 recordings / 18 signals). The energy geometry baseline measured 54.17% recall (CI [25.00%, 83.33%]) and 69.24% mean IoU. Its test precision was 50.00%, versus 100.00% for YOLO, so these held-out recall figures are not a matched-test-precision comparison even though both thresholds were calibrated against the same 80% validation precision target. The PAM CNN's all-signal accuracy, including Others, was 35.00% (chance 1/7 = 14.29%; CI [15.00%, 55.00%], 4 recordings / 20 signals); PAM oracle-box subtype accuracy was 0.00% (CI [0.00%, 0.00%], 4 recordings / 4 PAM signals). Every one of these cells is low-support under the project's 30-signal rule.

The detector trained for all 135 requested epochs. PAM training requested 125 epochs but stopped after epoch 23 because validation loss failed to improve for four consecutive epochs; the best observed validation loss was 1.5791 at epoch 19, followed by 1.5890, 1.6244, 1.6697, and 1.7021. This was validation early stopping, not exhaustion of the workflow runtime. The latest run remains **reduced run: CPU, small data, not comparable to target**. The earlier run from commit b95c885 is historical evidence only and is not the current result.

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

The PAM CNN separately reached 100% accuracy on 24 same-training frames after 122 epochs. Those frames covered 2-PSK, 4-PSK, 8-PSK, 16-QAM, 256-QAM, and Others; 64-QAM was absent from this small sample. In the latest held-out run, all-signal accuracy including Others was 35.00% versus seven-class uniform chance of 14.29%, while PAM oracle-box subtype accuracy was 0.00% on four PAM signals. Both held-out measurements have low support. The training-only fit therefore does not establish subtype generalization; class coverage and the effect of extraction error remain unresolved.

## Next run and limits

The `medium_cpu` profile applied the overfit-compatible settings: 1280-pixel images, physical and nominal batch size 4, a 10-iteration warm-up floor, max-det 3000, and 135 requested detector epochs. With 12 training images and three batches per epoch, this corresponds to 405 optimizer updates. The one-epoch probe measured 22.31 seconds on the hosted runner; this is a single-epoch timing observation, not a guarantee for later epochs. The full pipeline measured 1,278.9997 seconds on that runner. See [configs/medium_cpu.yaml](configs/medium_cpu.yaml), [DECISIONS.md](DECISIONS.md), and [the workflow](.github/workflows/reduced_cpu_run.yml).

The medium run did not establish that the detector generalizes: the training-only 8-recording memorization check reached 100% recall and mAP@0.5 at confidence 0.001, while the latest held-out main-test recall was 4.17%. The low held-out recall could reflect the tiny training/validation support, the validation-selected threshold, or a train/test distribution gap; the run does not isolate which explanation is responsible. These are hypotheses, not confirmed causes. Do not tune against either test set. The 90-main / 20-shifted-recording `milestone2.yaml` full synthetic profile, the actual DeepSig recordings, and the optional full GPU run remain unmeasured. The project still uses six detector classes rather than the paper's full twelve-class detector set. The Zenodo Rights field is unverified; no external dataset is used here.
