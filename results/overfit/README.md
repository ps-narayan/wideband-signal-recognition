# Historical training-only memorization diagnostics

This folder contains earlier six-class experiments and is retained for audit history. The active full-taxonomy, eight-recording train-only check is [results/major_project_overfit](../major_project_overfit/README.md), and the current held-out results are in [RESULTS.md](../../RESULTS.md).

All memorization checks use eight recordings from the training split. Validation and test recordings are excluded.

- The initial 640-pixel overfit check failed. Label coordinates, generated classes, saved checkpoint class names, and the human-readable ground-truth overlays agree.
- The clean cap-sensitivity checks compare 100, 300, 1,000, and 3,000 detections per image on the earlier 1280-pixel YOLOv5n, enlarged-box YOLOv5n, and YOLOv5s checkpoints. Every pass starts with a deleted prediction directory. The measurements are summarized in [cap_sensitivity_summary.json](cap_sensitivity_summary.json) and [cap_sensitivity/README.md](cap_sensitivity/README.md).
- The 1280-pixel YOLOv5n run with batch-aligned updates and a 10-iteration warm-up floor reached 74.04% same-image mAP@0.5 after 100 epochs.
- Extending that same training recipe to 200 epochs produced 100.00% class-agnostic recall and 100.00% class mAP@0.5, with mean IoU 0.8994, on 8 recordings / 48 signals. It emitted 2,238 false positives at confidence 0.001. Exact metrics are in [1280-nbs4-warmup10-200](experiments/1280-nbs4-warmup10-200/README.md).
- The PAM CNN fit its 24 same-training-frame sample to 100% after 122 epochs. This sample did not include 64-QAM.

The passing detector run proves that the model can fit this small synthetic training sample. It does not measure generalization or replace the held-out results. See [DIAGNOSIS.md](../../DIAGNOSIS.md) for the controlled schedule comparisons and remaining hypotheses.
