# Training-only memorization check

These 8 spectrograms use generator-derived labels. To avoid duplicate images in the repository, the shared visual ground-truth examples are in [the overlay index](../../ground_truth_boxes/index.json). The detector check uses only the listed training recordings, never validation or test data. This detector-only variant skipped the PAM check. Exact measured outcomes are in `overfit_metrics.json`.

