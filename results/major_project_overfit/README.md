# Training-only memorization check

These 8 spectrograms use generator-derived labels; the first four have boxes drawn in color, with class names in `ground_truth_boxes/index.json`. The detector check uses only the listed training recordings, never validation or test data. The PAM memorization check is included in the JSON. Exact measured outcomes are in `overfit_metrics.json`.
