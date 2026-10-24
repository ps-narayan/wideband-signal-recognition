# Results

No reduced CPU workflow run has completed yet. All measured values, confidence intervals, support counts, confusion matrices, plots, and target statuses are **not yet measured**. The workflow will replace this page only after evaluation finishes successfully; it will not publish partial or fabricated metric tables.

The workflow uses the explicit label **reduced run: CPU, small data, not comparable to target**. It reports observed estimates and recording-level bootstrap intervals, but marks the portfolio acceptance targets **not comparable** because the data and training budget are reduced. The local processing-time target stays not measured until the owner's machine is timed.

The full GPU run in the Colab notebook is optional and remains unmeasured.

SNR is defined as 10 log10(total clean signal power / total added noise power over the full I/Q segment). Per-signal PAM in-band SNR is reported separately.

A successful run will include:

- Measured acceptance-target values and status with recording and signal support.
- Per-class and per-global-SNR detector, baseline, PAM, and end-to-end metrics with recording-level 95% confidence intervals.
- LOW-SUPPORT labels for any cell with fewer than 30 signals.
- Main and shifted-test confusion matrices and plots.
- The raw measured metrics JSON, plots, and summary under results/reduced_cpu/.
- Detector completion status, including whether a saved checkpoint was evaluated after the training budget expired.

Thresholds are selected using validation only. The main and shifted test sets stay frozen; any investigation of a missed target must use validation data.
