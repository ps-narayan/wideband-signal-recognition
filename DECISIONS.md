# Design decisions

- Use a self-contained seeded NumPy/SciPy generator so the project can run without external data downloads, unverified data terms, or large disk requirements.
- Activate all twelve detector labels in the paper's specified order so every synthetic recording exercises the full detector class map.
- Train a seven-output PAM CNN for the six requested PSK/QAM subclasses plus `Others`; use `Others` samples from non-PAM classes through the same extraction chain.
- Use a 0.025 cycles/sample minimum bandwidth and a 100,000-sample minimum duration. On a 640 x 640 raster these become about 16 by 64 pixels, reducing the risk of disappearing boxes after resize. This is a deliberate scale deviation.
- Keep the energy detector as a geometry baseline, not a class gate, because it cannot supply modulation labels.
- Assign parent recording IDs before data generation and keep each parent's segments in one split. Fix low- and high-SNR test strata in the configuration and calibrate confidence on validation only.
- Store spectrograms, labels, and per-recording seed/parameters. Regenerate I/Q only when training/evaluation needs it; deployment reads the actual user file.
- Pin YOLOv5 source to `f9578796b8a64889aa9e66dc06914d19f40d338e` and pin Python dependencies to make results reproducible.
- Use recording-level bootstrap intervals because signals from one recording are not independent. Mark any cell with fewer than 30 signals as low support.
- Keep the primary synthetic layout free of time-frequency overlap. Model overlapping transmissions only in a future stress test.
- Do not claim standards-compliant transmissions, real-world generalization, APSK subtype support, or performance not shown by measured run artifacts.
- The Zenodo record Rights field is unverified and no external dataset is used. The 53.55 GiB download is not a project requirement.
# 2026-11-03: CLI passes its validation-selected confidence threshold directly to YOLO NMS, avoiding work on proposals it will discard while preserving the accepted set.
