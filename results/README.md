# Results index

- `major_project_cpu/` contains the current synthetic-only run: the full 12-class detector, seven-label PAM CNN, metrics, recording-level bootstrap intervals, support counts, split audit, summary, timing probe, confusion/accuracy plots, and generated run manifest.
- `major_project_overfit/` contains the current training-only 12-class YOLO and seven-class PAM memorization checks, plus generator-derived ground-truth overlays. These are not held-out performance results.
- `reduced_cpu/` and `overfit/` are historical reduced-scope artifacts from the earlier six-class experiments. They are retained for provenance and are not the current project results.

No file in this directory contains external DeepSig or Zenodo recordings. The 53.55 GiB download is not required; the current data source is the seeded synthetic generator. The Zenodo record's Rights field remains unverified.
