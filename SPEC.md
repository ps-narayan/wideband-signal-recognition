# Project specification

## Objective

Given a complex I/Q recording, find each narrowband burst's time and frequency extent and assign a modulation class. For a detector output of `PAM`, filter and downconvert the extracted samples, split them into 1024-sample I/Q frames, and classify the supported subtype. The intended user is a spectrum-monitoring operator; bounding boxes can also help with interference hunting.

## Active taxonomy

Stage 1 uses 12 detector classes in this order: `AM-DSB`, `AM-SSB`, `FM`, `PAM`, `OFDM`, `2-FSK`, `2-CPFSK`, `4-FSK`, `4-CPFSK`, `OOK`, `GMSK1`, `GMSK2`. Stage 2 uses seven outputs: `2-PSK`, `4-PSK`, `8-PSK`, `16-QAM`, `64-QAM`, `256-QAM`, `Others`. `Others` is trained on non-PAM signals using the same extraction path.

## Acceptance targets

These are project targets, not expected or assumed outcomes. Each is marked met, missed, or not measured from an actual run in `RESULTS.md`.

| Metric | Acceptance target | Evaluation rule |
|---|---:|---|
| Detector recall at global SNR >= 5 dB | >= 80% | Class-agnostic one-to-one match at IoU >= 0.5, confidence calibrated on validation |
| Mean IoU for true positives at global SNR >= 5 dB | >= 0.75 | Same matched detections as above |
| Detector class mAP@0.5 at global SNR >= 5 dB | >= 60% | Class-aware average precision at IoU 0.5 |
| PAM oracle-box accuracy at global SNR >= 10 dB | >= 80% | Generator ground-truth boxes; chance is 1/7 = 14.29% |
| PAM oracle-box accuracy at global SNR [0, 1) dB | > 50% | Generator ground-truth boxes; chance is 14.29% |
| Local processing time for a 1M-sample segment | < 5 seconds | Measured on the project owner's CPU; report exact timing method |

The end-to-end signal metric requires IoU >= 0.5, a correct detector class, and a correct PAM subtype for PAM signals. Report detector-only, PAM-only, and end-to-end metrics separately. Confidence intervals resample parent recordings. Cells with fewer than 30 signals are labeled `LOW-SUPPORT`.

## Signal and image conventions

- Normalized sample rate is 1 sample/second; normalized frequency spans -0.5 to 0.5 cycles/sample.
- Training segments contain 1,000,000 complex samples. The Hann STFT uses a 512-sample window, no overlap, and 4096 FFT points. This covers 999,936 samples. Time-to-pixel mapping uses those covered samples and does not stretch the final 64 uncovered samples into the image.
- Spectrogram pixels are log magnitude, normalized by subtracting the mean and dividing by standard deviation, then resized to 640 x 640.
- Main generator minimum bandwidth is 0.025 cycles/sample and minimum duration is 100,000 samples. At 640 x 640 these are about 16 frequency pixels and 64 time pixels. They are deliberately larger than the paper's minimum visibility example to survive resizing.
- Ground-truth boxes come from generator parameters, never from the rendered image. The generator enforces frequency guard bands tied to the extraction filter transition. Main signals do not overlap in time-frequency; overlap is not part of the primary benchmark.
- Global SNR is 10 log10(total clean signal power / total added noise power) over the full I/Q segment. Also record approximate in-band SNR for each PAM signal.

## Data and split protocol

No external recordings are required. The NumPy/SciPy generator writes reproducible spectrograms, YOLO labels, and per-segment seed/parameter JSON. Training and evaluation regenerate I/Q from the seed. Deployment always reads I/Q from the supplied SigMF or IQ file.

The active configuration assigns parent recording IDs before data generation: 48 train, 12 validation, 60 main-test, and 12 distinct shifted-test recordings. The main test has 30 parent recordings at 0.5 dB global SNR and 30 at 15 dB. All signals from one parent recording stay in one split. The split audit is published with the measurements.

## Model and deployment

- Baseline: connected-component/energy proposals provide geometry only and are not a class gate.
- Detector: YOLOv5n, initialized from the configured model definition, with pinned source commit `f9578796b8a64889aa9e66dc06914d19f40d338e`.
- Extraction: convert the box to samples and normalized frequency, bandpass, shift the band center to baseband, compensate filter delay, and take 1024-sample frames.
- PAM model: compact 1-D CNN receiving two channels, I and Q.
- CLI: reads SigMF and supported raw/NumPy inputs and emits JSON plus an annotated spectrogram. It must not require generator seeds.

## Reproducibility and exclusions

Seeds, pinned package versions, configuration, split IDs, checkpoints, CI and tests are recorded. No DeepSig/Zenodo data are integrated. The generator is an educational approximation and does not claim radio-standard compliance. Fading, carrier offset beyond extraction error, phase noise, hardware impairments, quantization, AGC, sample-clock drift, and primary-layout overlap are not modeled.
