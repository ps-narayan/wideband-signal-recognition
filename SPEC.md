# Project specification

## Objective

Build a small, explainable reproduction of the wideband signal-recognition
pipeline. Input is complex I/Q data. Output is a list of occupied time/frequency
boxes and modulation labels, with a second-stage PAM subtype when applicable.
The portfolio use case is spectrum monitoring; the same output can support
interference hunting.

## Acceptance criteria

| Area | Measurable criterion | Design consequence |
|---|---|---|
| Detector | On a recording-disjoint synthetic test set at global SNR >= 5 dB: recall >= 80%, mean IoU >= 0.75, and mAP@0.5 >= 60% for selected classes. | Class labels are always required, so the energy/connected-component baseline is a geometry floor and fallback, not a gate. Compare energy baseline and YOLOv5n class-agnostic recall and mean IoU at matched 80% precision, with operating points chosen on validation data. Judge YOLO class mAP@0.5 separately. |
| PAM subtype | Accuracy >= 80% at global SNR >= 10 dB and >50% at 0 dB. Report by class and SNR. | Use 1024x2 I/Q frames and a compact CNN first. Report oracle-box and detector-box accuracy separately. |
| End to end | A signal is correct only when one-to-one matched IoU >= 0.5, detector class is correct, and PAM subtype is correct for PAM. Report end-to-end precision/recall by global SNR. | Keep the joint result separate from detector-only and PAM-only scores. |
| Runtime | <5 seconds per 1M-sample segment on the user's machine. | Measure on CPU and Colab T4 separately; do not infer local speed from Colab. |
| Raster visibility | The smallest generated box has bandwidth >=0.008 cycles/sample and duration >=8192 samples. At 640x640 it occupies at least 5 pixels on both axes. Run 50 random seeds at -5 dB global SNR and report worst local contrast in standard deviations above the local median. | If the worst case is below 1 local standard deviation, pause and review raising the minimum bandwidth or duration before detector training. |
| Geometry | Time/frequency-to-pixel round trips recover values within one source sample or one output pixel. Time mapping uses exactly 999,936 covered samples for 1,000,000 input samples with 512-sample non-overlapping windows. | Do not map the final 64 uncovered samples into the image. |
| Separation | Main generator contains no time-frequency-overlapping boxes. Simultaneous signals have edge separation >= 9/16 times the larger box bandwidth. Closer cases are labeled unresolved. | Use the same guard rule as the extraction-filter transition. Overlap is optional stress data only. |
| Data split | Use 90 parent recording IDs and 900 1M-sample segments; keep all segments from a recording together. Split parent IDs 60/10/20 for train/validation/test. | Split at recording level before any segment-level sampling or augmentation. Assert no ID crosses splits. |
| Shift robustness | Evaluate both held-out seeds within the training parameter ranges and a second test set with different bandwidth and SNR-placement ranges. | Report the two tests separately. |
| Confidence | Every per-class and per-SNR result includes a recording-level bootstrap confidence interval and counts of recordings/signals. Cells with fewer than 30 signals are marked low-support. | Resample recordings, not individual boxes or frames. |
| Storage | Store spectrograms, labels, and JSON generator parameters/seeds; do not retain I/Q for all 900 segments. Regenerate I/Q for training/evaluation. | The deployed CLI reads I/Q from the user-supplied file and never requires a seed. On one saved synthetic IQ file, assert seed-regenerated and file-input extraction produce the same frames. |

## Fixed signal and image conventions

- Normalized sampling rate is Fs=1 Sps; normalized frequencies are
  in [-0.5, 0.5) cycles/sample. Convert to Hz only when file metadata gives a
  physical sample rate.
- STFT array is time-by-frequency; transpose it for the image so horizontal is
  time and vertical is frequency, with positive frequencies at the top. Use a
  Hann window, 512 samples, no overlap, 4096-point FFT, log magnitude, then
  global image mean removal and standard-deviation normalization.
- A 1,000,000-sample segment yields 1,953 windows and covers 999,936 samples.
- Resize directly to 640x640. This distorts the two axes and is a deliberate
  small-model deviation; visibility is measured after resize.
- Minimum bandwidth is 0.008 cycles/sample and minimum duration is 8192 samples.
- Global SNR is 10 log10(total clean signal power / total added noise power)
  measured over the full I/Q segment. Log approximate in-band SNR per PAM
  signal as well; the two values need not match.
- Main synthetic data has no overlapping time-frequency boxes. A separate,
  optional stress set may contain overlaps. Main placement enforces the guard
  band below; closer simultaneous pairs are unresolved.
- Training parameter profile: bandwidth log-uniform in [0.008, 0.08] cycles/sample
  and global SNR uniform in [-5, 20] dB. The shifted test profile is bandwidth
  uniform in [0.02, 0.12] and SNR uniform in [-5, 5] dB. These are project
  design ranges, not facts attributed to the paper.

## Extraction rule

For box bandwidth B, use a complex-baseband low-pass FIR after downconversion.
Set passband half-width to 5B/8 (wanted half-bandwidth B/2 plus a B/8 guard),
then a transition of 7B/16; stopband begins at 17B/16 from the estimated center.
Thus a neighboring box must be separated edge-to-edge by at least 9B/16 for
that crop. For two simultaneous boxes require a gap of at least
9/16 * max(B1, B2). The generator uses a larger conservative margin when time
edges may jitter by five percent: `e * (B1 + B2) + (9/16) * (1 + 2e) * max(B1, B2)`,
where `e=0.05`. If the time gap is larger than `e * (duration1 + duration2)`,
the boxes remain time-disjoint under that jitter and need no frequency guard.
Use an odd, linear-phase Kaiser-windowed FIR with 1111 taps
at the minimum bandwidth (0.008), beta=6.0, passband ripple <=0.1 dB and
stopband attenuation >=60 dB. Compensate its 555-sample group delay. For wider
boxes the transition is wider; verify the measured response for each design.
Any closer pair is unresolved, not silently treated as isolated.

## Model/data scope

- Detector classes: AM-DSB, FM, OFDM, 2-FSK, OOK, and merged PAM.
- PAM subtypes: 2/4/8-PSK, 16/64/256-QAM, and Others. Others initially uses
  AM-DSB, FM, 2-FSK, and OOK under the same extraction procedure. The synthetic
  detector set has one example of each detector class per segment, so detector
  class counts are balanced by construction; this does not reproduce the paper's original imbalance. PAM subtype skew is handled with inverse-frequency loss,
  and PAM-only recording undersampling is not needed for this synthetic set.
- PAM generation randomizes carrier phase, symbol timing, and pulse-shaping
  roll-off. Main training starts synthetic-only. DeepSig is optional after
  access and license checks; the user will verify the Zenodo companion's size
  and license.
- Energy detection is a class-agnostic geometry floor/fallback. Labels are
  always required, so it is not a gate to YOLOv5n. Compare class-agnostic recall
  and mean IoU at matched 80% precision (choose each operating threshold using
  validation data, then freeze it for test). If a model cannot attain that
  validation precision, report its curve and nearest operating point. Judge
  YOLOv5n class mAP@0.5 separately.

## Evaluation/reporting

Report detector precision, recall, mAP@0.5, mean IoU of true positives,
per-class metrics and confusion matrices; PAM accuracy/confusion matrices;
oracle-box versus detector-box PAM accuracy; non-PAM-to-PAM-classifier Others
rate; and end-to-end precision/recall. All are broken out by global SNR where
applicable. Include recording and signal counts, recording-level bootstrap
confidence intervals, and low-support flags. Label every SNR plot with the
global SNR definition and record PAM in-band SNR estimates.

## Coupling experiments

- Compare PAM accuracy with oracle boxes and detector boxes to measure how
  localization affects extraction and subtype classification.
- Log the six detector-class counts after merging all PSK/QAM/APSK-style PAM
  signals into one detector label; report per-class results to expose imbalance.
- Verify recording-ID disjointness and, for demonstration only, compare against
  a random-segment split to quantify leakage optimism.
- If real data becomes available, compare synthetic-only with synthetic
  pretraining plus real fine-tuning on recording-held-out real data; report FM
  separately as well as aggregate results.
- If PAM-only undersampling is used, log retained PAM signal/frame counts by
  subtype and compare PAM accuracy before and after undersampling.
- Send non-PAM detector boxes to the PAM classifier and report the fraction
  predicted as Others, both overall and by non-PAM class.

## Deliberate limitations

The initial generator does not model fading, frequency offset beyond deliberate
extraction-error tests, oscillator phase noise, IQ imbalance, DC offset,
nonlinear hardware distortion, ADC quantization, AGC behavior, or sample-clock
drift. It does not establish performance on real receivers or on omitted
modulation classes. These limitations must remain visible in the README and
demo.

## Milestone acceptance

1. Week 1: spec, reproducible synthetic generator, STFT renderer, physical box
   conversion, guard-band checks, and 50-seed visibility report.
2. Week 2: energy baseline and YOLOv5n detector; autoanchor logs inspected;
   PAM CNN and training curves.
3. Week 3: extraction filter, seed/file path equivalence, evaluation metrics,
   recording bootstrap, two test distributions, and SNR plots.
4. Week 4: SigMF/IQ CLI, JSON detections, annotated spectrogram, README, and
   measured results/limitations.

Each milestone stops for the user's run results before the next begins.
