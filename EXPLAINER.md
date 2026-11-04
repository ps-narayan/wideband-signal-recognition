# Explain it in plain language

## What does the project do?

A wideband recording can contain many narrower transmissions at different times and frequencies. The first model looks at a spectrogram like an image and draws a box around each transmission. It also assigns one of twelve modulation labels. When that label is `PAM`, a second model reads the corresponding raw I/Q waveform and picks one of six supported PSK/QAM types or `Others`.

This is useful for spectrum monitoring because an operator needs to know what occupied a band, when it happened, and how wide it was. The same boxes can guide an interference investigation.

## Why two stages?

A spectrogram makes time and frequency location visible, but PSK and QAM types often have similar average brightness patterns. The detector therefore merges the supported PSK/QAM families into one `PAM` label. The I/Q classifier sees symbol-level phase and amplitude patterns that the spectrogram hides.

The detector emits a time-frequency rectangle. The extraction code turns its edges back into sample numbers and normalized frequencies, filters the requested band, shifts the center toward zero frequency, and takes fixed 1024-sample I/Q frames. If the box is wrong, the filter can cut off useful signal or leave a residual frequency rotation, which can reduce PAM accuracy.

## Why these image settings?

The generator uses one-million-sample segments and a Hann window of 512 samples with 4096 FFT points and no overlap. This keeps the same broad spectrogram idea as the paper. The 640 x 640 detector image cannot preserve arbitrarily small boxes, so the synthetic generator uses at least 0.025 cycles/sample of bandwidth and 100,000 samples of duration. Those dimensions cover about 16 frequency pixels and 64 time pixels after resizing. This is a measured project deviation, not the paper's exact scale.

The STFT covers 999,936 samples from a one-million-sample input. Pixel mapping uses that covered interval, leaving the final 64 samples outside the image coordinate map. Labels come from the generator's time and frequency parameters, not from image inspection.

## What does SNR mean here?

Global SNR compares the total clean signal power with the total added noise power over the whole wideband recording. A single PAM signal's in-band SNR can be different because its bandwidth contains only part of the total noise and because several signals contribute to total signal power. The run logs both.

As noise increases, detections, box overlap, and modulation accuracy should generally get worse. If a lower-SNR split suddenly looks much better, check SNR measurement, split assignment, normalization, and labels before drawing a conclusion.

## How are train and test kept separate?

The split unit is a parent recording ID. Every rendered segment and derived I/Q frame from that parent stays in one split. Thresholds are selected with validation data only. The held-out main-test and shifted-test recordings are used only for final evaluation. Confidence intervals resample recordings because several signals from one recording are related.

## Why keep an energy baseline?

A simple energy/connected-component baseline suggests bright regions that may be transmissions. It can measure rough geometry, but it cannot name modulation classes. The detector comparison therefore reports class-agnostic recall and mean IoU at the validation-selected precision operating point, while class mAP is measured separately.

## What can the project claim?

Only values in `RESULTS.md` that are backed by the local metrics JSON and run artifacts. The included generator is synthetic and reproducible; this does not demonstrate real-world receiver performance. No result from the paper is presented as a project measurement.

## What is not modeled?

The primary layout has no time-frequency overlap. The generator also omits multipath fading, carrier-frequency error beyond extraction tests, oscillator phase noise, IQ imbalance, DC offset, nonlinear RF hardware, ADC quantization, AGC, and sample-clock drift. It does not generate APSK for the second-stage classifier, and the waveforms are not certified as standards-compliant. These are future extensions, not hidden assumptions.
