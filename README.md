# Wideband Signal Recognition

A small, explainable reproduction of an end-to-end wideband RF recognition system. It turns complex I/Q samples into time/frequency detections and modulation labels for spectrum monitoring and interference hunting. The measurable acceptance targets are in [SPEC.md](SPEC.md). The reduced CPU results in this README are from the GitHub-hosted CPU workflow. The optional full GPU Colab run remains unmeasured.

## Architecture

~~~mermaid
flowchart LR
    A[Real SigMF or IQ file] --> B[512 sample Hann STFT]
    B --> C[640 by 640 spectrogram]
    C --> D[Energy geometry baseline]
    C --> E[YOLOv5n detector]
    E --> F[Time and frequency boxes]
    F --> G[Resolution status and extraction]
    G --> H[Bandpass, downconvert, frame]
    H --> I[Optional PAM CNN]
    I --> J[Detections JSON]
    F --> J
    C --> K[Annotated spectrogram PNG]
~~~

Training and evaluation can regenerate I/Q from saved metadata and seeds. Deployment reads the user's input file directly and reports seed_dependent=false; it never needs generator metadata. The YOLOv5 source is pinned to commit f9578796b8a64889aa9e66dc06914d19f40d338e and checked at runtime.

## Quick start: reduced CPU run

The GitHub Actions workflow runs the tests first, then the 50-seed visibility check, the reduced detector/PAM pipeline, and evaluation. It uses a recording-disjoint split, keeps test sets fixed, and selects operating thresholds from validation only. The reduced configuration uses 18 main recordings, 4 shifted recordings, one 1M-sample segment per recording, 640 x 640 images, 3 YOLO epochs, 5 PAM epochs, and 200 recording-level bootstrap iterations. These values are in [DECISIONS.md](DECISIONS.md). This is labelled **reduced run: CPU, small data, not comparable to target**.

To start it manually, open the repository's Actions tab, choose Reduced CPU model run, select Run workflow, and confirm the main branch. It also starts on pushes to main except changes only to README.md, RESULTS.md, results/**, or docs/**. The job has a 4 hour 30 minute hard limit; the model pipeline has a 3 hour 30 minute runtime budget. Detector training stops early enough to leave up to 15 minutes for PAM fitting and 45 minutes for detector inference and evaluation. Data generation and dataset preparation reduce the detector-training allowance. If time expires during training, the job evaluates the saved detector checkpoint; the PAM classifier is evaluated with its best saved weights, or its initialized weights if no epoch fit before the limit, and the summary marks that status. If tests fail, training does not start. Before expensive steps, the workflow creates and deletes a unique temporary branch to verify that GitHub Actions can write with its built-in token. If that check fails, open repository Settings > Actions > General > Workflow permissions and select Read and write permissions. A successful evaluation commits its measured results to this repository with [skip ci]. Metrics, plots, visibility output, and logs are also available as a downloadable workflow artifact.

The reduced CPU results below and in [RESULTS.md](RESULTS.md) were written after a successful workflow evaluation. **The full GPU run in the Colab notebook remains optional and unmeasured.** The reduced CPU results are not comparable to the portfolio targets and do not replace the full run.

### Optional full GPU run

For a larger run, open [notebooks/milestone2_detector_pam_evaluation.ipynb](notebooks/milestone2_detector_pam_evaluation.ipynb) in Colab, select a T4 runtime, and run the notebook in order. It saves metrics and checkpoints to Google Drive. This optional notebook has not been run as part of the results reported here.

## Quick start: run the CLI on an IQ recording

The CLI accepts a SigMF .sigmf-meta or .sigmf-data pair, a complex-IQ .npy file, an .npz file with an iq array, or raw interleaved complex samples. SigMF's declared datatype and sample rate are read from metadata. Supported SigMF complex datatypes are cf32_le/be, cf64_le/be, ci16_le/be, cu16_le/be, ci8, and cu8; unsupported types stop with an explicit error. This is a supported subset of the [SigMF dataset specification](https://sigmf.org/sigmf-spec.pdf).

Install the same pinned runtime dependencies used by the notebook. For a local CPU machine, use the CPU PyTorch wheel index:

~~~powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements-colab.txt
python -m pip install --no-deps -e .
git clone https://github.com/ultralytics/yolov5.git yolov5_pinned
git -C yolov5_pinned checkout --detach f9578796b8a64889aa9e66dc06914d19f40d338e
git -C yolov5_pinned rev-parse HEAD
~~~

The last command must print the pinned commit above. Then run:

~~~powershell
python -m wideband_recognition.cli "capture.sigmf-meta" --detector-weights "checkpoints/yolov5n_best.pt" --pam-weights "checkpoints/pam_cnn.pt" --yolo-dir ".\yolov5_pinned" --output-dir ".\outputs\cli" --device cpu
~~~

For .npy, .npz, or raw IQ, pass --sample-rate-hz if time in seconds and frequency in Hz are wanted. Without a rate, output uses sample indices and normalized cycles/sample. frequency_range.offset_hz_from_baseband_center is the frequency offset around the complex baseband center; absolute_hz is included only when SigMF also supplies core:frequency. The detector threshold defaults to 0.25; pass the validation-calibrated threshold printed in the run's metrics.json for the measured operating point. Never select a threshold from test results.

The output folder contains detections.json and one annotated_segment_XXXXX.png per processed chunk. Each detection records confidence, detector class, time_sample_range, frequency_range, and resolution_status; time_seconds appears when a sample rate is known. The report also records processed and omitted sample counts because the 512-sample STFT leaves a short remainder outside each image. A PAM subtype is populated only when PAM weights are provided and the predicted box is resolvable and long enough for extraction. The CLI marks close boxes unresolved and skips PAM subtype inference for them.

### Reduced CPU demonstration without checkpoints

The repository includes one real CLI run on a generated 8,192-sample synthetic input using the geometry-only energy baseline. It generated 49 class-unknown proposals; those are not modulation detections, and their energy scores are not probabilities. The JSON is explicitly labeled "reduced run: CPU, small data, not comparable to target". This demonstrates file I/O, coordinate conversion and annotated-image output only. Inspect [examples/README.md](examples/README.md), [the detections JSON](examples/reduced_cpu_energy/detections.json), and [the annotated spectrogram](examples/reduced_cpu_energy/annotated_segment_00000.png).

## Results

The measured values below are from the completed reduced CPU workflow. This run is labelled **not comparable to target**; full GPU results remain optional and unmeasured. The paper's reference results are not project measurements.

<!-- REDUCED_CPU_RESULTS_START -->

**Run:** reduced run: CPU, small data, not comparable to target. Performance target status is not comparable due to reduced support. The full GPU run via the optional Colab notebook is still unmeasured.

| Acceptance metric | Target | Reduced CPU measured | 95% recording-bootstrap CI | Support | Status |
|---|---:|---:|---:|---|---|
| Detector recall, global SNR >= 5 dB | >= 80% | 0.00% | [0.00%, 0.00%] | 3 recordings / 18 signals; LOW-SUPPORT | not comparable |
| Mean IoU, global SNR >= 5 dB | >= 75% | not measured | [not measured, not measured] | 3 recordings / 18 signals; LOW-SUPPORT | not comparable |
| Class mAP@0.5, global SNR >= 5 dB | >= 60% | 0.00% | [0.00%, 0.00%] | 3 recordings / 18 signals; LOW-SUPPORT | not comparable |
| PAM oracle accuracy, global SNR >= 10 dB | >= 80% | 33.33% | [0.00%, 100.00%] | 3 recordings / 3 signals; LOW-SUPPORT | not comparable |
| PAM oracle accuracy, global SNR [0, 1) dB | > 50% | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT | not comparable |
| Owner's local processing time per 1M samples | < 5 seconds | not measured | not applicable | owner measurement pending | not comparable |

Main test support: 4 recordings / 4 segments; shifted test: 4 recordings / 4 segments.

See [RESULTS.md](RESULTS.md), [the measured metrics JSON](results/reduced_cpu/metrics.json), and [the recording split audit](results/reduced_cpu/split_audit.json) for per-class/per-SNR CIs, support, confusion matrices, split IDs, and plots. Cells under 30 signals are LOW-SUPPORT.
<!-- REDUCED_CPU_RESULTS_END -->

Per-class and per-SNR metrics, confidence intervals, support counts, confusion matrices, and plots are linked from [RESULTS.md](RESULTS.md). The [recording split audit](results/reduced_cpu/split_audit.json) lists the run's train, validation, and test recording IDs. Cells with fewer than 30 signals are marked LOW-SUPPORT. Both geometry methods use validation-calibrated operating points targeting 80% precision; YOLO did not attain that target while the energy baseline did, so this run does not provide a matched-precision comparison. Class mAP@0.5 is reported separately. Investigate misses with validation only; keep both test sets frozen.

## Final limitations

- The signal generator is synthetic and compact, not a standards-validated transmitter or a receiver emulator. There is no confirmed real DeepSig evaluation.
- The primary synthetic layout has no time-frequency overlap; overlap is not covered except as a possible future stress test.
- The shifted test is synthetic. A held-out seed from this generator does not establish real-world generalization.
- The model set is limited to the implemented detector classes. PAM merges phase/amplitude modulation for detection, then classifies a subtype from extracted I/Q. Omitted modulations are not represented.
- OOK versus narrow AM-DSB and FM versus PAM confusion are expected cases to measure and inspect.
- The model does not simulate fading, residual carrier-frequency offset beyond deliberate extraction-error tests, oscillator phase noise, IQ imbalance, DC offset, nonlinear RF hardware distortion, ADC quantization, AGC behavior, or sample-clock drift.
- The filter and classifier use the detector's estimated box; localization error can shift or clip the extracted signal and lower PAM accuracy.
- The 640-pixel resize can erase details in the smallest boxes. The minimum bandwidth and duration checks reduce this risk but do not prove detector localization succeeds.
- Raw IQ files without sample-rate metadata cannot produce time in seconds or frequency in Hz. Only the explicitly listed SigMF datatypes are supported by this reader. It accepts one complex channel and conforming sample data; multi-channel recordings, dataset header/trailer bytes, and sample discontinuities are rejected. The current implementation reads the selected file into memory before chunking it.
- The local processing-time target must be measured on the owner's hardware; a GitHub-hosted CPU or Colab T4 timing is not a substitute.
- License status for the Zenodo companion remains unverified by the project owner; no external dataset is included.

See [EXPLAINER.md](EXPLAINER.md) for interview notes and [DECISIONS.md](DECISIONS.md) for one-line design rationale. The project and dependency license notes are in [LICENSES.md](LICENSES.md).
