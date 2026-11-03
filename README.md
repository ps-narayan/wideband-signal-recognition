# Wideband Signal Recognition

The synthetic pipeline runs end to end, including a checkpoint-backed YOLOv5 + PAM CLI. The reduced CPU detector generalized poorly on held-out synthetic data: recall was 37.50%, below the energy baseline's 54.17%; PAM oracle-box subtype accuracy was 0.00% on four signals. The trained detector still has six classes. A separate SigMF converter writes the paper's 12-class label map, but the Zenodo/DeepSig recordings have not been downloaded, used for training, or evaluated. Full-scale and 12-class results are not measured.

Latest measured reduced CPU run: YOLOv5n trained for 135 requested epochs; all-SNR class-agnostic recall was 37.50% (recording-bootstrap CI [25.00%, 50.00%]; 4 recordings / 24 signals; LOW-SUPPORT), and mAP@0.5 at global SNR >= 5 dB was 28.78% (CI [20.00%, 64.02%]; 3 recordings / 18 signals; LOW-SUPPORT). The energy geometry baseline had higher held-out recall (54.17% vs 37.50%; CI [25.00%, 83.33%]; 4 recordings / 24 signals; LOW-SUPPORT), though test precision differed (energy 50.00%; YOLO 60.00%). PAM accuracy including `Others` was 35.00% (chance 1/7 = 14.29%; CI [15.00%, 55.00%]; 4 recordings / 20 signals; LOW-SUPPORT); PAM oracle-box subtype accuracy was 0.00% (chance 14.29%; CI [0.00%, 0.00%]; 4 recordings / 4 PAM signals; LOW-SUPPORT). The 12-class and full-scale results are not measured. Reduced CPU numbers are not comparable to the portfolio targets. See [DIAGNOSIS.md](DIAGNOSIS.md) and [RESULTS.md](RESULTS.md).

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
    H --> I[PAM CNN]
    I --> J[Detections JSON]
    F --> J
    C --> K[Annotated spectrogram PNG]
~~~

Training and evaluation can regenerate I/Q from saved metadata and seeds. Deployment reads the user's input file directly and reports seed_dependent=false; it never needs generator metadata. The YOLOv5 source is pinned to commit f9578796b8a64889aa9e66dc06914d19f40d338e and checked at runtime.

## Quick start: reduced CPU run

The GitHub Actions workflow runs tests first, then the 50-seed visibility check, the reduced detector/PAM pipeline, and evaluation. It keeps recording-disjoint splits and selects operating thresholds on validation only. The completed `medium_cpu` run used 18 main and 4 shifted recordings with one 1M-sample segment per recording, 1280 x 1280 images, and 135 YOLO epochs at batch size 4. PAM training requested 125 epochs and stopped at epoch 23 after validation loss failed to improve for four consecutive epochs. It used at most 16 PAM training frames per observed class and 500 recording-level bootstrap iterations. A one-epoch training-only timing probe ran before fitting. The held-out measurements are published below and in [RESULTS.md](RESULTS.md); see [DIAGNOSIS.md](DIAGNOSIS.md) for the measured outcome and limits. Results remain labelled **reduced run: CPU, small data, not comparable to target**. A manual workflow run can still select the original 3-epoch profile or the isolated 10-epoch validation probe.

To start it manually, open the repository's Actions tab, choose Reduced CPU model run, select Run workflow, and confirm the main branch. It also starts on pushes to main except changes only to README.md, RESULTS.md, results/**, or docs/**. The job has a 4 hour 30 minute hard limit; the model pipeline has a 3 hour 30 minute runtime budget. Detector training stops early enough to leave up to 15 minutes for PAM fitting and 45 minutes for detector inference and evaluation. Data generation and dataset preparation reduce the detector-training allowance. If time expires during training, the job evaluates the saved detector checkpoint; the PAM classifier is evaluated with its best saved weights, or its initialized weights if no epoch fit before the limit, and the summary marks that status. If tests fail, training does not start. Before expensive steps, the workflow creates and deletes a unique temporary branch to verify that GitHub Actions can write with its built-in token. If that check fails, open repository Settings > Actions > General > Workflow permissions and select Read and write permissions. A successful evaluation commits its measured results to this repository with [skip ci]. Metrics, plots, visibility output, and logs are also available as a downloadable workflow artifact.

The reduced CPU results below and in [RESULTS.md](RESULTS.md) were written after a successful workflow evaluation. **The full GPU run in the Colab notebook remains optional and unmeasured.** The reduced CPU results are not comparable to the portfolio targets and do not replace the full run.

### Optional full GPU run

For a larger run, open [notebooks/milestone2_detector_pam_evaluation.ipynb](notebooks/milestone2_detector_pam_evaluation.ipynb) in Colab, select a T4 runtime, and run the notebook in order. It saves metrics and checkpoints to Google Drive. This optional notebook has not been run as part of the results reported here.

## Quick start: run the CLI on an IQ recording

The CLI accepts a SigMF .sigmf-meta or .sigmf-data pair, a complex-IQ .npy file, an .npz file with an iq array, or raw interleaved complex samples. SigMF's declared datatype and sample rate are read from metadata. Supported SigMF complex datatypes are cf32_le/be, cf64_le/be, ci16_le/be, cu16_le/be, ci8, and cu8; unsupported types stop with an explicit error. This is a supported subset of the [SigMF dataset specification](https://sigmf.org/sigmf-spec.pdf).

The CLI defaults to YOLO+PAM and requires both trained checkpoints. `--detector-only` explicitly disables subtype classification, and `--backend energy` explicitly selects the class-agnostic geometry baseline. `--detector-class-map` accepts a JSON class list when a checkpoint uses a custom order; without it, the CLI uses the six-class order of the published reduced checkpoint.

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
python -m wideband_recognition.cli "capture.sigmf-meta" --detector-weights "checkpoints/yolov5n_best.pt" --pam-weights "checkpoints/pam_cnn.pt" --yolo-dir ".\yolov5_pinned" --output-dir ".\outputs\cli" --image-size 1280 --max-detections 3000 --device cpu
~~~

Use `--image-size` to match the checkpoint's training resolution; the medium CPU checkpoint uses 1280. For .npy, .npz, or raw IQ, pass `--sample-rate-hz` if time in seconds and frequency in Hz are wanted. Without a rate, output uses sample indices and normalized cycles/sample. `frequency_range.offset_hz_from_baseband_center` is the frequency offset around the complex baseband center; absolute_hz is included only when SigMF also supplies core:frequency. The detector threshold defaults to 0.25; pass the validation-calibrated threshold printed in the run's metrics.json for the measured operating point. Never select a threshold from test results.

The output folder contains detections.json and one annotated_segment_XXXXX.png per processed chunk. Each detection records confidence, detector class, time_sample_range, frequency_range, and resolution_status; time_seconds appears when a sample rate is known. The report also records processed and omitted sample counts because the 512-sample STFT leaves a short remainder outside each image. The default cascade requires PAM weights and records a subtype only when the detected PAM box can be resolved and extracted. The CLI marks close boxes unresolved and skips subtype inference for them.

A checkpoint-backed CLI run completed on one generated 1M-sample, 16-QAM SigMF file with global SNR 10 dB and the validation-calibrated confidence threshold. It produced zero accepted detections, so it emitted no PAM subtype. This is an integration smoke example, not an accuracy measurement; see [the saved JSON and annotated image](examples/reduced_cpu_yolo/README.md).

### Prepare annotated SigMF recordings for the 12-class detector

The importer reads each complex-I/Q segment in turn, builds ground-truth boxes from SigMF annotations, and writes spectrograms, YOLO labels, recording/segment manifests, a dataset YAML, and a class-name JSON file. Assign splits per parent recording before conversion, keeping every segment of a recording in one split:

~~~powershell
python scripts/prepare_sigmf_dataset.py --meta "data/recording.sigmf-meta" --output "outputs/sigmf_dataset" --recording-id "recording-001" --split train
~~~

Repeat for each recording, choosing `train`, `val`, or `test` at the recording level. Explicit PSK/QAM/APSK labels map to `PAM`. The converter requires GMSK1 or GMSK2 labels and rejects generic `GMSK`, since the paper separates those detector classes by spectrogram shape. This is a data-preparation path only: the existing training workflow still uses its six-class synthetic profile, and no external recordings have been converted in a project run.

### Reduced CPU demonstration without checkpoints

The repository includes one real CLI run on a generated 8,192-sample synthetic input using the geometry-only energy baseline. It generated 49 class-unknown proposals; those are not modulation detections, and their energy scores are not probabilities. The JSON is explicitly labeled "reduced run: CPU, small data, not comparable to target". This demonstrates file I/O, coordinate conversion and annotated-image output only. Inspect [examples/README.md](examples/README.md), [the detections JSON](examples/reduced_cpu_energy/detections.json), and [the annotated spectrogram](examples/reduced_cpu_energy/annotated_segment_00000.png).

## Results

The measured values below are from the completed reduced CPU workflow. This run is labelled **not comparable to target**; full GPU results remain optional and unmeasured. The paper's reference results are not project measurements.

<!-- REDUCED_CPU_RESULTS_START -->

**Run:** reduced run: CPU, small data, not comparable to target. Performance target status is not comparable due to reduced support. The full GPU run via the optional Colab notebook is still unmeasured.

| Acceptance metric | Target | Reduced CPU measured | 95% recording-bootstrap CI | Support | Status |
|---|---:|---:|---:|---|---|
| Detector recall, global SNR >= 5 dB | >= 80% | 38.89% | [16.67%, 50.00%] | 3 recordings / 18 signals; LOW-SUPPORT | not comparable |
| Mean IoU, global SNR >= 5 dB | >= 75% | 78.57% | [66.47%, 89.27%] | 3 recordings / 18 signals; LOW-SUPPORT | not comparable |
| Class mAP@0.5, global SNR >= 5 dB | >= 60% | 28.78% | [20.00%, 64.02%] | 3 recordings / 18 signals; LOW-SUPPORT | not comparable |
| PAM oracle accuracy (chance 1/7 = 14.29%), global SNR >= 10 dB | >= 80% | 0.00% | [0.00%, 0.00%] | 3 recordings / 3 signals; LOW-SUPPORT | not comparable |
| PAM oracle accuracy (chance 1/7 = 14.29%), global SNR [0, 1) dB | > 50% | not measured | [not measured, not measured] | 0 recordings / 0 signals; LOW-SUPPORT | not comparable |
| Owner's local processing time per 1M samples | < 5 seconds | not measured | not applicable | owner measurement pending | not comparable |

Main test support: 4 recordings / 4 segments; shifted test: 4 recordings / 4 segments.

See [RESULTS.md](RESULTS.md), [the measured metrics JSON](results/reduced_cpu/metrics.json), and [the recording split audit](results/reduced_cpu/split_audit.json) for per-class/per-SNR CIs, support, confusion matrices, split IDs, and plots. Cells under 30 signals are LOW-SUPPORT.
<!-- REDUCED_CPU_RESULTS_END -->

Per-class and per-SNR metrics, confidence intervals, support counts, confusion matrices, and plots are linked from [RESULTS.md](RESULTS.md). The [recording split audit](results/reduced_cpu/split_audit.json) lists the run's train, validation, and test recording IDs. Cells with fewer than 30 signals are marked LOW-SUPPORT. YOLO and the geometry baseline both attained their validation-calibrated 80% precision target, but measured test precision differed (60% versus 50%), so the held-out recall comparison is not a matched-test-precision comparison. Class mAP@0.5 is reported separately. Investigate misses with validation only; keep both test sets frozen.

## Final limitations

- The signal generator is synthetic and compact, not a standards-validated transmitter or a receiver emulator. There is no confirmed real DeepSig evaluation.
- The primary synthetic layout has no time-frequency overlap; overlap is not covered except as a possible future stress test.
- The shifted test is synthetic. A held-out seed from this generator does not establish real-world generalization.
- The published reduced detector checkpoint and its measured run use six labels: AM-DSB, FM, OFDM, 2-FSK, OOK, and PAM. The SigMF preparation route supports the paper's 12-class label order, but those additional classes have not been trained or evaluated, and the synthetic benchmark training workflow remains six-class. PAM oracle-box subtype accuracy was 0.00% on four PAM test signals, so no subtype-performance claim is warranted.
- OOK versus narrow AM-DSB and FM versus PAM confusion are expected cases to measure and inspect.
- The model does not simulate fading, residual carrier-frequency offset beyond deliberate extraction-error tests, oscillator phase noise, IQ imbalance, DC offset, nonlinear RF hardware distortion, ADC quantization, AGC behavior, or sample-clock drift.
- The filter and classifier use the detector's estimated box; localization error can shift or clip the extracted signal and lower PAM accuracy.
- The 640-pixel reduced profile can erase details in the smallest boxes. The medium CPU diagnostic profile resizes to 1280 pixels. Minimum bandwidth and duration checks reduce visibility risk but do not prove detector localization succeeds.
- Raw IQ files without sample-rate metadata cannot produce time in seconds or frequency in Hz. Only the explicitly listed SigMF datatypes are supported by this reader. It accepts one complex channel and conforming sample data; multi-channel recordings, dataset header/trailer bytes, and sample discontinuities are rejected. The current implementation reads the selected file into memory before chunking it.
- The local processing-time target must be measured on the owner's hardware; a GitHub-hosted CPU or Colab T4 timing is not a substitute.
- The Zenodo record's Rights field was verified as CC BY-SA 4.0 and its four archives total 57,472,189,381 bytes. No data were downloaded because the complete record exceeds local free disk; the importer has not been run on that record. DeepSig access/licensing remains unverified and no real-data results exist.

See [EXPLAINER.md](EXPLAINER.md) for interview notes and [DECISIONS.md](DECISIONS.md) for one-line design rationale. The project and dependency license notes are in [LICENSES.md](LICENSES.md).
