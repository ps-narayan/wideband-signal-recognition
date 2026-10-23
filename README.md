# Wideband Signal Recognition

A small, explainable reproduction of an end-to-end wideband RF recognition system. It turns complex I/Q samples into time/frequency detections and modulation labels for spectrum monitoring and interference hunting. The measured acceptance targets are in [SPEC.md](SPEC.md); Milestone 2 results are not yet measured because the Colab notebook has not been run.

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

## Quick start: reproduce the model

The complete Milestone 2 notebook installs pinned dependencies, runs tests and a CPU smoke path, verifies the YOLOv5 revision and visibility checks, generates data, trains, evaluates, and saves reports/checkpoints/plots to Google Drive.

1. Create a public GitHub repository and push this project using the commands in the handoff.
2. Open [notebooks/milestone2_detector_pam_evaluation.ipynb](notebooks/milestone2_detector_pam_evaluation.ipynb) in Colab and select a T4 GPU runtime.
3. Run the notebook in order and paste back the final printed summary.
4. Colab saves metrics JSON, confidence intervals, support counts, plots, confusion matrices, logs, and checkpoints in the new Drive run folder. It does not archive full I/Q arrays.
5. The notebook prints the exact run folder, shaped as /content/drive/MyDrive/wideband_signal_recognition/milestone2_<UTC timestamp>. Its model files are checkpoints/yolov5n_best.pt and checkpoints/pam_cnn.pt.
6. DeepSig and Zenodo data are not used. Read [LICENSES.md](LICENSES.md) before adding external data.

After training, put a real IQ or SigMF input in Drive, set IQ_FILE to its path, and run this in a later Colab cell while the notebook runtime is still active. This reuses the pinned project and YOLO checkout and reads the trained checkpoints directly from Drive:

~~~python
from pathlib import Path
import subprocess
import sys

IQ_FILE = Path("/content/drive/MyDrive/captures/capture.sigmf-meta")  # replace with your Drive input path
subprocess.run([
    sys.executable, "-m", "wideband_recognition.cli", str(IQ_FILE),
    "--detector-weights", str(DRIVE_OUT / "checkpoints" / "yolov5n_best.pt"),
    "--pam-weights", str(DRIVE_OUT / "checkpoints" / "pam_cnn.pt"),
    "--yolo-dir", str(YOLO_DIR),
    "--output-dir", str(DRIVE_OUT / "cli_demo"),
    "--device", "0",
], cwd=PROJECT_DIR, check=True)
~~~

The CLI writes cli_demo/detections.json and annotated spectrogram images under that same Drive run folder. If the Colab runtime has ended, reopen the notebook, mount Drive, clone/install the repository as in its first cells, set DRIVE_OUT to the completed run folder printed earlier, set YOLO_DIR to /content/yolov5_pinned_f9578796, and run the same cell.

The notebook's results are not reproduced or estimated in this repository. Since the notebook has not yet been run, [RESULTS.md](RESULTS.md) and the results table below are intentionally unfilled.

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
python -m wideband_recognition.cli "C:\path\to\capture.sigmf-meta" --detector-weights "C:\path\to\milestone2\checkpoints\yolov5n_best.pt" --pam-weights "C:\path\to\milestone2\checkpoints\pam_cnn.pt" --yolo-dir ".\yolov5_pinned" --output-dir ".\outputs\cli" --device cpu
~~~

For .npy, .npz, or raw IQ, pass --sample-rate-hz if time in seconds and frequency in Hz are wanted. Without a rate, output uses sample indices and normalized cycles/sample. frequency_range.offset_hz_from_baseband_center is the frequency offset around the complex baseband center; absolute_hz is included only when SigMF also supplies core:frequency. The detector threshold defaults to 0.25; pass the validation-calibrated threshold printed in the run's metrics.json for the measured operating point. Never select a threshold from test results.

The output folder contains detections.json and one annotated_segment_XXXXX.png per processed chunk. Each detection records confidence, detector class, time_sample_range, frequency_range, and resolution_status; time_seconds appears when a sample rate is known. The report also records processed and omitted sample counts because the 512-sample STFT leaves a short remainder outside each image. A PAM subtype is populated only when PAM weights are provided and the predicted box is resolvable and long enough for extraction. The CLI marks close boxes unresolved and skips PAM subtype inference for them.

### Reduced CPU demonstration without checkpoints

The repository includes one real CLI run on a generated 8,192-sample synthetic input using the geometry-only energy baseline. It generated 49 class-unknown proposals; those are not modulation detections, and their energy scores are not probabilities. The JSON is explicitly labeled "reduced run: CPU, small data, not comparable to target". This demonstrates file I/O, coordinate conversion and annotated-image output only. Inspect [examples/README.md](examples/README.md), [the detections JSON](examples/reduced_cpu_energy/detections.json), and [the annotated spectrogram](examples/reduced_cpu_energy/annotated_segment_00000.png).

## Results

Milestone 2 has not been run, so all measured cells are **not yet measured** and target statuses are pending. No performance values have been inferred or copied from the paper.

| Measurement | Target | Measured value | Recording-level 95% CI | Recordings / signals | Status |
|---|---:|---:|---:|---:|---|
| YOLO class-agnostic recall, global SNR >= 5 dB | >= 80% | not yet measured | not yet measured | not yet measured | Pending run |
| YOLO mean IoU of true positives, global SNR >= 5 dB | >= 0.75 | not yet measured | not yet measured | not yet measured | Pending run |
| YOLO class mAP@0.5, global SNR >= 5 dB | >= 60% | not yet measured | not yet measured | not yet measured | Pending run |
| PAM oracle-box subtype accuracy, global SNR >= 10 dB | >= 80% | not yet measured | not yet measured | not yet measured | Pending run |
| PAM subtype accuracy, 0 dB bin | > 50% | not yet measured | not yet measured | not yet measured | Pending run |
| End-to-end precision / recall by global SNR | Report | not yet measured | not yet measured | not yet measured | Pending run |
| CPU processing time per 1M samples on owner's machine | < 5 s | not yet measured | n/a | n/a | Owner measurement pending |

**Low support:** any per-class or per-SNR cell with fewer than 30 signals must be explicitly marked low-support; show its recording and signal support next to the interval. Full per-class, per-SNR, shifted-test, confusion-matrix, and bootstrap results belong in [RESULTS.md](RESULTS.md). The paper's reference results are not this project's measurements.

**Acceptance status:** every performance target is **not yet measured**, so none can be marked met or missed. When the actual run summary is available, fill only its numbers and intervals. For any missed target, report the miss and a validation-only next experiment; keep the test set frozen and do not tune it.

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
- The local processing-time target must be measured on the owner's hardware; a Colab T4 timing is not a substitute.
- License status for the Zenodo companion remains unverified by the project owner; no external dataset is included.

See [EXPLAINER.md](EXPLAINER.md) for interview notes and [DECISIONS.md](DECISIONS.md) for one-line design rationale. The project and dependency license notes are in [LICENSES.md](LICENSES.md).



