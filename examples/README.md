# Saved CLI example

This directory contains a reproducible sample input generator and one completed CLI invocation.

The sample has 8,192 complex samples. The CLI ran the energy geometry baseline on CPU and produced 49 proposals for one spectrogram segment. The proposals use detector class unknown; their energy scores are not probabilities, and this run has no trained YOLO detector or PAM classifier. The JSON includes the required notice: reduced run: CPU, small data, not comparable to target.

Files:

- generate_synthetic_cli_sample.py creates the local NumPy input. By default it writes the ignored file work/synthetic_cli_sample.npy with seed 31415.
- reduced_cpu_energy/detections.json is the saved CLI output.
- reduced_cpu_energy/annotated_segment_00000.png is the corresponding annotated spectrogram.

To reproduce from the repository root in PowerShell:

~~~powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements-colab.txt
python -m pip install --no-deps -e .
python examples/generate_synthetic_cli_sample.py
python -m wideband_recognition.cli work/synthetic_cli_sample.npy --backend energy --device cpu --segment-samples 8192 --output-dir examples/reduced_cpu_energy
~~~

This demonstration checks file loading, spectrogram generation, box-coordinate conversion, JSON writing and image annotation. Its proposal count is not a detection metric and cannot be compared with any acceptance target.

## Trained YOLO+PAM CLI example

`reduced_cpu_yolo/` contains a real run of the trained reduced-CPU YOLOv5n + PAM CNN cascade on a generated SigMF recording. The input contains one 16-QAM signal, 1,000,000 samples, and measured global SNR 10.0004 dB (requested 10 dB). Inference used the checkpoint's 1280-pixel image size and the validation-calibrated confidence threshold of 0.2663. It completed and wrote JSON plus an annotated spectrogram, but **accepted zero detections**, so it made no PAM subtype prediction. This is a CLI integration example and a negative sample outcome, not an accuracy result.

The checkpoint came from the successful reduced CPU Actions artifact for workflow run `41640721078`; model weights are not stored in this repository. Generate the same SigMF input with `generate_synthetic_sigmf.py`, then run the CLI using the extracted `yolov5n_best.pt` and `pam_cnn.pt` checkpoints and the pinned YOLOv5 checkout. The generated `.sigmf-data` file stays local under the ignored `work/` directory.

~~~powershell
python -m pip install --no-deps -e .
python examples/generate_synthetic_sigmf.py
python -m wideband_recognition.cli work/synthetic-sigmf/synthetic.sigmf-meta `
  --detector-weights checkpoints/yolov5n_best.pt `
  --pam-weights checkpoints/pam_cnn.pt `
  --yolo-dir yolov5_pinned `
  --output-dir outputs/cli-yolo `
  --image-size 1280 --confidence-threshold 0.2663 --device cpu
~~~

The checkpoint remains six-class, and this example does not demonstrate successful detection or subtype recognition. The full 12-class detector and full-scale performance remain unmeasured.
