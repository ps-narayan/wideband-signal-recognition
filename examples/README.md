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
