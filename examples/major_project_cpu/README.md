# Current CPU CLI examples

These are completed runs of the YOLOv5n + PAM CNN CLI on generated 1,000,000-sample SigMF inputs. They use the fixed confidence threshold selected from the run's validation split (0.21875284). The raw `.sigmf-data` inputs and model weights are excluded from Git; both inputs can be regenerated with the commands below after the local CPU run has produced its checkpoints.

- `16qam/detections.json` and `16qam/annotated_segment_00000.png`: one synthetic 16-QAM signal, seed 2026. The calibrated CLI accepted zero detections, so no PAM subtype was returned.
- `multi/detections.json` and `multi/annotated_segment_00000.png`: 12 synthetic signals, seed 2027. The CLI returned 11 boxes. Ten overlapped a ground-truth box at IoU >=0.5 irrespective of class; six matched both class and IoU. There were no PAM detections and no subtype predictions.

These are illustrative single-recording outputs, not accuracy estimates. The separate acceptance metrics use the recording-disjoint held-out sets in `results/major_project_cpu/`. The current CLI does not demonstrate successful 16-QAM detection or PAM subtype recognition.

To regenerate from the project root in PowerShell after a local CPU run:

~~~powershell
$env:PYTHONPATH = 'src'
$threshold = 0.21875284
python scripts/generate_synthetic_sigmf.py work/demo_16qam.sigmf-meta --layout pam16qam --seed 2026 --samples 1000000 --global-snr-db 15
python -m wideband_recognition.cli work/demo_16qam.sigmf-meta --detector-weights outputs/major_project_cpu/checkpoints/yolov5n_best.pt --pam-weights outputs/major_project_cpu/checkpoints/pam_cnn.pt --yolo-dir work/yolov5_pinned --image-size 640 --confidence-threshold $threshold --device cpu --output-dir examples/major_project_cpu/16qam
python scripts/generate_synthetic_sigmf.py work/demo_multi.sigmf-meta --layout multi --seed 2027 --samples 1000000 --global-snr-db 15
python -m wideband_recognition.cli work/demo_multi.sigmf-meta --detector-weights outputs/major_project_cpu/checkpoints/yolov5n_best.pt --pam-weights outputs/major_project_cpu/checkpoints/pam_cnn.pt --yolo-dir work/yolov5_pinned --image-size 640 --confidence-threshold $threshold --device cpu --output-dir examples/major_project_cpu/multi
~~~
