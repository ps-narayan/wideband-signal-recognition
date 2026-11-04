# Example outputs

`major_project_cpu/` contains two measured SigMF CLI runs from the current 12-class detector and seven-class PAM CNN. See its README for reproducible generation commands and the actual detection counts. At the validation-selected threshold, the 16-QAM example returned zero detections; the mixed example returned boxes but no PAM subtype. These are honest integration examples, not successful PAM recognition evidence.

`reduced_cpu_energy/` and `reduced_cpu_yolo/` are historical artifacts from the earlier six-class reduced CPU project. They are not results of the current full 12-class run. `generate_synthetic_cli_sample.py` reproduces the earlier energy-baseline smoke input.
