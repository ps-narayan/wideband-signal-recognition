"""Validate and tabulate clean max-det sensitivity memorization reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(root: Path) -> dict:
    variants = {}
    for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        rows = []
        for result_dir in sorted(variant_dir.glob("maxdet-*")):
            metrics_path = result_dir / "overfit_metrics.json"
            if not metrics_path.is_file():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            yolo = metrics["yolo"]
            if not yolo.get("prediction_directory_cleared_before_inference"):
                raise ValueError(f"Prediction folder was not verified clean in {metrics_path}")
            threshold = yolo["same_image_metrics_by_confidence_threshold"]["0.001"]
            rows.append({
                "max_detections_per_image": yolo["same_image_max_detections_per_image"],
                "predictions_at_confidence_0_001": yolo["prediction_count_at_conf_0_001"],
                "class_agnostic": threshold["class_agnostic"],
                "class_aware": threshold["class_aware"],
                "class_map50": threshold["class_map50"],
                "inference_seconds": yolo["same_image_inference_seconds"],
                "prediction_directory_cleared_before_inference": True,
                "recordings": metrics["recordings"],
                "signals": metrics["label_and_coordinate_checks"]["label_rows"],
            })
        if rows:
            variants[variant_dir.name] = sorted(rows, key=lambda row: row["max_detections_per_image"])
    return {
        "scope": "same-image training-only cap sensitivity; no validation or test recordings",
        "confidence_threshold": 0.001,
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/overfit/cap_sensitivity"))
    parser.add_argument("--output", type=Path, default=Path("results/overfit/cap_sensitivity_summary.json"))
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: len(rows) for name, rows in result["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
