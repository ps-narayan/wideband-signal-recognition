"""Publish measured artifacts from the self-contained 12-class CPU run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

try:  # Support both `python scripts/publish_major_project_results.py` and package imports in tests.
    from .publish_reduced_results import build_results, build_split_audit, normalize_summary, normalize_zero_detection_precision
except ImportError:  # pragma: no cover - exercised by the direct-script entry point.
    from publish_reduced_results import build_results, build_split_audit, normalize_summary, normalize_zero_detection_precision


START = "<!-- MAJOR_PROJECT_RESULTS_START -->"
END = "<!-- MAJOR_PROJECT_RESULTS_END -->"


def _pct(value):
    return "not measured" if value is None else f"{float(value) * 100:.2f}%"


def _ci(values):
    if not values or len(values) != 2 or any(value is None for value in values):
        return "not measured"
    return f"[{_pct(values[0])}, {_pct(values[1])}]"


def _support(value):
    if not value:
        return "not measured"
    low = "; LOW-SUPPORT" if value.get("low_support") else ""
    return f"{value.get('recordings', 'n/a')} recordings / {value.get('signals', 'n/a')} signals{low}"


def build_cli_section(metrics: dict) -> str:
    demos = metrics.get("cli_examples", {})
    examples = demos.get("examples", {})
    if not examples:
        return ""
    lines = [
        "## CLI demonstration outputs (not accuracy estimates)",
        "The runs below use a validation-selected threshold on separate generated examples; these samples are not part of either held-out evaluation set.",
        "",
        "| Example | Ground-truth signals | Accepted boxes | Class + IoU >= 0.5 matches | PAM detections | Subtype predictions | CLI seconds / 1M-sample segment |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in examples.items():
        lines.append(
            f"| {name} | {row.get('ground_truth_signals', 'not measured')} | "
            f"{row.get('accepted_detections', 'not measured')} | "
            f"{row.get('class_and_iou50_matches', 'not measured')} | "
            f"{row.get('pam_detections', 'not measured')} | "
            f"{row.get('pam_subtype_predictions', 'not measured')} | "
            f"{row.get('cli_seconds_per_segment', 'not measured')} |"
        )
    lines.extend(["", f"Confidence threshold: {demos.get('confidence_threshold', 'not measured')} (selected on validation).", ""])
    return "\n".join(lines)


def build_readme_section(metrics: dict) -> str:
    targets = [
        ("Detector recall, global SNR >= 5 dB", "recall_at_global_snr_ge_5_db", ">= 80%"),
        ("Mean IoU, global SNR >= 5 dB", "mean_iou_at_global_snr_ge_5_db", ">= 0.75"),
        ("Class mAP@0.5, global SNR >= 5 dB", "map50_at_global_snr_ge_5_db", ">= 60%"),
        ("PAM oracle-box accuracy, global SNR >= 10 dB (chance 1/7 = 14.29%)", "pam_oracle_accuracy_global_snr_ge_10_db", ">= 80%"),
        ("PAM oracle-box accuracy, global SNR [0, 1) dB (chance 1/7 = 14.29%)", "pam_oracle_accuracy_near_0_db_0_to_1", "> 50%"),
        ("Local processing time per 1M-sample segment", "processing_time_per_segment_local_seconds", "< 5 seconds"),
    ]
    lines = [
        START,
        f"**Measured run:** {metrics.get('run_label', 'not measured')}. Device: {metrics.get('device', 'not measured')}. "
        "Synthetic data only; no DeepSig or Zenodo recordings were downloaded or used.",
        "",
        "| Acceptance target | Threshold | Measured | Recording-level 95% CI | Support | Status |",
        "|---|---:|---:|---:|---|---|",
    ]
    for label, key, target in targets:
        row = metrics.get("acceptance_targets", {}).get(key, {})
        measured = row.get("measured")
        if key == "processing_time_per_segment_local_seconds":
            display = "not measured" if measured is None else f"{float(measured):.3f} s"
            confidence = "not applicable"
            support = row.get("measurement_scope", "not measured")
        else:
            display = _pct(measured)
            confidence = _ci(row.get("ci95"))
            support = _support(row.get("support"))
        status = "not measured" if measured is None else ("met" if row.get("pass") else "missed")
        lines.append(f"| {label} | {target} | {display} | {confidence} | {support} | {status} |")
    training = metrics.get("training_status", {})
    timing = metrics.get("runtime_seconds_pipeline", {})
    lines.extend([
        "",
        f"Detector training: {training.get('detector', 'not measured')} ({training.get('detector_epochs_completed', 'not measured')} completed / {training.get('detector_epochs_requested', 'not measured')} requested epochs). "
        f"PAM training: {training.get('pam', 'not measured')} ({training.get('pam_epochs_completed', 'not measured')} completed / {training.get('pam_epochs_requested', 'not measured')} requested epochs).",
        f"Measured full pipeline runtime: {timing.get('total', 'not measured')} seconds; data generation: {timing.get('data_generation', 'not measured')} seconds.",
        "The target statuses use measured point estimates. CIs and recording/signal support are shown for uncertainty; any cell below 30 signals is low-support.",
    ])
    cli_examples = metrics.get("cli_examples", {}).get("examples", {})
    if cli_examples:
        compact = "; ".join(
            f"{name}: {row.get('accepted_detections', 'not measured')} boxes, "
            f"{row.get('pam_detections', 'not measured')} PAM boxes, "
            f"{row.get('pam_subtype_predictions', 'not measured')} subtype outputs"
            for name, row in cli_examples.items()
        )
        lines.append(f"CLI smoke examples (not acceptance metrics; validation-selected threshold): {compact}.")
    lines.append(END)
    return "\n".join(lines)


def publish(run_dir: Path, repo: Path) -> dict:
    source = run_dir / "results" / "metrics.json"
    manifest_path = run_dir / "results" / "dataset_manifest.json"
    if not source.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Run metrics and dataset manifest are required before publishing.")
    metrics = normalize_zero_detection_precision(json.loads(source.read_text(encoding="utf-8")))
    metrics["results_title"] = "Synthetic 12-Class CPU Run Results"
    detector_csv = run_dir / "checkpoints" / "yolo" / "milestone2" / "results.csv"
    if detector_csv.is_file():
        with detector_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        metrics.setdefault("training_status", {})["detector_epochs_completed"] = max(0, len(rows) - 1)
    dest = repo / "results" / "major_project_cpu"
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "plots").mkdir(parents=True)
    metrics["metrics_path"] = "results/major_project_cpu/metrics.json"
    metrics_path = dest / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_audit = build_split_audit(manifest)
    if not split_audit["main_splits_disjoint"] or not split_audit["shifted_test_disjoint_from_main"]:
        raise ValueError("Refusing to publish because recording IDs cross split boundaries.")
    counts = metrics.get("split_counts_recordings", {})
    if any(counts.get(split) != count for split, count in split_audit["main_recording_counts"].items()):
        raise ValueError("Published split counts disagree with measured run metadata.")
    (dest / "split_audit.json").write_text(json.dumps(split_audit, indent=2) + "\n", encoding="utf-8")
    (dest / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for path in (run_dir / "plots").glob("*.png"):
        shutil.copy2(path, dest / "plots" / path.name)
    timing = run_dir / "results" / "yolo_timing_probe.json"
    if timing.is_file():
        shutil.copy2(timing, dest / "timing_probe.json")
    summary = run_dir / "results" / "summary.txt"
    if summary.is_file():
        safe = []
        root = str(repo.resolve())
        runtime = metrics.get("acceptance_targets", {}).get("processing_time_per_segment_local_seconds", {})
        cli_examples = metrics.get("cli_examples", {}).get("examples", {})
        normalized_summary = normalize_summary(summary.read_text(encoding="utf-8"), metrics)
        for line in normalized_summary.splitlines():
            if line.startswith("Output folder:"):
                line = "Output folder: results/major_project_cpu"
            elif line.startswith("Metrics JSON:"):
                line = "Metrics JSON: results/major_project_cpu/metrics.json"
            elif line.startswith("processing_time_per_segment_local_seconds:") and runtime.get("measured") is not None:
                status = "PASS" if runtime.get("pass") else "FAIL"
                measured = float(runtime["measured"])
                runs = ", ".join(f"{float(value):.3f}" for value in runtime.get("local_measurements_seconds", []))
                line = (
                    f"processing_time_per_segment_local_seconds: {status}; target={runtime.get('target')}; "
                    f"measured_max={measured:.6f}; per_example_seconds=[{runs}]; "
                    "local CPU SigMF CLI timing; see metrics.json for scope."
                )
            else:
                line = line.replace(root, ".")
            safe.append(line)
        if cli_examples:
            safe.extend(["", "CLI examples (validation-selected confidence threshold; not test-set metrics):"])
            for name, row in cli_examples.items():
                safe.append(
                    f"  {name}: truths={row.get('ground_truth_signals')}; accepted_detections={row.get('accepted_detections')}; "
                    f"class_and_iou50_matches={row.get('class_and_iou50_matches')}; "
                    f"PAM_detections={row.get('pam_detections')}; subtype_predictions={row.get('pam_subtype_predictions')}; "
                    f"seconds_per_segment={row.get('cli_seconds_per_segment')}"
                )
        (dest / "summary.txt").write_text("\n".join(safe) + "\n", encoding="utf-8")
    else:
        (dest / "summary.txt").write_text("Run summary not present; see metrics.json.\n", encoding="utf-8")

    from publish_reduced_results import build_results
    results = build_results(metrics, repo, split_audit)
    results += "\n" + build_cli_section(metrics)
    (repo / "RESULTS.md").write_text(results, encoding="utf-8")
    readme = repo / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError(f"README must contain {START} and {END} markers.")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    readme.write_text(before + build_readme_section(metrics) + after, encoding="utf-8")
    return split_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    audit = publish(args.run_dir.resolve(), args.repo_root.resolve())
    print(json.dumps({
        "results": str(args.repo_root.resolve() / "RESULTS.md"),
        "artifact_dir": "results/major_project_cpu",
        "main_splits_disjoint": audit["main_splits_disjoint"],
        "shifted_test_disjoint": audit["shifted_test_disjoint_from_main"],
    }, indent=2))


if __name__ == "__main__":
    main()
