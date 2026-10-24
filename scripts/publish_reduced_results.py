"""Publish only measured outputs from a completed reduced CPU run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

START = "<!-- REDUCED_CPU_RESULTS_START -->"

END = "<!-- REDUCED_CPU_RESULTS_END -->"


def pct(value):
    return "not measured" if value is None else f"{float(value) * 100:.2f}%"


def ci(row):
    values = (row or {}).get("ci95")
    return "not measured" if not values or len(values) != 2 else f"[{pct(values[0])}, {pct(values[1])}]"


def support(row):
    if not row:
        return "not measured"
    suffix = "; LOW-SUPPORT" if row.get("low_support") else ""
    return f"{row.get('recordings', 'n/a')} recordings / {row.get('signals', 'n/a')} signals{suffix}"


def metric(label, row, key="estimate"):
    row = row or {}
    return f"| {label} | {pct(row.get(key))} | {ci(row)} | {support(row)} |"


def matrix(labels, values):
    if not labels or not values:
        return "Not measured."
    out = ["| Actual \\\\ Predicted | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    out += [f"| {name} | " + " | ".join(str(int(v)) for v in row) + " |"
            for name, row in zip(labels, values)]
    return "\n".join(out)


def build_results(m, repo):
    main, shifted = m["main_test"], m["shifted_test"]
    yolo, energy, pam = main["yolo"], main["energy_baseline"], main["pam"]
    target_info = [
        ("recall_at_global_snr_ge_5_db", "Detector recall, global SNR >= 5 dB", ">= 80%"),
        ("mean_iou_at_global_snr_ge_5_db", "Mean IoU, global SNR >= 5 dB", ">= 0.75"),
        ("map50_at_global_snr_ge_5_db", "Class mAP@0.5, global SNR >= 5 dB", ">= 60%"),
        ("pam_oracle_accuracy_global_snr_ge_10_db", "PAM oracle-box accuracy, global SNR >= 10 dB", ">= 80%"),
        ("pam_oracle_accuracy_near_0_db_0_to_1", "PAM oracle-box accuracy, global SNR [0, 1) dB", "> 50%"),
        ("processing_time_per_segment_local_seconds", "Processing time per 1M samples on owner's machine", "< 5 seconds"),
    ]
    lines = [
        "# Reduced CPU Run Results", "",
        f"Run label: {m.get('run_label', 'reduced run: CPU, small data, not comparable to target')}",
        "Comparison: reduced run: CPU, small data, not comparable to target. Target statuses are not comparable; this run does not establish whether the portfolio targets are met or missed.",
        f"Device: {m.get('device', 'not measured')}",
        f"YOLOv5 source commit: {m.get('yolo', {}).get('commit', 'not measured')}",
        f"Detector training status: {m.get('training_status', {}).get('detector', 'not measured')}; checkpoint: {m.get('training_status', {}).get('detector_checkpoint_used', 'not measured')}; requested epochs: {m.get('training_status', {}).get('detector_epochs_requested', 'not measured')}.",
        f"PAM training status: {m.get('training_status', {}).get('pam', 'not measured')}; epochs completed: {m.get('training_status', {}).get('pam_epochs_completed', 'not measured')} of {m.get('training_status', {}).get('pam_epochs_requested', 'not measured')}.",
        f"Recording support: main test {m.get('support', {}).get('main_test_recordings', 'not measured')} recordings / {m.get('support', {}).get('main_test_segments', 'not measured')} segments; shifted test {m.get('support', {}).get('shifted_test_recordings', 'not measured')} recordings / {m.get('support', {}).get('shifted_test_segments', 'not measured')} segments.",
        f"SNR definition: {m.get('snr_definition', 'not measured')}.",
        "Confidence intervals use recording-level bootstrap resampling. Cells with fewer than 30 signals are marked LOW-SUPPORT.",
        "", "## Acceptance targets", "",
        "Measured values are shown below. The reduced protocol makes performance targets not comparable to the full-scale acceptance criteria. The owner-machine timing remains not measured.",
        "", "| Target | Threshold | Measured | Recording-level 95% CI | Support | Status |",
        "|---|---:|---:|---:|---|---|"
    ]
    for key, label, threshold in target_info:
        r = m.get("acceptance_targets", {}).get(key, {})
        val = r.get("measured")
        sup = support(r.get("support")) if r.get("support") else "not measured"
        ci_text = ci({"ci95": r.get("ci95")})
        if key == "processing_time_per_segment_local_seconds":
            val, sup, ci_text = None, "not applicable", "not applicable"
        lines.append(f"| {label} | {threshold} | {pct(val)} | {ci_text} | {sup} | not comparable |")
    calibration = m.get("validation_calibration", {})
    yolo_cal = calibration.get("yolo", {})
    energy_cal = calibration.get("energy_baseline", {})
    lines += [
        "",
        "Validation-only operating-point calibration:",
        f"YOLO: target precision={pct(calibration.get('target_precision'))}; attained target={yolo_cal.get('attained_target', 'not measured')}; threshold={yolo_cal.get('threshold', 'not measured')}; validation precision={pct(yolo_cal.get('precision'))}.",
        f"Energy baseline: target precision={pct(calibration.get('target_precision'))}; attained target={energy_cal.get('attained_target', 'not measured')}; mask threshold={calibration.get('energy_mask_threshold', 'not measured')}; confidence threshold={calibration.get('energy_confidence_threshold', 'not measured')}; validation precision={pct(energy_cal.get('precision'))}.",
    ]
    lines += ["", "## Main recording-disjoint test", "", "### Detector and geometry baseline", "",
              "| Metric | Estimate | Recording-level 95% CI | Recordings / signals |",
              "|---|---:|---:|---|"]
    for model, data in (("YOLOv5n", yolo), ("Energy baseline", energy)):
        for key, title in (("precision", "geometry precision"), ("recall", "class-agnostic recall"),
                            ("mean_iou_true_positives", "mean IoU of true positives")):
            lines.append(metric(f"{model}: {title}", data["matched_precision_with_recording_bootstrap"][key]))
    lines.append(metric("YOLOv5n class mAP@0.5, all SNR", yolo["map50_all_snr"]["map50_recording_bootstrap"]))
    lines.append(metric("YOLOv5n class mAP@0.5, global SNR >= 5 dB", yolo["map50_global_snr_ge_5_db"]["map50_recording_bootstrap"]))
    lines += ["", "### Per-class detector metrics", "",
              "| Class metric | Estimate | Recording-level 95% CI | Recordings / signals |",
              "|---|---:|---:|---|"]
    for name, rows in yolo["per_class_with_recording_bootstrap"].items():
        for key, title in (("precision", "precision"), ("recall", "recall"),
                           ("mean_iou_true_positives", "mean IoU"), ("ap50", "AP@0.5")):
            lines.append(metric(f"{name}: {title}", rows[key]))
    lines += ["", "### Main test by global SNR", "",
              "| Global SNR bin | YOLO recall | 95% CI | Support | YOLO mean IoU | 95% CI | Support | PAM oracle accuracy | 95% CI | Support |",
              "|---|---:|---|---|---:|---|---|---:|---|---|"]
    for snr, row in yolo["by_global_snr_with_recording_bootstrap"].items():
        r, i, p = row.get("recall", {}), row.get("mean_iou_true_positives", {}), pam["oracle_accuracy_by_global_snr"].get(snr, {})
        lines.append(f"| {snr} dB | {pct(r.get('estimate'))} | {ci(r)} | {support(r)} | {pct(i.get('estimate'))} | {ci(i)} | {support(i)} | {pct(p.get('accuracy'))} | {ci(p)} | {support(p)} |")
    snr_keys = list(yolo["by_global_snr_with_recording_bootstrap"])
    for mapping in (
        energy["by_global_snr_with_recording_bootstrap"],
        pam["oracle_accuracy_by_global_snr"],
        pam["detector_box_accuracy_by_global_snr"],
        pam["non_pam_detections_classified_as_others"]["by_global_snr"],
        main["end_to_end"]["by_global_snr_with_recording_bootstrap"],
    ):
        snr_keys.extend(key for key in mapping if key not in snr_keys)
    snr_keys.sort(key=lambda key: float(key.lstrip("[<=").split(",", 1)[0]))
    lines += ["", "### Per-SNR metrics for detector, baseline, PAM, and end-to-end pipeline", "",
              "Each row is a measured metric with its own recording-bootstrap interval and support.", "",
              "| SNR bin | Metric | Estimate | Recording-level 95% CI | Recordings / signals |",
              "|---|---|---:|---:|---|"]
    for snr in snr_keys:
        for model_name, mapping, metric_key in (
            ("YOLO", yolo["by_global_snr_with_recording_bootstrap"].get(snr, {}), None),
            ("Energy baseline", energy["by_global_snr_with_recording_bootstrap"].get(snr, {}), None),
        ):
            for key in ("precision", "recall", "mean_iou_true_positives"):
                row = mapping.get(key)
                if row:
                    lines.append(f"| {snr} dB | {model_name} {key} | {pct(row.get('estimate'))} | {ci(row)} | {support(row)} |")
        for label, mapping, value_key in (
            ("PAM oracle-box accuracy", pam["oracle_accuracy_by_global_snr"], "accuracy"),
            ("PAM detector-box accuracy", pam["detector_box_accuracy_by_global_snr"], "accuracy"),
            ("Non-PAM detections classified as Others", pam["non_pam_detections_classified_as_others"]["by_global_snr"], "rate"),
            ("End-to-end precision", main["end_to_end"]["by_global_snr_with_recording_bootstrap"], "precision"),
            ("End-to-end recall", main["end_to_end"]["by_global_snr_with_recording_bootstrap"], "recall"),
        ):
            item = mapping.get(snr, {})
            row = item.get(value_key) if label.startswith("End-to-end") else item
            if row:
                lines.append(f"| {snr} dB | {label} | {pct(row.get(value_key))} | {ci(row)} | {support(row)} |")
    lines += ["", "### PAM accuracy and false-class results by class", "",
              "| Class / metric | Estimate | Recording-level 95% CI | Recordings / signals |",
              "|---|---:|---:|---|"]
    for subtype in pam["oracle_accuracy_by_subtype"]:
        lines.append(metric(f"{subtype}: oracle-box accuracy", pam["oracle_accuracy_by_subtype"][subtype], "accuracy"))
        lines.append(metric(f"{subtype}: detector-box accuracy", pam["detector_box_accuracy_by_subtype"][subtype], "accuracy"))
    for label, row in pam["all_signal_classifier_accuracy_by_class"].items():
        lines.append(metric(f"All-signal classifier: {label}", row, "accuracy"))
    for detector_class, row in pam["non_pam_detections_classified_as_others"]["by_detector_class"].items():
        lines.append(metric(f"False-class {detector_class} detections classified as Others", row, "rate"))
    lines += ["", "### PAM and end-to-end metrics", "",
              "| Metric | Estimate | Recording-level 95% CI | Recordings / signals |",
              "|---|---:|---:|---|"]
    lines.append(metric("PAM accuracy with oracle boxes", pam["oracle_accuracy"], "accuracy"))
    lines.append(metric("PAM accuracy with detector boxes", pam["detector_box_accuracy"], "accuracy"))
    lines.append(metric("PAM CNN all-signal accuracy incl. Others", pam["all_signal_classifier_accuracy"], "accuracy"))
    lines.append(metric("Non-PAM detections predicted as Others", pam["non_pam_detections_classified_as_others"]["overall"], "rate"))
    e2e = main["end_to_end"]
    for name in ("precision", "recall"):
        lines.append(f"| End-to-end {name} | {pct(e2e.get(name))} | {ci({'ci95': e2e.get(name + '_ci95')})} | {e2e.get('recordings', 'n/a')} recordings / {e2e.get('signals', 'n/a')} signals{'; LOW-SUPPORT' if e2e.get('low_support') else ''} |")
    inband = pam["approximate_pam_inband_snr_db"]
    lines += ["", f"Approximate PAM in-band SNR (dB): {json.dumps(inband, sort_keys=True)}.",
              "", "### Confusion matrices (measured counts)", "",
              "Detector classes, main test:", "",
              matrix(yolo.get("detector_confusion_labels", []), yolo.get("detector_confusion_matrix", [])),
              "", "PAM subtype, oracle boxes:", "",
              matrix(pam.get("oracle_subtype_confusion_labels", []), pam.get("oracle_subtype_confusion_matrix", [])),
              "", "All-signal CNN classes:", "",
              matrix(pam.get("all_classes_confusion_labels", []), pam.get("all_classes_confusion_matrix", [])),
              "", "## Shifted-parameter test", "",
              "This distribution-shift test was not used for calibration.", "",
              "| Metric | Estimate | Recording-level 95% CI | Recordings / signals |",
              "|---|---:|---:|---|"]
    for name, row, key in (
        ("YOLO precision", shifted["yolo_matched_precision_with_recording_bootstrap"]["precision"], "estimate"),
        ("YOLO recall", shifted["yolo_matched_precision_with_recording_bootstrap"]["recall"], "estimate"),
        ("YOLO mean IoU", shifted["yolo_matched_precision_with_recording_bootstrap"]["mean_iou_true_positives"], "estimate"),
        ("YOLO mAP@0.5", shifted["yolo_map50"]["map50_recording_bootstrap"], "estimate"),
        ("PAM oracle-box accuracy", shifted["pam_oracle_accuracy"], "accuracy"),
        ("PAM detector-box accuracy", shifted["pam_detector_box_accuracy"], "accuracy"),
    ):
        lines.append(metric(name, row, key))
    lines += ["", "| Shifted detector class | Precision (95% CI) | Recall (95% CI) | Mean IoU (95% CI) | AP@0.5 (95% CI) | Support |",
              "|---|---:|---:|---:|---:|---|"]
    for name, rows in shifted["yolo_per_class_with_recording_bootstrap"].items():
        lines.append(f"| {name} | {pct(rows['precision'].get('estimate'))} {ci(rows['precision'])} | {pct(rows['recall'].get('estimate'))} {ci(rows['recall'])} | {pct(rows['mean_iou_true_positives'].get('estimate'))} {ci(rows['mean_iou_true_positives'])} | {pct(rows['ap50'].get('estimate'))} {ci(rows['ap50'])} | {support(rows['ap50'])} |")
    lines += ["", "### Shifted-test metrics by global SNR", "",
              "| SNR bin | Metric | Estimate | Recording-level 95% CI | Recordings / signals |",
              "|---|---|---:|---:|---|"]
    shifted_maps = [
        ("YOLO", shifted["yolo_by_global_snr_with_recording_bootstrap"]),
        ("Energy baseline", shifted["energy_baseline_by_global_snr_with_recording_bootstrap"]),
    ]
    shifted_snr_keys = list(shifted_maps[0][1])
    for _, mapping in shifted_maps[1:]:
        shifted_snr_keys.extend(key for key in mapping if key not in shifted_snr_keys)
    for snr in sorted(shifted_snr_keys, key=lambda key: float(key.lstrip("[<=").split(",", 1)[0])):
        for model_name, mapping in shifted_maps:
            for key, row in mapping.get(snr, {}).items():
                lines.append(f"| {snr} dB | {model_name} {key} | {pct(row.get('estimate'))} | {ci(row)} | {support(row)} |")
        for label, mapping, value_key in (
            ("PAM oracle-box accuracy", shifted["pam_oracle_accuracy_by_global_snr"], "accuracy"),
            ("PAM detector-box accuracy", shifted["pam_detector_box_accuracy_by_global_snr"], "accuracy"),
            ("Non-PAM detections classified as Others", shifted["non_pam_detections_classified_as_others"]["by_global_snr"], "rate"),
        ):
            row = mapping.get(snr)
            if row:
                lines.append(f"| {snr} dB | {label} | {pct(row.get(value_key))} | {ci(row)} | {support(row)} |")
        e2e_snr = shifted["end_to_end_by_global_snr_with_recording_bootstrap"].get(snr, {})
        for key in ("precision", "recall"):
            row = e2e_snr.get(key)
            if row:
                lines.append(f"| {snr} dB | End-to-end {key} | {pct(row.get('estimate'))} | {ci(row)} | {support(row)} |")
    lines += ["", "### Shifted-test confusion matrices", "",
              "Detector classes:", "",
              matrix(shifted.get("detector_confusion_labels", []), shifted.get("detector_confusion_matrix", [])),
              "", "PAM subtype, oracle boxes:", "",
              matrix(shifted.get("pam_subtype_confusion_labels", []), shifted.get("pam_subtype_confusion_matrix", [])),
              "", "## Plots and measured artifacts", "",
              "The metrics file preserves measured values, CIs, supports, validation thresholds, and confusion counts."]
    plot_dir = repo / "results" / "reduced_cpu" / "plots"
    if plot_dir.exists():
        lines.extend(f"- [{p.name}](results/reduced_cpu/plots/{p.name})" for p in sorted(plot_dir.glob("*.png")))
    else:
        lines.append("Plots: not yet measured.")
    lines += ["", "- [Complete measured metrics JSON](results/reduced_cpu/metrics.json)",
              "- [Run summary](results/reduced_cpu/summary.txt)", "",
              "## Runtime", "",
              f"Measured total pipeline runtime on hosted CPU (seconds): {m.get('runtime_seconds_pipeline', {}).get('total', 'not measured')}.",
              f"Measured data generation runtime (seconds): {m.get('runtime_seconds_pipeline', {}).get('data_generation', 'not measured')}.",
              f"Detector training status: {m.get('training_status', {}).get('detector', 'not measured')}; evaluation completed: {m.get('training_status', {}).get('evaluation_completed', 'not measured')}.",
              "The under-five-second target is specific to the owner's local machine and remains not measured there; hosted run time is not comparable.",
              "", "## Interpretation", "",
              "Use this reduced CPU execution to inspect the software path and measured outputs. It does not establish full-scale target performance. Investigate misses using validation data only; keep both test sets frozen."]
    return "\n".join(lines) + "\n"


def update_readme(repo, m):
    path = repo / "README.md"
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError("README reduced-results markers are missing")
    text = text.replace(
        "The reduced CPU workflow publishes actual measured results after a successful run; no reduced CPU run has been reported yet.",
        "The reduced CPU results in this README are from the GitHub-hosted CPU workflow. The optional full GPU Colab run remains unmeasured.",
    )
    text = text.replace(
        "The reduced CPU numbers will be written here and in [RESULTS.md](RESULTS.md) only after a successful workflow evaluation.",
        "The reduced CPU results below and in [RESULTS.md](RESULTS.md) were written after a successful workflow evaluation.",
    )
    text = text.replace(
        "The reduced CPU workflow is the source of measured values in this section. Before its first successful evaluation, measurements are **not yet measured**. A completed reduced run is labelled **not comparable to target**; full GPU results remain optional and unmeasured.",
        "The measured values below are from the completed reduced CPU workflow. This run is labelled **not comparable to target**; full GPU results remain optional and unmeasured.",
    )
    targets = [
        ("Detector recall, global SNR >= 5 dB", "recall_at_global_snr_ge_5_db"),
        ("Mean IoU, global SNR >= 5 dB", "mean_iou_at_global_snr_ge_5_db"),
        ("Class mAP@0.5, global SNR >= 5 dB", "map50_at_global_snr_ge_5_db"),
        ("PAM oracle accuracy, global SNR >= 10 dB", "pam_oracle_accuracy_global_snr_ge_10_db"),
        ("PAM oracle accuracy, global SNR [0, 1) dB", "pam_oracle_accuracy_near_0_db_0_to_1"),
        ("Owner's local processing time per 1M samples", "processing_time_per_segment_local_seconds"),
    ]
    rows = ["", "**Run:** reduced run: CPU, small data, not comparable to target. Performance target status is not comparable due to reduced support. The full GPU run via the optional Colab notebook is still unmeasured.", "",
            "| Acceptance metric | Target | Reduced CPU measured | 95% recording-bootstrap CI | Support | Status |",
            "|---|---:|---:|---:|---|---|"]
    for label, key in targets:
        r = m.get("acceptance_targets", {}).get(key, {})
        value = r.get("measured")
        ci_text = ci({"ci95": r.get("ci95")})
        sup = support(r.get("support")) if r.get("support") else "owner measurement pending"
        if key == "processing_time_per_segment_local_seconds":
            value, ci_text, sup = None, "not applicable", "owner measurement pending"
        threshold = r.get("target")
        if isinstance(threshold, (int, float)):
            threshold = f">= {threshold * 100:.0f}%" if threshold >= 0.1 else (f"> {threshold * 100:.0f}%" if key.endswith("0_to_1") else f"< {threshold:g} seconds")
        rows.append(f"| {label} | {threshold or 'report'} | {pct(value)} | {ci_text} | {sup} | not comparable |")
    rows += ["", f"Main test support: {m.get('support', {}).get('main_test_recordings', 'not measured')} recordings / {m.get('support', {}).get('main_test_segments', 'not measured')} segments; shifted test: {m.get('support', {}).get('shifted_test_recordings', 'not measured')} recordings / {m.get('support', {}).get('shifted_test_segments', 'not measured')} segments.", "",
             "See [RESULTS.md](RESULTS.md) and [the measured metrics JSON](results/reduced_cpu/metrics.json) for per-class/per-SNR CIs, support, confusion matrices, and plots. Cells under 30 signals are LOW-SUPPORT.", ""]
    replacement = START + "\n" + "\n".join(rows) + END
    text = text[:text.index(START)] + replacement + text[text.index(END) + len(END):]
    path.write_text(text, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    run, repo = args.run_dir.resolve(), args.repo_root.resolve()
    source = run / "results" / "metrics.json"
    if not source.is_file():
        raise FileNotFoundError(f"Measured metrics are missing: {source}")
    m = json.loads(source.read_text(encoding="utf-8"))
    if m.get("training_status", {}).get("evaluation_completed") is not True:
        raise ValueError("Refusing to publish because evaluation did not complete.")
    if m.get("comparison_scope") != "not_comparable":
        raise ValueError("Refusing to publish a run not labelled not comparable.")
    dest = repo / "results" / "reduced_cpu"
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "plots").mkdir(parents=True)
    m["metrics_path"] = "results/reduced_cpu/metrics.json"
    (dest / "metrics.json").write_text(json.dumps(m, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = run / "results" / "summary.txt"
    if summary.exists():
        safe_lines = []
        for line in summary.read_text(encoding="utf-8").splitlines():
            if line.startswith("Output folder:"):
                line = "Output folder: results/reduced_cpu"
            elif line.startswith("Metrics JSON:"):
                line = "Metrics JSON: results/reduced_cpu/metrics.json"
            safe_lines.append(line)
        (dest / "summary.txt").write_text("\n".join(safe_lines) + "\n", encoding="utf-8")
    for plot in (run / "plots").glob("*.png"):
        shutil.copy2(plot, dest / "plots" / plot.name)
    (repo / "RESULTS.md").write_text(build_results(m, repo), encoding="utf-8")
    update_readme(repo, m)
    print(f"Published measured reduced CPU results from {source}")



if __name__ == "__main__":
    main()
