"""Publish only measured outputs from a completed reduced CPU run."""
from __future__ import annotations
import argparse
import json
import re
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


def build_split_audit(manifest):
    """Return an auditable recording split and reject inconsistent IDs."""
    main_rows = manifest.get("main")
    if not isinstance(main_rows, list) or not main_rows:
        raise ValueError("Recording split manifest has no main recording rows.")

    split_names = ("train", "val", "test")
    ids = {name: set() for name in split_names}
    assignments = {}
    for row in main_rows:
        recording_id, split = row.get("recording_id"), row.get("split")
        if not recording_id or split not in ids:
            raise ValueError(f"Invalid main recording split row: {row!r}")
        assignments.setdefault(recording_id, set()).add(split)
        ids[split].add(recording_id)

    overlaps = {
        "train__val": sorted(ids["train"] & ids["val"]),
        "train__test": sorted(ids["train"] & ids["test"]),
        "val__test": sorted(ids["val"] & ids["test"]),
    }
    inconsistent = sorted(recording_id for recording_id, splits in assignments.items() if len(splits) != 1)
    shifted_ids = sorted({row["recording_id"] for row in manifest.get("shifted", []) if row.get("recording_id")})
    main_ids = set().union(*ids.values())
    shifted_overlap = sorted(main_ids & set(shifted_ids))
    return {
        "main_recording_ids_by_split": {name: sorted(ids[name]) for name in split_names},
        "main_recording_counts": {name: len(ids[name]) for name in split_names},
        "pairwise_main_split_overlaps": overlaps,
        "inconsistent_main_recording_assignments": inconsistent,
        "main_splits_disjoint": not any(overlaps.values()) and not inconsistent,
        "shifted_test_recording_ids": shifted_ids,
        "shifted_test_overlap_with_main": shifted_overlap,
        "shifted_test_disjoint_from_main": not shifted_overlap,
    }


def format_target(key, value):
    if value is None:
        return "report"
    if key == "processing_time_per_segment_local_seconds":
        return f"< {value:g} seconds"
    if key.endswith("near_0_db_0_to_1"):
        return f"> {value * 100:.0f}%"
    return f">= {value * 100:.0f}%"


def metric(label, row, key="estimate"):
    row = row or {}
    return f"| {label} | {pct(row.get(key))} | {ci(row)} | {support(row)} |"


def pam_metric(label, row, key="estimate"):
    return metric(f"{label} (chance level: 1/7 = 14.29%)", row, key)


def target_display_value(key, value):
    if value is None:
        return "not measured"
    if key == "processing_time_per_segment_local_seconds":
        return f"{float(value):.3f} s"
    return pct(value)


def matrix(labels, values):
    if not labels or not values:
        return "Not measured."
    out = ["| Actual \\\\ Predicted | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    out += [f"| {name} | " + " | ".join(str(int(v)) for v in row) + " |"
            for name, row in zip(labels, values)]
    return "\n".join(out)


def normalize_zero_detection_precision(metrics):
    """Represent undefined precision as null when the operating point emitted no boxes."""
    for split_name in ("main_test", "shifted_test"):
        split = metrics.get(split_name, {})
        end_to_end = split.get("end_to_end", {})
        if end_to_end.get("predictions") == 0:
            end_to_end["precision"] = None
            end_to_end["precision_ci95"] = [None, None]
        yolo_by_snr = split.get("yolo", {}).get("by_global_snr_with_recording_bootstrap", {})
        e2e_by_snr = end_to_end.get("by_global_snr_with_recording_bootstrap", {})
        for snr_bin, row in e2e_by_snr.items():
            detector_precision = yolo_by_snr.get(snr_bin, {}).get("precision", {})
            if detector_precision.get("estimate") is None:
                row.setdefault("precision", {})["estimate"] = None
                row["precision"]["ci95"] = [None, None]
    shifted = metrics.get("shifted_test", {})
    shifted_snr = shifted.get("end_to_end_by_global_snr_with_recording_bootstrap", {})
    yolo_snr = shifted.get("yolo_by_global_snr_with_recording_bootstrap", {})
    for snr_bin, row in shifted_snr.items():
        detector_precision = yolo_snr.get(snr_bin, {}).get("precision", {})
        if detector_precision.get("estimate") is None:
            row.setdefault("precision", {})["estimate"] = None
            row["precision"]["ci95"] = [None, None]
    return metrics


def normalize_summary(summary, metrics):
    """Make legacy zero-detection precision and PAM chance notes explicit."""
    main = metrics.get("main_test", {})
    shifted = metrics.get("shifted_test", {})
    main_zero = main.get("end_to_end", {}).get("predictions") == 0
    shifted_zero = shifted.get("end_to_end", {}).get("predictions") == 0
    main_snr = main.get("yolo", {}).get("by_global_snr_with_recording_bootstrap", {})
    shifted_snr = shifted.get("yolo_by_global_snr_with_recording_bootstrap", {})
    out = []
    for line in summary.splitlines():
        lower = line.lower()
        if "accuracy" in lower and (
            "pam" in lower or "subtype" in lower or "all-signal classifier" in lower
        ) and "chance level" not in lower:
            head, sep, tail = line.partition(":")
            if sep:
                line = f"{head} (PAM chance level 1/7 = 14.29%):{tail}"
        if "precision" in lower:
            line = line.replace("n/a; CI95=[n/a, n/a]", "not measured; CI95=[not measured, not measured]")
            should_be_unmeasured = (
                (main_zero and lower.startswith("end-to-end precision"))
                or (shifted_zero and lower.startswith("shifted end-to-end precision"))
            )
            match = re.search(r"global snr (\[[^]]+\] dB)", lower)
            if match and lower.startswith("end-to-end precision"):
                snr = match.group(1).replace(" dB", "")
                should_be_unmeasured = should_be_unmeasured or main_snr.get(snr, {}).get("precision", {}).get("estimate") is None
            if match and lower.startswith("shifted end-to-end precision"):
                snr = match.group(1).replace(" dB", "")
                should_be_unmeasured = should_be_unmeasured or shifted_snr.get(snr, {}).get("precision", {}).get("estimate") is None
            if should_be_unmeasured:
                line = re.sub(r": [^;]+; CI95=\[[^]]+\]", ": not measured; CI95=[not measured, not measured]", line, count=1)
        # Normalize legacy undefined per-class and per-SNR precision fields.
        line = re.sub(
            r"(\bP=)n/a CI95=\[(?:None|n/a),\s*(?:None|n/a)\]",
            r"\1not measured CI95=[not measured, not measured]",
            line, flags=re.IGNORECASE,
        )
        line = re.sub(
            r"(\bprecision:\s*)n/a; CI95=\[(?:None|n/a),\s*(?:None|n/a)\]",
            r"\1not measured; CI95=[not measured, not measured]",
            line, flags=re.IGNORECASE,
        )
        # Undefined mean-IoU and zero-support accuracy cells use the same
        # explicit wording as undefined precision.
        line = re.sub(
            r"(\bIoU=)n/a CI95=\[(?:None|n/a),\s*(?:None|n/a)\]",
            r"\1not measured CI95=[not measured, not measured]", line, flags=re.IGNORECASE,
        )
        line = re.sub(
            r"(accuracy(?:[^:]*):\s*)n/a; CI95=\[(?:n/a|None),\s*(?:n/a|None)\]",
            r"\1not measured; CI95=[not measured, not measured]", line, flags=re.IGNORECASE,
        )
        line = re.sub(
            r"(\b(?:P|IoU)=)not measured CI95=\[(?:None|n/a),\s*(?:None|n/a)\]",
            r"\1not measured CI95=[not measured, not measured]", line, flags=re.IGNORECASE,
        )
        out.append(line.rstrip())
    return "\n".join(out) + "\n"


def build_results(m, repo, split_audit):
    main, shifted = m["main_test"], m["shifted_test"]
    yolo, energy, pam = main["yolo"], main["energy_baseline"], main["pam"]
    artifact_dir = Path(m.get("metrics_path", "results/reduced_cpu/metrics.json")).parent.as_posix()
    not_comparable = m.get("comparison_scope") == "not_comparable"
    comparison_text = (
        "This reduced run is not comparable to the full-scale acceptance protocol."
        if not_comparable else
        "Target status uses each measured point estimate; recording-level confidence intervals show uncertainty."
    )
    target_info = [
        ("recall_at_global_snr_ge_5_db", "Detector recall, global SNR >= 5 dB", ">= 80%"),
        ("mean_iou_at_global_snr_ge_5_db", "Mean IoU, global SNR >= 5 dB", ">= 0.75"),
        ("map50_at_global_snr_ge_5_db", "Class mAP@0.5, global SNR >= 5 dB", ">= 60%"),
        ("pam_oracle_accuracy_global_snr_ge_10_db", "PAM oracle-box accuracy (chance 1/7 = 14.29%), global SNR >= 10 dB", ">= 80%"),
        ("pam_oracle_accuracy_near_0_db_0_to_1", "PAM oracle-box accuracy (chance 1/7 = 14.29%), global SNR [0, 1) dB", "> 50%"),
        ("processing_time_per_segment_local_seconds", "Local CPU processing time per 1M samples", "< 5 seconds"),
    ]
    lines = [
        f"# {m.get('results_title', 'Reduced CPU Run Results')}", "",
        f"Run label: {m.get('run_label', 'reduced run: CPU, small data, not comparable to target')}",
        f"Comparison: {comparison_text}",
        "Data source: self-contained reproducible synthetic SigMF/IQ generator. The 53.55 GiB Zenodo/DeepSig download requirement is dropped; no external recordings were used.",
        f"Device: {m.get('device', 'not measured')}",
        f"YOLOv5 source commit: {m.get('yolo', {}).get('commit', 'not measured')}",
        f"Detector training status: {m.get('training_status', {}).get('detector', 'not measured')}; checkpoint: {m.get('training_status', {}).get('detector_checkpoint_used', 'not measured')}; requested epochs: {m.get('training_status', {}).get('detector_epochs_requested', 'not measured')}.",
        f"PAM training status: {m.get('training_status', {}).get('pam', 'not measured')}; epochs completed: {m.get('training_status', {}).get('pam_epochs_completed', 'not measured')} of {m.get('training_status', {}).get('pam_epochs_requested', 'not measured')}.",
        f"Recording support: main test {m.get('support', {}).get('main_test_recordings', 'not measured')} recordings / {m.get('support', {}).get('main_test_segments', 'not measured')} segments; shifted test {m.get('support', {}).get('shifted_test_recordings', 'not measured')} recordings / {m.get('support', {}).get('shifted_test_segments', 'not measured')} segments.",
        f"Recording-ID leakage audit: train={split_audit['main_recording_counts']['train']}, validation={split_audit['main_recording_counts']['val']}, test={split_audit['main_recording_counts']['test']}; main splits are pairwise disjoint; shifted test IDs are separate. See [split_audit.json]({artifact_dir}/split_audit.json) for IDs.",
        f"SNR definition: {m.get('snr_definition', 'not measured')}.",
        "Confidence intervals use recording-level bootstrap resampling. Cells with fewer than 30 signals are marked LOW-SUPPORT.",
        "", "## Acceptance targets", "",
        "Measured values are shown below. A missing value is marked not measured.",
        "", "| Target | Threshold | Measured | Recording-level 95% CI | Support | Status |",
        "|---|---:|---:|---:|---|---|"
    ]
    for key, label, threshold in target_info:
        r = m.get("acceptance_targets", {}).get(key, {})
        val = r.get("measured")
        sup = support(r.get("support")) if r.get("support") else "not measured"
        ci_text = ci({"ci95": r.get("ci95")})
        if key == "processing_time_per_segment_local_seconds":
            ci_text = "not applicable"
            if val is not None:
                sup = r.get("measurement_scope", "local synthetic CLI timing")
        if not_comparable:
            status = "not comparable"
        elif val is None:
            status = "not measured"
        else:
            status = "met" if r.get("pass") else "missed"
        display_value = target_display_value(key, val)
        lines.append(f"| {label} | {threshold} | {display_value} | {ci_text} | {sup} | {status} |")
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
              "| Global SNR bin | YOLO recall | 95% CI | Support | YOLO mean IoU | 95% CI | Support | PAM oracle accuracy (chance 1/7 = 14.29%) | 95% CI | Support |",
              "|---|---:|---|---|---:|---|---|---:|---|---|"]
    for snr, row in yolo["by_global_snr_with_recording_bootstrap"].items():
        r, i, p = row.get("recall", {}), row.get("mean_iou_true_positives", {}), pam["oracle_accuracy_by_global_snr"].get(snr, {})
        lines.append(f"| {snr} dB | {pct(r.get('estimate'))} | {ci(r)} | {support(r)} | {pct(i.get('estimate'))} | {ci(i)} | {support(i)} | {pct(p.get('accuracy'))} (chance 1/7 = 14.29%) | {ci(p)} | {support(p)} |")
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
            ("PAM oracle-box accuracy (chance 1/7 = 14.29%)", pam["oracle_accuracy_by_global_snr"], "accuracy"),
            ("PAM detector-box accuracy (chance 1/7 = 14.29%)", pam["detector_box_accuracy_by_global_snr"], "accuracy"),
            ("Non-PAM detections classified as Others", pam["non_pam_detections_classified_as_others"]["by_global_snr"], "rate"),
            ("End-to-end precision", main["end_to_end"]["by_global_snr_with_recording_bootstrap"], "precision"),
            ("End-to-end recall", main["end_to_end"]["by_global_snr_with_recording_bootstrap"], "recall"),
        ):
            item = mapping.get(snr, {})
            row = item.get(value_key) if label.startswith("End-to-end") else item
            if row:
                lines.append(f"| {snr} dB | {label} | {pct(row.get(value_key))} | {ci(row)} | {support(row)} |")
    lines += ["", "### PAM accuracy and false-class results by class", "",
              "PAM classifier chance accuracy is 1/7 = 14.29%; every PAM accuracy below carries this reference.",
              "| Class / metric | Estimate | Recording-level 95% CI | Recordings / signals |",
              "|---|---:|---:|---|"]
    for subtype in pam["oracle_accuracy_by_subtype"]:
        lines.append(pam_metric(f"{subtype}: oracle-box accuracy", pam["oracle_accuracy_by_subtype"][subtype], "accuracy"))
        lines.append(pam_metric(f"{subtype}: detector-box accuracy", pam["detector_box_accuracy_by_subtype"][subtype], "accuracy"))
    for label, row in pam["all_signal_classifier_accuracy_by_class"].items():
        lines.append(pam_metric(f"All-signal classifier: {label}", row, "accuracy"))
    for detector_class, row in pam["non_pam_detections_classified_as_others"]["by_detector_class"].items():
        lines.append(metric(f"False-class {detector_class} detections classified as Others", row, "rate"))
    lines += ["", "### PAM and end-to-end metrics", "",
              "| Metric | Estimate | Recording-level 95% CI | Recordings / signals |",
              "|---|---:|---:|---|"]
    lines.append(pam_metric("PAM accuracy with oracle boxes", pam["oracle_accuracy"], "accuracy"))
    lines.append(pam_metric("PAM accuracy with detector boxes", pam["detector_box_accuracy"], "accuracy"))
    lines.append(pam_metric("PAM CNN all-signal accuracy incl. Others", pam["all_signal_classifier_accuracy"], "accuracy"))
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
        ("PAM oracle-box accuracy (chance 1/7 = 14.29%)", shifted["pam_oracle_accuracy"], "accuracy"),
        ("PAM detector-box accuracy (chance 1/7 = 14.29%)", shifted["pam_detector_box_accuracy"], "accuracy"),
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
            ("PAM oracle-box accuracy (chance 1/7 = 14.29%)", shifted["pam_oracle_accuracy_by_global_snr"], "accuracy"),
            ("PAM detector-box accuracy (chance 1/7 = 14.29%)", shifted["pam_detector_box_accuracy_by_global_snr"], "accuracy"),
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
    plot_dir = repo / artifact_dir / "plots"
    if plot_dir.exists():
        lines.extend(f"- [{p.name}]({artifact_dir}/plots/{p.name})" for p in sorted(plot_dir.glob("*.png")))
    else:
        lines.append("Plots: not yet measured.")
    lines += ["", f"- [Complete measured metrics JSON]({artifact_dir}/metrics.json)",
              f"- [Run summary]({artifact_dir}/summary.txt)", "",
              "## Runtime", "",
              f"Measured total pipeline runtime on the configured device (seconds): {m.get('runtime_seconds_pipeline', {}).get('total', 'not measured')}.",
              f"Measured data generation runtime (seconds): {m.get('runtime_seconds_pipeline', {}).get('data_generation', 'not measured')}.",
              f"YOLO timing probe: {('not run for this profile' if not m.get('timing_probe') else str(m['timing_probe'].get('wall_seconds', 'not measured')) + ' seconds for ' + str(m['timing_probe'].get('batches_per_epoch', 'not measured')) + ' training batches')}. See [{artifact_dir}/timing_probe.json]({artifact_dir}/timing_probe.json) when present.",
              f"Detector training status: {m.get('training_status', {}).get('detector', 'not measured')}; evaluation completed: {m.get('training_status', {}).get('evaluation_completed', 'not measured')}.",
              f"The local per-segment timing target is {'measured in this run' if m.get('acceptance_targets', {}).get('processing_time_per_segment_local_seconds', {}).get('measured') is not None else 'not yet measured'}; see the acceptance table above.",
              "", "## Interpretation", "",
              "Use this reduced CPU execution to inspect the software path and measured outputs. It does not establish full-scale target performance. Investigate misses using validation data only; keep both test sets frozen."]
    return "\n".join(lines) + "\n"


def latest_run_summary(m):
    main = m.get("main_test", {})
    yolo = main.get("yolo", {}).get("matched_precision_with_recording_bootstrap", {})
    energy = main.get("energy_baseline", {}).get("matched_precision_with_recording_bootstrap", {})
    yolo_recall = yolo.get("recall", {})
    energy_recall = energy.get("recall", {})
    yolo_precision = yolo.get("precision", {})
    energy_precision = energy.get("precision", {})
    map_row = main.get("yolo", {}).get("map50_global_snr_ge_5_db", {}).get("map50_recording_bootstrap", {})
    pam = main.get("pam", {}).get("all_signal_classifier_accuracy", {})
    training = m.get("training_status", {})
    calibration = m.get("validation_calibration", {})
    return (
        "The synthetic-to-CLI pipeline runs end to end, but the reduced CPU YOLOv5n did not learn useful detections. It processes I/Q through detection, PAM classification, and JSON plus annotated-spectrogram output. "
        f"Latest measured reduced CPU run: YOLOv5n trained for {training.get('detector_epochs_requested', 'not measured')} requested epochs; recall {pct(yolo_recall.get('estimate'))} (recording-bootstrap CI {ci(yolo_recall)}; {support(yolo_recall)}) and mAP@0.5 at global SNR >= 5 dB {pct(map_row.get('estimate'))} (CI {ci(map_row)}; {support(map_row)}). "
        f"The energy geometry baseline had higher held-out recall ({pct(energy_recall.get('estimate'))} vs {pct(yolo_recall.get('estimate'))}; CI {ci(energy_recall)}; {support(energy_recall)}), but this is not a matched-precision comparison (energy precision {pct(energy_precision.get('estimate'))}; YOLO precision {pct(yolo_precision.get('estimate'))}). "
        f"Validation calibration attained the 80% precision target for YOLO: {calibration.get('yolo', {}).get('attained_target', 'not measured')}; energy baseline: {calibration.get('energy_baseline', {}).get('attained_target', 'not measured')}. "
        f"PAM all-signal accuracy was {pct(pam.get('accuracy'))} (uniform seven-class chance: 1/7 = 14.29%; CI {ci(pam)}; {support(pam)}). "
        "These are reduced CPU measurements and are not comparable to the full-scale targets; full GPU results remain unmeasured. See [DIAGNOSIS.md](DIAGNOSIS.md) for the training-only bug check and [RESULTS.md](RESULTS.md) for complete results."
    )


def update_readme(repo, m):
    path = repo / "README.md"
    text = path.read_text(encoding="utf-8")
    architecture_marker = "\n## Architecture\n"
    if architecture_marker in text:
        title, rest = text.split(architecture_marker, 1)
        heading = next((line for line in title.splitlines() if line.startswith("# ")), "# Wideband Signal Recognition")
        text = heading + "\n\n" + latest_run_summary(m) + "\n" + architecture_marker + rest
    text = text.replace(
        "| Acceptance metric | Target | Reduced CPU measured | Recording-level 95% CI | Support | Status |\n"
        "|---|---:|---:|---:|---|---|\n" + START,
        START,
    )
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
        ("PAM oracle accuracy (chance 1/7 = 14.29%), global SNR >= 10 dB", "pam_oracle_accuracy_global_snr_ge_10_db"),
        ("PAM oracle accuracy (chance 1/7 = 14.29%), global SNR [0, 1) dB", "pam_oracle_accuracy_near_0_db_0_to_1"),
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
        threshold = format_target(key, r.get("target"))
        rows.append(f"| {label} | {threshold} | {pct(value)} | {ci_text} | {sup} | not comparable |")
    rows += ["", f"Main test support: {m.get('support', {}).get('main_test_recordings', 'not measured')} recordings / {m.get('support', {}).get('main_test_segments', 'not measured')} segments; shifted test: {m.get('support', {}).get('shifted_test_recordings', 'not measured')} recordings / {m.get('support', {}).get('shifted_test_segments', 'not measured')} segments.", "",
             "See [RESULTS.md](RESULTS.md), [the measured metrics JSON](results/reduced_cpu/metrics.json), and [the recording split audit](results/reduced_cpu/split_audit.json) for per-class/per-SNR CIs, support, confusion matrices, split IDs, and plots. Cells under 30 signals are LOW-SUPPORT.", ""]
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
    m = normalize_zero_detection_precision(json.loads(source.read_text(encoding="utf-8")))
    manifest_source = run / "results" / "dataset_manifest.json"
    if not manifest_source.is_file():
        raise FileNotFoundError(f"Recording split manifest is missing: {manifest_source}")
    split_audit = build_split_audit(json.loads(manifest_source.read_text(encoding="utf-8")))
    if not split_audit["main_splits_disjoint"] or not split_audit["shifted_test_disjoint_from_main"]:
        raise ValueError("Refusing to publish results with overlapping recording IDs across splits.")
    expected_counts = m.get("split_counts_recordings", {})
    if any(expected_counts.get(name) != count for name, count in split_audit["main_recording_counts"].items()):
        raise ValueError("Metric split counts do not match the recording manifest.")
    if m.get("training_status", {}).get("evaluation_completed") is not True:
        raise ValueError("Refusing to publish because evaluation did not complete.")
    if m.get("comparison_scope") != "not_comparable":
        raise ValueError("Refusing to publish a run not labelled not comparable.")
    dest = repo / "results" / "reduced_cpu"
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "plots").mkdir(parents=True)
    (dest / "split_audit.json").write_text(json.dumps(split_audit, indent=2) + "\n", encoding="utf-8")
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
        normalized_summary = normalize_summary("\n".join(safe_lines), m)
        (dest / "summary.txt").write_text(normalized_summary, encoding="utf-8")
    timing_probe = run / "results" / "yolo_timing_probe.json"
    if timing_probe.is_file():
        shutil.copy2(timing_probe, dest / "timing_probe.json")
    for plot in (run / "plots").glob("*.png"):
        shutil.copy2(plot, dest / "plots" / plot.name)
    (repo / "RESULTS.md").write_text(build_results(m, repo, split_audit), encoding="utf-8")
    update_readme(repo, m)
    print(f"Published measured reduced CPU results from {source}")



if __name__ == "__main__":
    main()
