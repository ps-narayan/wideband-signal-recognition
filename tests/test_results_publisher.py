from scripts.publish_reduced_results import (
    END, START, build_split_audit, format_target, matrix, pct, support, target_display_value, update_readme,
)


def test_missing_metrics_stay_unmeasured():
    assert pct(None) == "not measured"
    assert support(None) == "not measured"
    assert matrix([], []) == "Not measured."


def test_low_support_is_explicit():
    assert support({"recordings": 2, "signals": 17, "low_support": True}) == (
        "2 recordings / 17 signals; LOW-SUPPORT"
    )


def test_readme_publisher_changes_only_marked_results_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "intro\n| Acceptance metric | Target | Reduced CPU measured | Recording-level 95% CI | Support | Status |\n"
        "|---|---:|---:|---:|---|---|\n" + START + "\nplaceholder\n" + END + "\noutro\n",
        encoding="utf-8",
    )
    keys = (
        "recall_at_global_snr_ge_5_db",
        "mean_iou_at_global_snr_ge_5_db",
        "map50_at_global_snr_ge_5_db",
        "pam_oracle_accuracy_global_snr_ge_10_db",
        "pam_oracle_accuracy_near_0_db_0_to_1",
        "processing_time_per_segment_local_seconds",
    )
    metrics = {
        "acceptance_targets": {
            key: {"measured": None, "ci95": None, "support": None, "target": None}
            for key in keys
        },
        "support": {},
    }
    update_readme(tmp_path, metrics)
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n" + START)
    assert text.endswith(END + "\noutro\n")
    assert "placeholder" not in text
    assert "not measured" in text
    assert "not comparable" in text


def test_empty_detector_cells_are_not_reported_as_zero():
    from scripts.run_milestone2 import _bootstrap_detection

    empty = _bootstrap_detection([], "recall", 0.5, 10)
    assert empty["estimate"] is None
    assert empty["ci95"] == [None, None]
    assert empty["signals"] == 0

    no_ground_truth = _bootstrap_detection(
        [{"recording_id": "r0", "gt": [], "pred": []}], "recall", 0.5, 10
    )
    assert no_ground_truth["estimate"] is None
    assert no_ground_truth["ci95"] == [None, None]


def test_target_formatting_preserves_strictness_and_units():
    assert format_target("pam_oracle_accuracy_near_0_db_0_to_1", 0.5) == "> 50%"
    assert format_target("processing_time_per_segment_local_seconds", 5) == "< 5 seconds"
    assert target_display_value("processing_time_per_segment_local_seconds", 29.891) == "29.891 s"


def test_recording_split_audit_detects_disjoint_and_overlapping_ids():
    manifest = {
        "main": [
            {"recording_id": "r-train", "split": "train"},
            {"recording_id": "r-val", "split": "val"},
            {"recording_id": "r-test", "split": "test"},
        ],
        "shifted": [{"recording_id": "shift-0"}],
    }
    audit = build_split_audit(manifest)
    assert audit["main_splits_disjoint"]
    assert audit["shifted_test_disjoint_from_main"]
    assert audit["main_recording_counts"] == {"train": 1, "val": 1, "test": 1}

    manifest["main"].append({"recording_id": "r-train", "split": "test"})
    audit = build_split_audit(manifest)
    assert not audit["main_splits_disjoint"]
    assert audit["pairwise_main_split_overlaps"]["train__test"] == ["r-train"]


def test_major_project_readme_table_uses_only_measured_values_and_status():
    from scripts.publish_major_project_results import build_readme_section

    metrics = {
        "run_label": "synthetic CPU run",
        "device": "cpu",
        "training_status": {"detector": "completed", "detector_epochs_completed": 80,
                            "detector_epochs_requested": 80, "pam": "completed",
                            "pam_epochs_completed": 20, "pam_epochs_requested": 60},
        "runtime_seconds_pipeline": {"total": 123.0, "data_generation": 10.0},
        "acceptance_targets": {
            "recall_at_global_snr_ge_5_db": {
                "target": 0.8, "measured": 0.85, "ci95": [0.7, 0.9], "pass": True,
                "support": {"recordings": 30, "signals": 30, "low_support": False},
            },
        },
    }
    output = build_readme_section(metrics)
    assert "85.00%" in output and "met" in output
    assert "not measured" in output
    assert "Others" not in output  # the summary table does not invent classifier scores


def test_major_project_publisher_reports_measured_cli_demo_counts_only():
    from scripts.publish_major_project_results import build_cli_section, build_readme_section

    metrics = {
        "cli_examples": {
            "confidence_threshold": 0.2,
            "examples": {
                "16-QAM": {
                    "ground_truth_signals": 1,
                    "accepted_detections": 0,
                    "class_and_iou50_matches": 0,
                    "pam_detections": 0,
                    "pam_subtype_predictions": 0,
                    "cli_seconds_per_segment": 29.5,
                }
            },
        }
    }
    assert "| 16-QAM | 1 | 0 | 0 | 0 | 0 | 29.5 |" in build_cli_section(metrics)
    assert "16-QAM: 0 boxes, 0 PAM boxes, 0 subtype outputs" in build_readme_section(metrics)
