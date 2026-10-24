from scripts.publish_reduced_results import (
    END, START, build_split_audit, format_target, matrix, pct, support, update_readme,
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
