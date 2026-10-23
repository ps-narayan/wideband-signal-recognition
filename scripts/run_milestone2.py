"""Generate synthetic data, train YOLOv5n and PAMCNN, and evaluate held-out recordings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict

import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wideband_recognition.anchor_preflight import YOLOV5N_V7_ANCHORS_PX, best_possible_recall
from wideband_recognition.baseline import energy_proposals
from wideband_recognition.dataset import (
    grouped_split, metadata_box, segment_snr_bin, write_dataset_yaml, write_segment_artifacts,
)
from wideband_recognition.extraction import extract_frames
from wideband_recognition.geometry import pixels_to_box
from wideband_recognition.layout import resolution_statuses
from wideband_recognition.metrics import (
    average_precision, calibrate_confidence, class_map50, detection_stats, iou_xyxy,
    match_image, recording_bootstrap,
)
from wideband_recognition.pam_model import PAMCNN, PAM_LABELS
from wideband_recognition.signals import DETECTOR_CLASSES, NON_PAM, PAM_SUBTYPES, generate_recording
from wideband_recognition.synthetic_data import sample_specs


YOLOV5_REPOSITORY = "https://github.com/ultralytics/yolov5.git"
YOLOV5_COMMIT = "f9578796b8a64889aa9e66dc06914d19f40d338e"
IMAGE_SIZE = 640
N_SAMPLES = 1_000_000
FRAME_SAMPLES = 1024
SEED = 2026


def _run(command, *, cwd=None, log_path=None, env=None):
    """Run a command while echoing and optionally retaining its complete log."""
    print("+", " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(
        list(map(str, command)), cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    handle = open(log_path, "w", encoding="utf-8") if log_path else None
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            if handle:
                handle.write(line)
        code = process.wait()
    finally:
        if handle:
            handle.close()
    if code:
        raise subprocess.CalledProcessError(code, command)


def _check_yolo(yolo_dir: Path) -> dict:
    sha = subprocess.check_output(
        ["git", "-C", str(yolo_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if sha != YOLOV5_COMMIT:
        raise RuntimeError(f"YOLOv5 revision mismatch: expected {YOLOV5_COMMIT}, got {sha}")
    license_text = (yolo_dir / "LICENSE").read_text(encoding="utf-8")
    if "GNU AFFERO GENERAL PUBLIC LICENSE" not in license_text or "Version 3" not in license_text:
        raise RuntimeError("Pinned YOLOv5 LICENSE did not match the verified AGPL version")
    train_help = subprocess.run(
        [sys.executable, "train.py", "--help"], cwd=yolo_dir, text=True, capture_output=True
    )
    detect_help = subprocess.run(
        [sys.executable, "detect.py", "--help"], cwd=yolo_dir, text=True, capture_output=True
    )
    if train_help.returncode or detect_help.returncode:
        raise RuntimeError("Pinned YOLOv5 train.py/detect.py help checks failed")
    for option in (
        "--cfg", "--weights", "--epochs", "--img", "--hyp", "--noautoanchor",
        "--batch-size", "--data", "--device", "--workers", "--project",
        "--name", "--exist-ok", "--seed", "--save-period",
    ):
        if option not in train_help.stdout:
            raise RuntimeError(f"Pinned train.py is missing {option}")
    for option in (
        "--source", "--save-txt", "--save-conf", "--conf-thres", "--weights",
        "--img", "--iou-thres", "--max-det", "--device", "--project",
        "--name", "--exist-ok", "--nosave",
    ):
        if option not in detect_help.stdout:
            raise RuntimeError(f"Pinned detect.py is missing {option}")
    train_source = (yolo_dir / "train.py").read_text(encoding="utf-8")
    anchor_source = (yolo_dir / "utils" / "autoanchor.py").read_text(encoding="utf-8")
    model_path = yolo_dir / "models" / "yolov5n.yaml"
    if not model_path.is_file() or "check_anchors" not in train_source:
        raise RuntimeError("Pinned YOLOv5 train/AutoAnchor API did not match the verified interface")
    if "def check_anchors" not in anchor_source or "Best Possible Recall" not in anchor_source:
        raise RuntimeError("Pinned YOLOv5 AutoAnchor implementation could not be verified")
    if "bpr > 0.98" not in anchor_source:
        raise RuntimeError("Pinned YOLOv5 AutoAnchor BPR cutoff changed; review before training")
    model_config = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    anchors = np.asarray(model_config.get("anchors", []), dtype=np.float64).reshape(-1, 2)
    if anchors.shape != YOLOV5N_V7_ANCHORS_PX.shape or not np.array_equal(anchors, YOLOV5N_V7_ANCHORS_PX):
        raise RuntimeError("Pinned YOLOv5n anchors changed; recalculate the minimum-box preflight")
    minimum_wh = [8192 / 999_936 * IMAGE_SIZE, 0.008 * IMAGE_SIZE]
    return {
        "repository": YOLOV5_REPOSITORY,
        "commit": sha,
        "license": "AGPL-3.0",
        "license_verified": True,
        "train_cli_verified": True,
        "detect_cli_verified": True,
        "autoanchor_api_verified": True,
        "autoanchor_training_enabled": True,
        "minimum_box_width_height_pixels": minimum_wh,
        "default_anchor_bpr_for_minimum_box": best_possible_recall(
            tuple(minimum_wh), anchors_px=anchors
        ),
        "default_anchors_px": anchors.tolist(),
        "autoanchor_note": (
            "Pinned YOLOv5 train.py calls check_anchors unless --noautoanchor is set. "
            "It attempts anchor evolution when Best Possible Recall is <= 0.98. "
            "This ratio-fit check passes for the minimum raster box, but the box remains "
            "smaller than the stride-eight feature cell, so localization recall must be measured."
        ),
    }


def _main_manifest(n_recordings, segments_per_recording, seed):
    split_map = grouped_split(n_recordings, seed=seed)
    rows = []
    for recording_id in sorted(split_map):
        recording_index = int(recording_id.rsplit("-", 1)[1])
        for segment_index in range(segments_per_recording):
            rows.append({
                "segment_id": f"{recording_id}-s{segment_index:02d}",
                "recording_id": recording_id,
                "split": split_map[recording_id],
                "profile": "train",
                "seed": seed * 100_000 + recording_index * segments_per_recording + segment_index,
            })
    return rows, split_map


def _shift_manifest(n_recordings, segments_per_recording, seed):
    return [
        {
            "segment_id": f"shift-{recording_index:03d}-s{segment_index:02d}",
            "recording_id": f"shift-{recording_index:03d}",
            "split": "shifted",
            "profile": "shifted",
            "seed": seed * 200_000 + 90_000_000 + recording_index * segments_per_recording + segment_index,
        }
        for recording_index in range(n_recordings)
        for segment_index in range(segments_per_recording)
    ]


def _load_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _label_boxes(path: Path) -> list[dict]:
    result = []
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        class_id, cx, cy, width, height = map(float, line.split())
        result.append({
            "class": DETECTOR_CLASSES[int(class_id)],
            "xyxy": [
                (cx - width / 2) * IMAGE_SIZE, (cy - height / 2) * IMAGE_SIZE,
                (cx + width / 2) * IMAGE_SIZE, (cy + height / 2) * IMAGE_SIZE,
            ],
        })
    return result


def _prediction_boxes(path: Path) -> list[dict]:
    result = []
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        values = list(map(float, line.split()))
        if len(values) not in (5, 6):
            continue
        class_id, cx, cy, width, height = values[:5]
        confidence = values[5] if len(values) == 6 else 1.0
        result.append({
            "class": DETECTOR_CLASSES[int(class_id)],
            "xyxy": [
                (cx - width / 2) * IMAGE_SIZE, (cy - height / 2) * IMAGE_SIZE,
                (cx + width / 2) * IMAGE_SIZE, (cy + height / 2) * IMAGE_SIZE,
            ],
            "confidence": confidence,
        })
    return result


def _make_items(rows, data_dir: Path, prediction_dir: Path | None = None):
    items, metadata_by_segment = [], {}
    for row in rows:
        metadata = _load_metadata(
            data_dir / "metadata" / row["split"] / f"{row['segment_id']}.json"
        )
        metadata_by_segment[row["segment_id"]] = metadata
        label_path = data_dir / "labels" / row["split"] / f"{row['segment_id']}.txt"
        pred_path = prediction_dir / f"{row['segment_id']}.txt" if prediction_dir else None
        snr = float(metadata["measured_global_snr_db"])
        items.append({
            "segment_id": row["segment_id"],
            "recording_id": row["recording_id"],
            "global_snr_db": snr,
            "snr_bin": segment_snr_bin(snr),
            "gt": _label_boxes(label_path),
            "pred": _prediction_boxes(pred_path) if pred_path else [],
        })
    return items, metadata_by_segment


def _energy_items(items, data_dir: Path, *, mask_threshold: float):
    result = []
    for item in items:
        split = "val" if (data_dir / "images" / "val" / f"{item['segment_id']}.png").exists() else (
            "shifted" if (data_dir / "images" / "shifted" / f"{item['segment_id']}.png").exists() else "test"
        )
        path = data_dir / "images" / split / f"{item['segment_id']}.png"
        result.append({**item, "pred": energy_proposals(path, mask_threshold=mask_threshold)})
    return result


def _calibrate_energy(data_dir: Path, val_rows):
    val_items, _ = _make_items(val_rows, data_dir)
    candidates = []
    for mask_threshold in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
        proposal_items = _energy_items(val_items, data_dir, mask_threshold=mask_threshold)
        calibration = calibrate_confidence(
            proposal_items, target_precision=0.80, class_agnostic=True
        )
        candidates.append((mask_threshold, calibration))
    feasible = [row for row in candidates if row[1]["attained_target"]]
    if feasible:
        return max(feasible, key=lambda row: (row[1]["recall"], row[1]["precision"]))
    return min(candidates, key=lambda row: (abs(row[1]["precision"] - 0.80), -row[1]["recall"]))


def _filtered_items(items, *, class_name=None, snr_bin=None):
    result = []
    for item in items:
        if snr_bin is not None and item["snr_bin"] != snr_bin:
            continue
        result.append({
            **{k: v for k, v in item.items() if k not in {"gt", "pred"}},
            "gt": [g for g in item["gt"] if class_name is None or g["class"] == class_name],
            "pred": [p for p in item["pred"] if class_name is None or p["class"] == class_name],
        })
    return result


def _bootstrap_detection(items, metric, threshold, iterations):
    # Match once, then resample additive recording-level counts for speed.
    by_recording = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0})
    for item in items:
        selected = [pred for pred in item["pred"] if pred["confidence"] >= threshold]
        matches, missed, extras = match_image(
            item["gt"], selected, iou_threshold=0.5, class_agnostic=True
        )
        row = by_recording[item["recording_id"]]
        row["tp"] += len(matches)
        row["fp"] += len(extras)
        row["fn"] += len(missed)
        row["iou_sum"] += sum(value[2] for value in matches)
    groups = list(by_recording.values())

    def statistic(sample):
        tp = sum(row["tp"] for row in sample)
        fp = sum(row["fp"] for row in sample)
        fn = sum(row["fn"] for row in sample)
        iou_sum = sum(row["iou_sum"] for row in sample)
        if metric == "precision":
            return tp / (tp + fp) if tp + fp else 0.0
        if metric == "recall":
            return tp / (tp + fn) if tp + fn else 0.0
        return iou_sum / tp if tp else 0.0

    estimate = statistic(groups)
    rng = np.random.default_rng(2026)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample_ids = rng.integers(0, len(groups), size=len(groups)) if groups else []
        samples[index] = statistic([groups[int(i)] for i in sample_ids])
    low, high = np.quantile(samples, [0.025, 0.975]) if iterations else (estimate, estimate)
    count = sum(len(item["gt"]) for item in items)
    return {
        "estimate": estimate, "ci95": [float(low), float(high)],
        "recordings": len(groups), "signals": count, "low_support": count < 30,
    }


def _binary_rate_summary(rows, *, class_filter=None, snr_bin=None, iterations=500):
    selected = [
        row for row in rows
        if (class_filter is None or row["detector_class"] == class_filter)
        and (snr_bin is None or row["snr_bin"] == snr_bin)
    ]
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["recording_id"]].append(row)
    groups = [{"rows": values} for values in grouped.values()]
    def statistic(sample):
        flat = [row for group in sample for row in group["rows"]]
        return sum(bool(row["correct"]) for row in flat) / len(flat) if flat else 0.0
    estimate, low, high = recording_bootstrap(groups, statistic, iterations=iterations)
    return {
        "rate": estimate, "ci95": [low, high], "recordings": len(groups),
        "signals": len(selected), "low_support": len(selected) < 30,
    }


def _accuracy_summary(rows, *, snr_bin=None, label_filter=None, iterations=500):
    selected = [
        row for row in rows
        if (snr_bin is None or row["snr_bin"] == snr_bin)
        and (label_filter is None or row["label"] == label_filter)
    ]
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["recording_id"]].append(row)
    groups = [{"rows": values} for values in grouped.values()]
    def statistic(sample):
        flat = [row for group in sample for row in group["rows"]]
        return sum(bool(row["correct"]) for row in flat) / len(flat) if flat else 0.0
    estimate, low, high = recording_bootstrap(groups, statistic, iterations=iterations)
    return {
        "accuracy": estimate, "ci95": [low, high], "recordings": len(groups),
        "signals": len(selected), "low_support": len(selected) < 30,
    }

def _build_frame_set(rows, data_dir: Path, *, jitter: bool):
    frames, labels, frame_rows = [], [], []
    for row in rows:
        metadata = _load_metadata(data_dir / "metadata" / row["split"] / f"{row['segment_id']}.json")
        specs, _ = sample_specs(
            int(metadata["seed"]), profile=metadata["profile"], n_samples=int(metadata["n_samples"])
        )
        recording = generate_recording(
            specs, int(metadata["n_samples"]), int(metadata["seed"]),
            float(metadata["requested_global_snr_db"]),
        )
        for signal_index, signal_meta in enumerate(metadata["signals"]):
            # The agreed Others class is limited to AM-DSB, FM, 2-FSK, and OOK.
            if signal_meta["detector_class"] == "OFDM":
                continue
            box = metadata_box(signal_meta)
            target = signal_meta["modulation"] if signal_meta["detector_class"] == "PAM" else "Others"
            signal_frames = extract_frames(
                recording.iq, box, frame_samples=FRAME_SAMPLES, n_frames=2,
                seed=int(metadata["seed"]) + signal_index * 7919, jitter=jitter,
            )
            for frame in signal_frames:
                frames.append(frame)
                labels.append(PAM_LABELS.index(target))
                frame_rows.append({
                    "recording_id": metadata["recording_id"],
                    "segment_id": metadata["segment_id"],
                    "signal_id": f"{metadata['segment_id']}:{signal_index}",
                    "label": target,
                    "detector_class": signal_meta["detector_class"],
                    "global_snr_db": float(metadata["measured_global_snr_db"]),
                    "snr_bin": segment_snr_bin(float(metadata["measured_global_snr_db"])),
                    "inband_snr_db": float(signal_meta["approximate_inband_snr_db"]),
                })
    if not frames:
        raise RuntimeError("PAM frame set is empty")
    return np.asarray(frames, dtype=np.float32), np.asarray(labels, dtype=np.int64), frame_rows


def _train_pam(
    train_data, val_data, checkpoint_path: Path, *, epochs: int, batch_size: int,
    seed: int, device: str,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    train_x, train_y, _ = train_data
    val_x, val_y, _ = val_data
    torch.manual_seed(seed)
    device_obj = torch.device(device)
    model = PAMCNN().to(device_obj)
    x_train = torch.from_numpy(train_x.transpose(0, 2, 1))
    y_train = torch.from_numpy(train_y)
    x_val = torch.from_numpy(val_x.transpose(0, 2, 1))
    y_val = torch.from_numpy(val_y)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0
    )
    counts = np.bincount(train_y, minlength=len(PAM_LABELS)).astype(np.float32)
    weights = np.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device_obj)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"train_loss": [], "validation_loss": [], "validation_accuracy": [], "class_counts": counts.tolist()}
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device_obj), batch_y.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            logits = model(x_val.to(device_obj))
            val_loss = float(criterion(logits, y_val.to(device_obj)).cpu())
            val_accuracy = float((logits.argmax(1).cpu() == y_val).float().mean())
        history["train_loss"].append(float(np.mean(losses)))
        history["validation_loss"].append(val_loss)
        history["validation_accuracy"].append(val_accuracy)
        print(
            f"PAM epoch {epoch + 1}/{epochs}: train_loss={history['train_loss'][-1]:.4f} "
            f"val_loss={val_loss:.4f} val_frame_accuracy={val_accuracy:.4f}",
            flush=True,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break
    if best_state is None:
        raise RuntimeError("PAM training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "labels": list(PAM_LABELS),
            "frame_samples": FRAME_SAMPLES,
            "seed": seed,
            "history": history,
        },
        checkpoint_path,
    )
    return model, history


def _predict_frame_labels(model, frames, *, device: str, batch_size=512):
    import torch

    model.eval()
    device_obj = torch.device(device)
    x = torch.from_numpy(np.asarray(frames).transpose(0, 2, 1))
    result = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(x[start : start + batch_size].to(device_obj))
            result.extend(logits.argmax(1).cpu().numpy().tolist())
    return np.asarray(result, dtype=np.int64)


def _signal_level_predictions(frame_predictions, frame_rows):
    grouped = defaultdict(list)
    for pred, row in zip(frame_predictions, frame_rows):
        grouped[row["signal_id"]].append((row, int(pred)))
    result = []
    for values in grouped.values():
        row = values[0][0]
        guess = Counter(pred for _, pred in values).most_common(1)[0][0]
        result.append({
            **row,
            "predicted": PAM_LABELS[guess],
            "correct": PAM_LABELS[guess] == row["label"],
        })
    return result


def _confusion(true_labels, predicted_labels, names):
    mapping = {name: index for index, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    for actual, predicted in zip(true_labels, predicted_labels):
        if actual in mapping and predicted in mapping:
            matrix[mapping[actual], mapping[predicted]] += 1
    return matrix


def _save_confusion(matrix, names, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, str(matrix[y, x]), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_pam_curves(history, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["validation_loss"], label="validation")
    axes[0].set_title("PAM CNN loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["validation_accuracy"], label="validation frames")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Validation frame accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _snr_sort_key(label: str) -> float:
    if label.startswith("["):
        return float(label[1:].split(",", 1)[0])
    if label.startswith("<"):
        return float(label[1:]) - 1e-9
    if label.startswith(">="):
        return float(label[2:])
    raise ValueError(f"unrecognized SNR bin: {label}")


def _save_detector_snr_plot(result, path: Path, title: str):
    bins = sorted(result, key=_snr_sort_key)
    if not bins:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric in ("precision", "recall", "mean_iou_true_positives"):
        estimates = np.asarray([result[key][metric]["estimate"] for key in bins], dtype=float)
        intervals = [result[key][metric].get("ci95", [value, value]) for key, value in zip(bins, estimates)]
        lower = np.maximum(0.0, estimates - np.asarray([interval[0] for interval in intervals], dtype=float))
        upper = np.maximum(0.0, np.asarray([interval[1] for interval in intervals], dtype=float) - estimates)
        ax.errorbar(bins, estimates, yerr=np.vstack((lower, upper)), marker="o", capsize=3, label=metric)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Global SNR bin (dB)")
    ax.set_ylabel("Metric")
    ax.set_title(
        title + "\nGlobal SNR = 10 log10(total clean signal power / total added noise power across full I/Q segment)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _score_test_pipeline(
    items, metadata_by_segment, confidence_threshold, model, device, *, iterations=500,
):
    """Compute oracle PAM, detector-box PAM, false-class Others, and joint correctness."""
    oracle_rows, detector_pam_rows = [], []
    e2e_rows = []
    false_counts = defaultdict(lambda: [0, 0])
    false_class_rows = []
    unresolved_detections = []
    detector_confusion_true, detector_confusion_pred = [], []

    for item in items:
        metadata = metadata_by_segment[item["segment_id"]]
        specs, _ = sample_specs(
            int(metadata["seed"]), profile=metadata["profile"], n_samples=int(metadata["n_samples"])
        )
        recording = generate_recording(
            specs, int(metadata["n_samples"]), int(metadata["seed"]),
            float(metadata["requested_global_snr_db"]),
        )
        gt_metadata = metadata["signals"]
        gt_boxes = item["gt"]
        pam_spec = next(signal for signal in gt_metadata if signal["detector_class"] == "PAM")
        pam_box = metadata_box(pam_spec)
        pam_index = gt_metadata.index(pam_spec)
        oracle_frames = extract_frames(
            recording.iq, pam_box, n_frames=2,
            seed=int(metadata["seed"]) + pam_index * 7919, jitter=False,
        )
        oracle_guesses = _predict_frame_labels(model, oracle_frames, device=device)
        oracle_name = PAM_LABELS[Counter(oracle_guesses.tolist()).most_common(1)[0][0]]
        oracle_rows.append({
            "recording_id": item["recording_id"], "signal_id": item["segment_id"] + ":PAM",
            "label": pam_spec["modulation"], "predicted": oracle_name,
            "correct": oracle_name == pam_spec["modulation"], "snr_bin": item["snr_bin"],
            "global_snr_db": item["global_snr_db"],
            "inband_snr_db": float(pam_spec["approximate_inband_snr_db"]),
        })

        predictions = [p for p in item["pred"] if p["confidence"] >= confidence_threshold]
        subtype_predictions = {}
        candidate_frames = {}
        prediction_boxes = {}
        for index, pred in enumerate(predictions):
            if pred["class"] not in DETECTOR_CLASSES:
                continue
            try:
                prediction_boxes[index] = pixels_to_box(
                    tuple(pred["xyxy"]), int(metadata["covered_samples"]),
                    IMAGE_SIZE, IMAGE_SIZE, class_name=pred["class"],
                )
            except ValueError:
                continue
        box_indices = sorted(prediction_boxes)
        box_status = resolution_statuses([prediction_boxes[index] for index in box_indices])
        unresolved_indices = {
            index for index, status in zip(box_indices, box_status) if status == "unresolved"
        }
        for index in sorted(unresolved_indices):
            pred = predictions[index]
            unresolved_detections.append({
                "recording_id": item["recording_id"],
                "segment_id": item["segment_id"],
                "prediction_index": index,
                "detector_class": pred["class"],
                "xyxy_pixels": pred["xyxy"],
                "status": "unresolved",
            })
        for index, pred in enumerate(predictions):
            if pred["class"] not in {"PAM", *NON_PAM} or index in unresolved_indices:
                continue
            box = prediction_boxes.get(index)
            if box is None:
                continue
            try:
                frames = extract_frames(
                    recording.iq, box, n_frames=2,
                    seed=int(metadata["seed"]) + index * 104729, jitter=False,
                )
            except ValueError:
                continue
            candidate_frames[index] = frames
        if candidate_frames:
            candidate_indices = sorted(candidate_frames)
            flat_frames = np.concatenate([candidate_frames[index] for index in candidate_indices], axis=0)
            flat_guesses = _predict_frame_labels(model, flat_frames, device=device)
            for position, index in enumerate(candidate_indices):
                guesses = flat_guesses[position * 2 : position * 2 + 2]
                subtype_predictions[index] = PAM_LABELS[Counter(guesses.tolist()).most_common(1)[0][0]]
        for index, predicted_subtype in subtype_predictions.items():
            pred = predictions[index]
            if pred["class"] in NON_PAM:
                is_others = predicted_subtype == "Others"
                false_counts[pred["class"]][1] += 1
                false_counts[pred["class"]][0] += int(is_others)
                false_class_rows.append({
                    "recording_id": item["recording_id"],
                    "detector_class": pred["class"],
                    "snr_bin": item["snr_bin"],
                    "correct": is_others,
                })

        used_gt = set()
        used_predictions = set()
        correct_count = 0
        for pred_index in sorted(range(len(predictions)), key=lambda i: predictions[i]["confidence"], reverse=True):
            pred = predictions[pred_index]
            overlaps = [
                (iou_xyxy(gt["xyxy"], pred["xyxy"]), gt_index)
                for gt_index, gt in enumerate(gt_boxes) if gt_index not in used_gt
            ]
            if not overlaps:
                continue
            overlap, gt_index = max(overlaps)
            if overlap < 0.5:
                continue
            used_gt.add(gt_index)
            used_predictions.add(pred_index)
            gt = gt_boxes[gt_index]
            gt_meta = next(signal for signal in gt_metadata if signal["detector_class"] == gt["class"])
            correct = pred["class"] == gt["class"]
            if correct and gt["class"] == "PAM":
                correct = subtype_predictions.get(pred_index) == gt_meta["modulation"]
                detector_pam_rows.append({
                    "recording_id": item["recording_id"], "signal_id": item["segment_id"] + ":PAM",
                    "label": gt_meta["modulation"], "predicted": subtype_predictions.get(pred_index, "unavailable"),
                    "correct": bool(correct), "snr_bin": item["snr_bin"],
                    "global_snr_db": item["global_snr_db"],
                })
            correct_count += int(correct)
            detector_confusion_true.append(gt["class"])
            detector_confusion_pred.append(pred["class"])
        for gt_index, gt in enumerate(gt_boxes):
            if gt_index not in used_gt:
                detector_confusion_true.append(gt["class"])
                detector_confusion_pred.append("missed")
        for pred_index, pred in enumerate(predictions):
            if pred_index not in used_predictions:
                detector_confusion_true.append("background")
                detector_confusion_pred.append(pred["class"])
        e2e_rows.append({
            "recording_id": item["recording_id"], "snr_bin": item["snr_bin"],
            "global_snr_db": item["global_snr_db"], "correct": correct_count,
            "predictions": len(predictions), "ground_truth": len(gt_boxes),
        })

    total_correct = sum(row["correct"] for row in e2e_rows)
    total_pred = sum(row["predictions"] for row in e2e_rows)
    total_gt = sum(row["ground_truth"] for row in e2e_rows)
    grouped = defaultdict(list)
    for row in e2e_rows:
        grouped[row["recording_id"]].append(row)
    groups = [{"rows": values} for values in grouped.values()]

    def ratio(sample, numerator, denominator):
        rows = [row for group in sample for row in group["rows"]]
        top, bottom = sum(row[numerator] for row in rows), sum(row[denominator] for row in rows)
        return top / bottom if bottom else 0.0

    precision_ci = recording_bootstrap(
        groups, lambda sample: ratio(sample, "correct", "predictions"), iterations=iterations
    )
    recall_ci = recording_bootstrap(
        groups, lambda sample: ratio(sample, "correct", "ground_truth"), iterations=iterations
    )
    others = sum(value[0] for value in false_counts.values())
    nonpam = sum(value[1] for value in false_counts.values())
    return {
        "oracle_rows": oracle_rows,
        "detector_pam_rows": detector_pam_rows,
        "false_class_rows": false_class_rows,
        "unresolved_detections": unresolved_detections,
        "e2e_rows": e2e_rows,
        "confusion_true": detector_confusion_true,
        "confusion_pred": detector_confusion_pred,
        "false_class_others_rate": {
            "rate": others / nonpam if nonpam else 0.0,
            "others": others, "total": nonpam,
            "per_non_pam_detector_class": {
                name: {"others": value[0], "total": value[1], "rate": value[0] / value[1] if value[1] else 0.0}
                for name, value in false_counts.items()
            },
        },
        "end_to_end": {
            "precision": total_correct / total_pred if total_pred else 0.0,
            "precision_ci95": list(precision_ci[1:]),
            "recall": total_correct / total_gt if total_gt else 0.0,
            "recall_ci95": list(recall_ci[1:]),
            "correct": total_correct, "predictions": total_pred, "ground_truth": total_gt,
            "recordings": len(groups),
            "signals": total_gt,
            "low_support": total_gt < 30,
        },
    }

def _bootstrap_ap(items, class_name, iterations):
    grouped = defaultdict(list)
    for item in items:
        grouped[item["recording_id"]].append(item)
    groups = [{"items": values} for values in grouped.values()]
    def statistic(sample):
        flat = [item for group in sample for item in group["items"]]
        return average_precision(flat, class_name)[0]
    estimate, low, high = recording_bootstrap(groups, statistic, iterations=iterations)
    signal_count = sum(sum(gt["class"] == class_name for gt in item["gt"]) for item in items)
    return {
        "estimate": estimate, "ci95": [low, high], "recordings": len(groups),
        "signals": signal_count, "low_support": signal_count < 30,
    }



def _bootstrap_map50(items, class_names, iterations):
    grouped = defaultdict(list)
    for item in items:
        grouped[item["recording_id"]].append(item)
    groups = [{"items": values} for values in grouped.values()]
    def statistic(sample):
        flat = [item for group in sample for item in group["items"]]
        return class_map50(flat, class_names)["map50"]
    estimate, low, high = recording_bootstrap(groups, statistic, iterations=iterations)
    signals = sum(len(item["gt"]) for item in items)
    return {
        "estimate": estimate, "ci95": [low, high], "recordings": len(groups),
        "signals": signals, "low_support": signals < 30,
    }


def _save_pam_snr_plot(oracle_by_snr, detector_by_snr, path: Path):
    bins = sorted(set(oracle_by_snr) | set(detector_by_snr), key=_snr_sort_key)
    fig, ax = plt.subplots(figsize=(10, 5))
    for values, label, marker in (
        (oracle_by_snr, "oracle boxes", "o"), (detector_by_snr, "detector boxes", "s")
    ):
        estimates = np.asarray([values.get(key, {}).get("accuracy", np.nan) for key in bins], dtype=float)
        intervals = [values.get(key, {}).get("ci95", [value, value]) for key, value in zip(bins, estimates)]
        lower = np.maximum(0.0, estimates - np.asarray([interval[0] for interval in intervals], dtype=float))
        upper = np.maximum(0.0, np.asarray([interval[1] for interval in intervals], dtype=float) - estimates)
        ax.errorbar(bins, estimates, yerr=np.vstack((lower, upper)), marker=marker, capsize=3, label=label)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Global SNR bin (dB)")
    ax.set_ylabel("PAM subtype accuracy")
    ax.set_title(
        "PAM accuracy versus SNR\n"
        "Global SNR = 10 log10(total clean signal power / total added noise power across full I/Q segment)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _end_to_end_by_snr(rows, iterations):
    result = {}
    for snr_bin in sorted({row["snr_bin"] for row in rows}, key=_snr_sort_key):
        selected = [row for row in rows if row["snr_bin"] == snr_bin]
        grouped = defaultdict(list)
        for row in selected:
            grouped[row["recording_id"]].append(row)
        groups = [{"rows": values} for values in grouped.values()]
        def precision(sample):
            flat = [r for g in sample for r in g["rows"]]
            denom = sum(r["predictions"] for r in flat)
            return sum(r["correct"] for r in flat) / denom if denom else 0.0
        def recall(sample):
            flat = [r for g in sample for r in g["rows"]]
            denom = sum(r["ground_truth"] for r in flat)
            return sum(r["correct"] for r in flat) / denom if denom else 0.0
        p = recording_bootstrap(groups, precision, iterations=iterations)
        r = recording_bootstrap(groups, recall, iterations=iterations)
        signal_count = sum(row["ground_truth"] for row in selected)
        support = {
            "recordings": len(groups), "signals": signal_count,
            "low_support": signal_count < 30,
        }
        result[snr_bin] = {
            "precision": {"estimate": p[0], "ci95": list(p[1:]), **support},
            "recall": {"estimate": r[0], "ci95": list(r[1:]), **support},
            **support,
        }
    return result


def _save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")



def _load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return config


def _check_config(config: dict) -> None:
    fixed = {
        "segment_samples": (config["data"]["segment_samples"], N_SAMPLES),
        "image_size": (config["data"]["image_size"], IMAGE_SIZE),
        "frame_samples": (config["pam_classifier"]["frame_samples"], FRAME_SAMPLES),
    }
    mismatches = [
        f"{name}: config={value}, implementation={expected}"
        for name, (value, expected) in fixed.items() if value != expected
    ]
    if mismatches:
        raise ValueError("Config disagrees with implemented physical dimensions: " + "; ".join(mismatches))

def _write_summary(metrics, path: Path):
    main = metrics["main_test"]
    shifted = metrics["shifted_test"]

    def fmt(value):
        return "n/a" if value is None else f"{float(value):.4f}"

    def estimate_line(name, row):
        ci = row.get("ci95", [None, None])
        return (
            f"{name}: {fmt(row.get('estimate', row.get('accuracy', row.get('rate'))))}; "
            f"CI95=[{fmt(ci[0])}, {fmt(ci[1])}]; "
            f"recordings={row.get('recordings', 'n/a')}; signals={row.get('signals', 'n/a')}; "
            f"low_support={row.get('low_support', 'n/a')}"
        )

    lines = [
        "Milestone 2 run summary (every metric below was measured by this run)",
        f"YOLOv5 source commit verified: {metrics['yolo']['commit']}",
        f"Output folder: {path.parent.parent}",
        "",
        "VALIDATION OPERATING POINTS (thresholds chosen on validation only)",
        f"YOLO target precision attained: {metrics['validation_calibration']['yolo']['attained_target']}; "
        f"threshold={fmt(metrics['validation_calibration']['yolo']['threshold'])}; "
        f"measured precision={fmt(metrics['validation_calibration']['yolo']['precision'])}",
        f"Energy target precision attained: {metrics['validation_calibration']['energy_baseline']['attained_target']}; "
        f"mask threshold={fmt(metrics['validation_calibration']['energy_mask_threshold'])}; "
        f"confidence threshold={fmt(metrics['validation_calibration']['energy_confidence_threshold'])}; "
        f"measured precision={fmt(metrics['validation_calibration']['energy_baseline']['precision'])}",
        "",
        "MAIN RECORDING-DISJOINT TEST",
    ]
    for model_name, model_data in (("YOLO", main["yolo"]), ("Energy baseline", main["energy_baseline"])):
        lines.append(model_name + " class-agnostic geometry metrics at the frozen validation operating point:")
        for metric in ("precision", "recall", "mean_iou_true_positives"):
            lines.append(estimate_line("  " + metric, model_data["matched_precision_with_recording_bootstrap"][metric]))
    map_all = main["yolo"]["map50_all_snr"]
    map_high = main["yolo"]["map50_global_snr_ge_5_db"]
    lines.append(estimate_line("YOLO mAP@0.5, all SNR", map_all["map50_recording_bootstrap"]))
    lines.append(estimate_line("YOLO mAP@0.5, global SNR at least 5 dB", map_high["map50_recording_bootstrap"]))
    lines.append("YOLO per-class precision, recall, mean IoU, and AP@0.5:")
    for name, row in main["yolo"]["per_class_with_recording_bootstrap"].items():
        lines.append(
            f"  {name}: P={fmt(row['precision']['estimate'])} CI95={row['precision']['ci95']}; "
            f"R={fmt(row['recall']['estimate'])} CI95={row['recall']['ci95']}; "
            f"IoU={fmt(row['mean_iou_true_positives']['estimate'])} CI95={row['mean_iou_true_positives']['ci95']}; "
            f"AP50={fmt(row['ap50']['estimate'])} CI95={row['ap50']['ci95']}; "
            f"recordings={row['ap50']['recordings']}; signals={row['ap50']['signals']}; "
            f"low_support={row['ap50']['low_support']}"
        )
    lines.append("YOLO and energy baseline recall/IoU by global SNR bin:")
    for snr_bin in main["yolo"]["by_global_snr_with_recording_bootstrap"]:
        yolo_row = main["yolo"]["by_global_snr_with_recording_bootstrap"][snr_bin]
        energy_row = main["energy_baseline"]["by_global_snr_with_recording_bootstrap"].get(snr_bin, {})
        def compact(row, metric):
            value = row.get(metric, {})
            ci = value.get("ci95", [None, None])
            return f"{fmt(value.get('estimate'))} CI95=[{fmt(ci[0])},{fmt(ci[1])}]"
        lines.append(
            f"  {snr_bin} dB: YOLO P={compact(yolo_row, 'precision')}, R={compact(yolo_row, 'recall')}, "
            f"IoU={compact(yolo_row, 'mean_iou_true_positives')}; "
            f"energy P={compact(energy_row, 'precision')}, R={compact(energy_row, 'recall')}, "
            f"IoU={compact(energy_row, 'mean_iou_true_positives')}"
        )
    pam = main["pam"]
    lines.extend([
        "",
        "PAM CLASSIFIER",
        estimate_line("Oracle-box subtype accuracy", pam["oracle_accuracy"]),
        estimate_line("Detector-box subtype accuracy (conditional on usable matched detections)", pam["detector_box_accuracy"]),
    ])
    for snr_bin, row in pam["oracle_accuracy_by_global_snr"].items():
        lines.append(estimate_line(f"Oracle PAM accuracy at global SNR {snr_bin} dB", row))
    for subtype, row in pam["oracle_accuracy_by_subtype"].items():
        lines.append(estimate_line(f"Oracle PAM accuracy for {subtype}", row))
    for subtype, row in pam["detector_box_accuracy_by_subtype"].items():
        lines.append(estimate_line(f"Detector-box PAM accuracy for {subtype}", row))
    lines.append(estimate_line("All-signal PAM CNN accuracy (PAM subtypes plus Others)", pam["all_signal_classifier_accuracy"]))
    for label, row in pam["all_signal_classifier_accuracy_by_class"].items():
        lines.append(estimate_line(f"All-signal classifier accuracy for {label}", row))
    lines.append(estimate_line("Non-PAM detections classified as Others", pam["non_pam_detections_classified_as_others"]["overall"]))
    for name, row in pam["non_pam_detections_classified_as_others"]["by_detector_class"].items():
        lines.append(estimate_line(f"  non-PAM class {name} -> Others", row))
    for snr_bin, row in pam["non_pam_detections_classified_as_others"]["by_global_snr"].items():
        lines.append(estimate_line(f"  non-PAM -> Others at global SNR {snr_bin} dB", row))
    lines.append("Approximate PAM in-band SNR dB (per-signal values are also in metadata): " + json.dumps(pam["approximate_pam_inband_snr_db"], sort_keys=True))
    lines.append(estimate_line("End-to-end precision", {
        "estimate": main["end_to_end"]["precision"],
        "ci95": main["end_to_end"]["precision_ci95"],
        "recordings": main["end_to_end"]["recordings"],
        "signals": main["end_to_end"]["signals"],
        "low_support": main["end_to_end"]["low_support"],
    }))
    lines.append(estimate_line("End-to-end recall", {
        "estimate": main["end_to_end"]["recall"],
        "ci95": main["end_to_end"]["recall_ci95"],
        "recordings": main["end_to_end"]["recordings"],
        "signals": main["end_to_end"]["signals"],
        "low_support": main["end_to_end"]["low_support"],
    }))
    for snr_bin, row in main["end_to_end"]["by_global_snr_with_recording_bootstrap"].items():
        lines.append(estimate_line(f"End-to-end precision at global SNR {snr_bin} dB", row["precision"]))
        lines.append(estimate_line(f"End-to-end recall at global SNR {snr_bin} dB", row["recall"]))
    lines.append(f"Unresolved detector boxes skipped for PAM extraction: {len(main['pam']['unresolved_detections'])}")
    lines.append("")
    lines.append("SHIFTED-PARAMETER TEST (not used for threshold selection)")
    lines.append(estimate_line("Shifted YOLO precision", shifted["yolo_matched_precision_with_recording_bootstrap"]["precision"]))
    lines.append(estimate_line("Shifted YOLO recall", shifted["yolo_matched_precision_with_recording_bootstrap"]["recall"]))
    lines.append(estimate_line("Shifted YOLO mean IoU", shifted["yolo_matched_precision_with_recording_bootstrap"]["mean_iou_true_positives"]))
    lines.append(estimate_line("Shifted YOLO mAP@0.5", shifted["yolo_map50"]["map50_recording_bootstrap"]))
    for name, row in shifted["yolo_per_class_with_recording_bootstrap"].items():
        lines.append(
            f"  shifted {name}: P={fmt(row['precision']['estimate'])} CI95={row['precision']['ci95']}; "
            f"R={fmt(row['recall']['estimate'])} CI95={row['recall']['ci95']}; "
            f"IoU={fmt(row['mean_iou_true_positives']['estimate'])} CI95={row['mean_iou_true_positives']['ci95']}; "
            f"AP50={fmt(row['ap50']['estimate'])} CI95={row['ap50']['ci95']}; "
            f"recordings={row['ap50']['recordings']}; signals={row['ap50']['signals']}; "
            f"low_support={row['ap50']['low_support']}"
        )
    lines.append(estimate_line("Shifted energy baseline precision", shifted["energy_baseline_matched_precision_with_recording_bootstrap"]["precision"]))
    lines.append(estimate_line("Shifted energy baseline recall", shifted["energy_baseline_matched_precision_with_recording_bootstrap"]["recall"]))
    lines.append(estimate_line("Shifted energy baseline mean IoU", shifted["energy_baseline_matched_precision_with_recording_bootstrap"]["mean_iou_true_positives"]))
    lines.append(estimate_line("Shifted PAM oracle accuracy", shifted["pam_oracle_accuracy"]))
    lines.append(estimate_line("Shifted PAM detector-box accuracy", shifted["pam_detector_box_accuracy"]))
    lines.append(estimate_line("Shifted end-to-end precision", {
        "estimate": shifted["end_to_end"]["precision"],
        "ci95": shifted["end_to_end"]["precision_ci95"],
        "recordings": shifted["end_to_end"]["recordings"],
        "signals": shifted["end_to_end"]["signals"],
        "low_support": shifted["end_to_end"]["low_support"],
    }))
    lines.append(estimate_line("Shifted end-to-end recall", {
        "estimate": shifted["end_to_end"]["recall"],
        "ci95": shifted["end_to_end"]["recall_ci95"],
        "recordings": shifted["end_to_end"]["recordings"],
        "signals": shifted["end_to_end"]["signals"],
        "low_support": shifted["end_to_end"]["low_support"],
    }))
    lines.append(f"Shifted unresolved detector boxes: {len(shifted['unresolved_detections'])}")
    lines.append("")
    lines.append("ACCEPTANCE TARGETS")
    for name, row in metrics["acceptance_targets"].items():
        if row["measured"] is None:
            lines.append(f"{name}: not measured here; target={row['target']}; {row.get('note', '')}")
            continue
        status = "PASS" if row["pass"] else "FAIL"
        lines.append(
            f"{name}: {status}; target={row['target']}; measured={fmt(row['measured'])}; "
            f"CI95={row.get('ci95', 'n/a')}; support={row.get('support', {})}"
        )
        if not row["pass"]:
            if name.startswith(("recall", "mean_iou", "map50")):
                lines.append("  Next: inspect missed/false boxes; test a higher image resolution or larger minimum signal on validation only, then rerun the frozen tests.")
            elif name.startswith("pam_"):
                lines.append("  Next: inspect the subtype confusion and in-band SNR; improve extraction or training without changing the held-out sets.")
    lines.extend([
        "",
        "Output files: metrics.json, plots, confusion matrices, training/inference logs, checkpoints, and generated dataset are in the Drive output folder.",
        f"Metrics JSON: {metrics['metrics_path']}",
        "Paper results are not project results and are not included.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    import torch

    config = _load_config(args.config)
    _check_config(config)
    yolo_dir = Path(args.yolo_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    data_work = Path(args.data_work).resolve()
    data_work.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Full training requires the Colab T4 GPU; run the local tests and smoke command instead.")
    device = "cuda:0"
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    for directory in (out_dir / "results", out_dir / "plots", out_dir / "checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)
    yolo_info = _check_yolo(yolo_dir)
    _save_json(out_dir / "results" / "yolov5_api_check.json", yolo_info)
    print(
        "Pinned YOLOv5 minimum-box anchor check: "
        + json.dumps({
            "minimum_wh_pixels": yolo_info["minimum_box_width_height_pixels"],
            "default_anchor_bpr": yolo_info["default_anchor_bpr_for_minimum_box"],
            "stride_eight_note": yolo_info["autoanchor_note"],
        }),
        flush=True,
    )

    data_dir = data_work / "dataset"
    if data_dir.exists():
        raise FileExistsError(f"{data_dir} already exists; choose a new --data-work folder.")
    data_dir.mkdir(parents=True)
    main_rows, recording_split = _main_manifest(
        args.recordings, args.segments_per_recording, args.seed
    )
    shifted_rows = _shift_manifest(args.shift_recordings, args.segments_per_recording, args.seed)
    all_rows = main_rows + shifted_rows
    start = time.perf_counter()
    for index, row in enumerate(all_rows, start=1):
        write_segment_artifacts(
            data_dir, row["segment_id"], row["recording_id"], row["split"],
            row["seed"], row["profile"], n_samples=N_SAMPLES, image_size=IMAGE_SIZE,
        )
        if index % 25 == 0 or index == len(all_rows):
            print(f"Generated image/label/metadata artifacts: {index}/{len(all_rows)}", flush=True)
    generation_seconds = time.perf_counter() - start
    data_yaml = write_dataset_yaml(data_dir, data_work / "wideband.yaml")
    split_counts = Counter(recording_split.values())
    _save_json(out_dir / "results" / "dataset_manifest.json", {
        "main": main_rows, "shifted": shifted_rows, "recording_split": recording_split,
    })
    _save_json(out_dir / "results" / "dataset_profile.json", {
        "main_segments": len(main_rows), "shifted_segments": len(shifted_rows),
        "segments_per_recording": args.segments_per_recording,
        "recordings_per_split": dict(split_counts),
        "generation_wall_seconds_colab": generation_seconds,
        "classes_per_segment": list(DETECTOR_CLASSES),
        "iq_retained_on_disk": False,
    })

    # Archive generated data before training so a Colab interruption cannot discard it.
    archived = out_dir / "dataset"
    if archived.exists():
        raise FileExistsError(f"{archived} already exists; choose a new --out-dir.")
    shutil.copytree(data_dir, archived)
    print(f"Archived generated dataset to Drive: {archived}", flush=True)

    yolo_project = out_dir / "checkpoints" / "yolo"
    project_root = Path(__file__).resolve().parents[1]
    hyp_path = project_root / "configs" / "yolov5_hyp.yaml"
    train_command = [
        sys.executable, "train.py", "--img", str(IMAGE_SIZE),
        "--batch-size", str(args.yolo_batch), "--epochs", str(args.yolo_epochs),
        "--data", str(data_yaml), "--weights", "", "--cfg", "models/yolov5n.yaml",
        "--hyp", str(hyp_path),
        "--device", "0", "--workers", "2", "--project", str(yolo_project),
        "--name", "milestone2", "--exist-ok", "--seed", str(args.seed),
        "--save-period", str(max(1, args.yolo_epochs // 5)),
    ]
    yolo_env = os.environ.copy()
    yolo_env["WANDB_MODE"] = "disabled"
    _run(train_command, cwd=yolo_dir, log_path=out_dir / "results" / "yolo_training.log", env=yolo_env)
    training_log = (out_dir / "results" / "yolo_training.log").read_text(encoding="utf-8")
    autoanchor_lines = [
        line for line in training_log.splitlines()
        if "AutoAnchor" in line or "Best Possible Recall" in line
    ]
    if not any("Best Possible Recall" in line for line in autoanchor_lines):
        raise RuntimeError("YOLOv5 training log did not contain the expected AutoAnchor BPR result")
    yolo_info["autoanchor_training_log_excerpt"] = autoanchor_lines
    _save_json(out_dir / "results" / "yolov5_api_check.json", yolo_info)
    weights = yolo_project / "milestone2" / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"YOLOv5 did not save best.pt at {weights}")
    shutil.copy2(weights, out_dir / "checkpoints" / "yolov5n_best.pt")

    prediction_dirs = {}
    for split in ("val", "test", "shifted"):
        source = data_dir / "images" / split
        result_dir = out_dir / "inference" / split
        command = [
            sys.executable, "detect.py", "--weights", str(weights), "--source", str(source),
            "--img", str(IMAGE_SIZE), "--conf-thres", "0.001", "--iou-thres", "0.6",
            "--max-det", "300", "--device", "0", "--save-txt", "--save-conf",
            "--project", str(result_dir.parent), "--name", result_dir.name,
            "--exist-ok", "--nosave",
        ]
        _run(command, cwd=yolo_dir, log_path=out_dir / "results" / f"yolo_detect_{split}.log", env=yolo_env)
        prediction_dirs[split] = result_dir / "labels"

    val_rows = [row for row in main_rows if row["split"] == "val"]
    train_rows = [row for row in main_rows if row["split"] == "train"]
    test_rows = [row for row in main_rows if row["split"] == "test"]
    val_items, _ = _make_items(val_rows, data_dir, prediction_dirs["val"])
    yolo_calibration = calibrate_confidence(val_items, target_precision=0.80, class_agnostic=True)
    energy_mask_threshold, energy_calibration = _calibrate_energy(data_dir, val_rows)
    test_items, test_meta = _make_items(test_rows, data_dir, prediction_dirs["test"])
    shifted_items, shifted_meta = _make_items(shifted_rows, data_dir, prediction_dirs["shifted"])
    test_energy = _energy_items(test_items, data_dir, mask_threshold=energy_mask_threshold)
    shifted_energy = _energy_items(shifted_items, data_dir, mask_threshold=energy_mask_threshold)

    print("Regenerating seed-based I/Q for PAM frame training and evaluation.", flush=True)
    train_data = _build_frame_set(train_rows, data_dir, jitter=True)
    val_data = _build_frame_set(val_rows, data_dir, jitter=False)
    pam_model, pam_history = _train_pam(
        train_data, val_data, out_dir / "checkpoints" / "pam_cnn.pt",
        epochs=args.pam_epochs, batch_size=args.pam_batch, seed=args.seed, device=device,
    )
    _save_json(out_dir / "results" / "pam_training_history.json", pam_history)
    _save_pam_curves(pam_history, out_dir / "plots" / "pam_training_curves.png")

    main_joint = _score_test_pipeline(
        test_items, test_meta, float(yolo_calibration["threshold"]), pam_model, device,
        iterations=args.bootstrap_iterations,
    )
    shifted_joint = _score_test_pipeline(
        shifted_items, shifted_meta, float(yolo_calibration["threshold"]), pam_model, device,
        iterations=args.bootstrap_iterations,
    )
    main_yolo_stats = detection_stats(
        test_items, confidence_threshold=float(yolo_calibration["threshold"]), class_agnostic=True
    )
    main_energy_stats = detection_stats(
        test_energy, confidence_threshold=float(energy_calibration["threshold"]), class_agnostic=True
    )
    shifted_yolo_stats = detection_stats(
        shifted_items, confidence_threshold=float(yolo_calibration["threshold"]), class_agnostic=True
    )
    shifted_energy_stats = detection_stats(
        shifted_energy, confidence_threshold=float(energy_calibration["threshold"]), class_agnostic=True
    )
    metric_names = ("precision", "recall", "mean_iou_true_positives")
    main_yolo_ci = {
        metric: _bootstrap_detection(
            test_items, metric, float(yolo_calibration["threshold"]), args.bootstrap_iterations
        ) for metric in metric_names
    }
    main_energy_ci = {
        metric: _bootstrap_detection(
            test_energy, metric, float(energy_calibration["threshold"]), args.bootstrap_iterations
        ) for metric in metric_names
    }
    shifted_yolo_ci = {
        metric: _bootstrap_detection(
            shifted_items, metric, float(yolo_calibration["threshold"]), args.bootstrap_iterations
        ) for metric in metric_names
    }
    shifted_energy_ci = {
        metric: _bootstrap_detection(
            shifted_energy, metric, float(energy_calibration["threshold"]), args.bootstrap_iterations
        ) for metric in metric_names
    }
    main_map = class_map50(test_items, DETECTOR_CLASSES)
    shifted_map = class_map50(shifted_items, DETECTOR_CLASSES)
    high_snr_items = [item for item in test_items if item["global_snr_db"] >= 5.0]
    high_snr_map = class_map50(high_snr_items, DETECTOR_CLASSES)
    for map_result, map_items in (
        (main_map, test_items), (high_snr_map, high_snr_items), (shifted_map, shifted_items)
    ):
        map_result["map50_recording_bootstrap"] = _bootstrap_map50(
            map_items, DETECTOR_CLASSES, args.bootstrap_iterations
        )
        map_result["per_class_with_recording_bootstrap"] = {
            name: _bootstrap_ap(map_items, name, args.bootstrap_iterations)
            for name in DETECTOR_CLASSES
        }
        map_result["per_class"] = map_result["per_class_with_recording_bootstrap"]

    per_class = {}
    for name in DETECTOR_CLASSES:
        class_items = _filtered_items(test_items, class_name=name)
        stats = detection_stats(
            class_items, confidence_threshold=float(yolo_calibration["threshold"]),
            class_agnostic=False,
        )
        per_class[name] = {
            "precision": _bootstrap_detection(class_items, "precision", float(yolo_calibration["threshold"]), args.bootstrap_iterations),
            "recall": _bootstrap_detection(class_items, "recall", float(yolo_calibration["threshold"]), args.bootstrap_iterations),
            "mean_iou_true_positives": _bootstrap_detection(
                class_items, "mean_iou_true_positives", float(yolo_calibration["threshold"]), args.bootstrap_iterations
            ),
            "ap50": _bootstrap_ap(class_items, name, args.bootstrap_iterations),
            "tp": stats["tp"], "fp": stats["fp"], "fn": stats["fn"],
        }

    shifted_per_class = {}
    for name in DETECTOR_CLASSES:
        class_items = _filtered_items(shifted_items, class_name=name)
        stats = detection_stats(
            class_items, confidence_threshold=float(yolo_calibration["threshold"]),
            class_agnostic=False,
        )
        shifted_per_class[name] = {
            "precision": _bootstrap_detection(class_items, "precision", float(yolo_calibration["threshold"]), args.bootstrap_iterations),
            "recall": _bootstrap_detection(class_items, "recall", float(yolo_calibration["threshold"]), args.bootstrap_iterations),
            "mean_iou_true_positives": _bootstrap_detection(
                class_items, "mean_iou_true_positives", float(yolo_calibration["threshold"]), args.bootstrap_iterations
            ),
            "ap50": _bootstrap_ap(class_items, name, args.bootstrap_iterations),
            "tp": stats["tp"], "fp": stats["fp"], "fn": stats["fn"],
        }

    detector_by_snr, baseline_by_snr = {}, {}
    for snr_bin in sorted({item["snr_bin"] for item in test_items}):
        subset = [item for item in test_items if item["snr_bin"] == snr_bin]
        baseline_subset = [item for item in test_energy if item["snr_bin"] == snr_bin]
        detector_by_snr[snr_bin], baseline_by_snr[snr_bin] = {}, {}
        for metric in ("precision", "recall", "mean_iou_true_positives"):
            detector_by_snr[snr_bin][metric] = _bootstrap_detection(
                subset, metric, float(yolo_calibration["threshold"]), args.bootstrap_iterations
            )
            baseline_by_snr[snr_bin][metric] = _bootstrap_detection(
                baseline_subset, metric, float(energy_calibration["threshold"]), args.bootstrap_iterations
            )
    shifted_by_snr, shifted_baseline_by_snr = {}, {}
    for snr_bin in sorted({item["snr_bin"] for item in shifted_items}):
        subset = [item for item in shifted_items if item["snr_bin"] == snr_bin]
        baseline_subset = [item for item in shifted_energy if item["snr_bin"] == snr_bin]
        shifted_by_snr[snr_bin] = {
            metric: _bootstrap_detection(
                subset, metric, float(yolo_calibration["threshold"]), args.bootstrap_iterations
            )
            for metric in ("precision", "recall", "mean_iou_true_positives")
        }
        shifted_baseline_by_snr[snr_bin] = {
            metric: _bootstrap_detection(
                baseline_subset, metric, float(energy_calibration["threshold"]), args.bootstrap_iterations
            )
            for metric in ("precision", "recall", "mean_iou_true_positives")
        }
    _save_detector_snr_plot(
        detector_by_snr, out_dir / "plots" / "yolo_metrics_vs_snr.png", "YOLOv5n detector"
    )
    _save_detector_snr_plot(
        baseline_by_snr, out_dir / "plots" / "energy_baseline_metrics_vs_snr.png",
        "Energy geometry baseline",
    )
    _save_detector_snr_plot(
        shifted_by_snr, out_dir / "plots" / "yolo_metrics_vs_snr_shifted.png",
        "YOLOv5n detector, shifted test",
    )
    _save_detector_snr_plot(
        shifted_baseline_by_snr, out_dir / "plots" / "energy_baseline_metrics_vs_snr_shifted.png",
        "Energy geometry baseline, shifted test",
    )

    oracle_by_snr, oracle_by_class = {}, {}
    detector_pam_by_snr, detector_pam_by_class = {}, {}
    for key in sorted({row["snr_bin"] for row in main_joint["oracle_rows"]}):
        oracle_by_snr[key] = _accuracy_summary(
            main_joint["oracle_rows"], snr_bin=key, iterations=args.bootstrap_iterations
        )
    for key in sorted({row["snr_bin"] for row in main_joint["detector_pam_rows"]}):
        detector_pam_by_snr[key] = _accuracy_summary(
            main_joint["detector_pam_rows"], snr_bin=key, iterations=args.bootstrap_iterations
        )
    for name in PAM_SUBTYPES:
        oracle_by_class[name] = _accuracy_summary(
            main_joint["oracle_rows"], label_filter=name, iterations=args.bootstrap_iterations
        )
        detector_pam_by_class[name] = _accuracy_summary(
            main_joint["detector_pam_rows"], label_filter=name, iterations=args.bootstrap_iterations
        )
    _save_pam_snr_plot(
        oracle_by_snr, detector_pam_by_snr, out_dir / "plots" / "pam_accuracy_vs_snr.png"
    )

    oracle_accuracy = _accuracy_summary(
        main_joint["oracle_rows"], iterations=args.bootstrap_iterations
    )
    detector_box_accuracy = _accuracy_summary(
        main_joint["detector_pam_rows"], iterations=args.bootstrap_iterations
    )
    test_frame_x, test_frame_y, test_frame_rows = _build_frame_set(test_rows, data_dir, jitter=False)
    test_frame_predictions = _predict_frame_labels(pam_model, test_frame_x, device=device)
    all_classifier_signal_rows = _signal_level_predictions(test_frame_predictions, test_frame_rows)
    all_classifier_accuracy = _accuracy_summary(
        all_classifier_signal_rows, iterations=args.bootstrap_iterations
    )
    all_classifier_by_class = {
        label: _accuracy_summary(
            all_classifier_signal_rows, label_filter=label, iterations=args.bootstrap_iterations
        )
        for label in PAM_LABELS
    }
    all_classifier_confusion = _confusion(
        [row["label"] for row in all_classifier_signal_rows],
        [row["predicted"] for row in all_classifier_signal_rows],
        list(PAM_LABELS),
    )
    _save_confusion(
        all_classifier_confusion, list(PAM_LABELS), out_dir / "plots" / "pam_all_classes_confusion.png",
        "PAM CNN frame-classifier confusion (PAM subtypes and Others)",
    )
    shifted_oracle_accuracy = _accuracy_summary(
        shifted_joint["oracle_rows"], iterations=args.bootstrap_iterations
    )
    shifted_oracle_by_snr = {
        key: _accuracy_summary(
            shifted_joint["oracle_rows"], snr_bin=key, iterations=args.bootstrap_iterations
        )
        for key in sorted({row["snr_bin"] for row in shifted_joint["oracle_rows"]})
    }
    shifted_oracle_by_class = {
        name: _accuracy_summary(
            shifted_joint["oracle_rows"], label_filter=name, iterations=args.bootstrap_iterations
        )
        for name in PAM_SUBTYPES
    }
    shifted_detector_box_accuracy = _accuracy_summary(
        shifted_joint["detector_pam_rows"], iterations=args.bootstrap_iterations
    )
    shifted_detector_pam_by_snr = {
        key: _accuracy_summary(
            shifted_joint["detector_pam_rows"], snr_bin=key, iterations=args.bootstrap_iterations
        )
        for key in sorted({row["snr_bin"] for row in shifted_joint["detector_pam_rows"]}, key=_snr_sort_key)
    }
    shifted_detector_pam_by_class = {
        name: _accuracy_summary(
            shifted_joint["detector_pam_rows"], label_filter=name,
            iterations=args.bootstrap_iterations,
        )
        for name in PAM_SUBTYPES
    }
    _save_pam_snr_plot(
        shifted_oracle_by_snr, shifted_detector_pam_by_snr,
        out_dir / "plots" / "pam_accuracy_vs_snr_shifted.png",
    )
    high_snr_rows = [row for row in main_joint["oracle_rows"] if row["global_snr_db"] >= 10.0]
    zero_snr_rows = [row for row in main_joint["oracle_rows"] if 0.0 <= row["global_snr_db"] < 1.0]
    high_snr_accuracy = _accuracy_summary(high_snr_rows, iterations=args.bootstrap_iterations)
    zero_snr_accuracy = _accuracy_summary(zero_snr_rows, iterations=args.bootstrap_iterations)

    high_snr_detector = detection_stats(
        high_snr_items, confidence_threshold=float(yolo_calibration["threshold"]), class_agnostic=True
    )
    end_to_end_by_snr = _end_to_end_by_snr(main_joint["e2e_rows"], args.bootstrap_iterations)
    all_inband = [
        float(signal["approximate_inband_snr_db"])
        for metadata in test_meta.values()
        for signal in metadata["signals"] if signal["detector_class"] == "PAM"
    ]

    confusion_names = list(DETECTOR_CLASSES) + ["missed", "background"]
    detector_confusion = _confusion(
        main_joint["confusion_true"], main_joint["confusion_pred"], confusion_names
    )
    _save_confusion(
        detector_confusion, confusion_names, out_dir / "plots" / "detector_confusion.png",
        "YOLOv5n detector class confusion (matched boxes)",
    )
    shifted_confusion = _confusion(
        shifted_joint["confusion_true"], shifted_joint["confusion_pred"], confusion_names
    )
    _save_confusion(
        shifted_confusion, confusion_names, out_dir / "plots" / "detector_confusion_shifted.png",
        "YOLOv5n detector class confusion (shifted test)",
    )
    pam_names = list(PAM_SUBTYPES) + ["Others"]
    pam_confusion = _confusion(
        [row["label"] for row in main_joint["oracle_rows"]],
        [row["predicted"] for row in main_joint["oracle_rows"]], pam_names,
    )
    _save_confusion(
        pam_confusion, pam_names, out_dir / "plots" / "pam_subtype_confusion.png",
        "PAM subtype confusion (oracle boxes)",
    )
    shifted_pam_confusion = _confusion(
        [row["label"] for row in shifted_joint["oracle_rows"]],
        [row["predicted"] for row in shifted_joint["oracle_rows"]], pam_names,
    )
    _save_confusion(
        shifted_pam_confusion, pam_names,
        out_dir / "plots" / "pam_subtype_confusion_shifted.png",
        "PAM subtype confusion (shifted test, oracle boxes)",
    )

    acceptance = {
        "recall_at_global_snr_ge_5_db": {
            "target": 0.80, "measured": high_snr_detector["recall"],
            "ci95": _bootstrap_detection(
                high_snr_items, "recall", float(yolo_calibration["threshold"]), args.bootstrap_iterations
            )["ci95"],
            "pass": high_snr_detector["recall"] >= 0.80,
            "support": {
                "recordings": len({i["recording_id"] for i in high_snr_items}),
                "signals": sum(len(i["gt"]) for i in high_snr_items),
                "low_support": sum(len(i["gt"]) for i in high_snr_items) < 30,
            },
        },
        "mean_iou_at_global_snr_ge_5_db": {
            "target": 0.75, "measured": high_snr_detector["mean_iou_true_positives"],
            "ci95": _bootstrap_detection(
                high_snr_items, "mean_iou_true_positives",
                float(yolo_calibration["threshold"]), args.bootstrap_iterations
            )["ci95"],
            "pass": high_snr_detector["mean_iou_true_positives"] >= 0.75,
            "support": {
                "recordings": len({i["recording_id"] for i in high_snr_items}),
                "signals": sum(len(i["gt"]) for i in high_snr_items),
                "low_support": sum(len(i["gt"]) for i in high_snr_items) < 30,
            },
        },
        "map50_at_global_snr_ge_5_db": {
            "target": 0.60, "measured": high_snr_map["map50"],
            "ci95": high_snr_map["map50_recording_bootstrap"]["ci95"],
            "pass": high_snr_map["map50"] >= 0.60,
            "support": {
                "recordings": high_snr_map["map50_recording_bootstrap"]["recordings"],
                "signals": high_snr_map["map50_recording_bootstrap"]["signals"],
                "low_support": high_snr_map["map50_recording_bootstrap"]["low_support"],
            },
        },
        "pam_oracle_accuracy_global_snr_ge_10_db": {
            "target": 0.80, "measured": high_snr_accuracy["accuracy"],
            "ci95": high_snr_accuracy["ci95"],
            "pass": high_snr_accuracy["accuracy"] >= 0.80,
            "support": {key: high_snr_accuracy[key] for key in ("recordings", "signals", "low_support")},
        },
        "pam_oracle_accuracy_near_0_db_0_to_1": {
            "target": 0.50, "measured": zero_snr_accuracy["accuracy"],
            "ci95": zero_snr_accuracy["ci95"],
            "pass": zero_snr_accuracy["accuracy"] > 0.50,
            "support": {key: zero_snr_accuracy[key] for key in ("recordings", "signals", "low_support")},
        },
        "processing_time_per_segment_local_seconds": {
            "target": 5.0, "measured": None, "pass": None,
            "note": "Not measured by this Colab T4 run; measure on the user's laptop separately.",
        },
    }
    metrics = {
        "schema_version": 1,
        "config": config,
        "seed": args.seed,
        "yolo": yolo_info,
        "snr_definition": "10*log10(total clean signal power / total added noise power over the full I/Q segment)",
        "runtime_seconds_colab": {
            "data_generation": generation_seconds,
            "note": "Colab runtime; not a local-machine acceptance measurement.",
        },
        "split_counts_recordings": dict(split_counts),
        "validation_calibration": {
            "target_precision": 0.80,
            "yolo": yolo_calibration,
            "energy_mask_threshold": energy_mask_threshold,
            "energy_confidence_threshold": energy_calibration["threshold"],
            "energy_baseline": energy_calibration,
        },
        "acceptance_targets": acceptance,
        "main_test": {
            "yolo": {
                "matched_precision": main_yolo_stats,
                "matched_precision_with_recording_bootstrap": main_yolo_ci,
                "map50_all_snr": main_map,
                "map50_global_snr_ge_5_db": high_snr_map,
                "per_class_with_recording_bootstrap": per_class,
                "by_global_snr_with_recording_bootstrap": detector_by_snr,
            },
            "energy_baseline": {
                "matched_precision": main_energy_stats,
                "matched_precision_with_recording_bootstrap": main_energy_ci,
                "by_global_snr_with_recording_bootstrap": baseline_by_snr,
                "mask_threshold_selected_on_validation": energy_mask_threshold,
                "confidence_threshold_selected_on_validation": energy_calibration["threshold"],
            },
            "pam": {
                "oracle_accuracy": oracle_accuracy,
                "oracle_accuracy_by_global_snr": oracle_by_snr,
                "oracle_accuracy_by_subtype": oracle_by_class,
                "detector_box_accuracy": detector_box_accuracy,
                "detector_box_accuracy_by_global_snr": detector_pam_by_snr,
                "detector_box_accuracy_by_subtype": detector_pam_by_class,
                "oracle_subtype_confusion_matrix": pam_confusion.tolist(),
                "oracle_subtype_confusion_labels": pam_names,
                "unresolved_detection_count": len(main_joint["unresolved_detections"]),
                "unresolved_detections": main_joint["unresolved_detections"],
                "non_pam_detections_classified_as_others": {
                    "overall": _binary_rate_summary(
                        main_joint["false_class_rows"], iterations=args.bootstrap_iterations
                    ),
                    "by_detector_class": {
                        name: _binary_rate_summary(
                            main_joint["false_class_rows"], class_filter=name,
                            iterations=args.bootstrap_iterations,
                        )
                        for name in NON_PAM
                    },
                    "by_global_snr": {
                        key: _binary_rate_summary(
                            main_joint["false_class_rows"], snr_bin=key,
                            iterations=args.bootstrap_iterations,
                        )
                        for key in sorted({row["snr_bin"] for row in main_joint["false_class_rows"]})
                    },
                    "counts": {
                        "others": main_joint["false_class_others_rate"]["others"],
                        "total": main_joint["false_class_others_rate"]["total"],
                        "per_non_pam_detector_class": {
                            name: {key: value[key] for key in ("others", "total")}
                            for name, value in main_joint["false_class_others_rate"]["per_non_pam_detector_class"].items()
                        },
                    },
                },
                "all_signal_classifier_accuracy": all_classifier_accuracy,
                "all_signal_classifier_accuracy_by_class": all_classifier_by_class,
                "all_classes_confusion_matrix": all_classifier_confusion.tolist(),
                "all_classes_confusion_labels": list(PAM_LABELS),
                "approximate_pam_inband_snr_db": {
                    "count": len(all_inband),
                    "mean": float(np.mean(all_inband)) if all_inband else None,
                    "min": float(np.min(all_inband)) if all_inband else None,
                    "max": float(np.max(all_inband)) if all_inband else None,
                },
            },
            "end_to_end": {
                **main_joint["end_to_end"],
                "by_global_snr_with_recording_bootstrap": end_to_end_by_snr,
            },
        },
        "shifted_test": {
            "distribution": "bandwidth uniform [0.02, 0.12]; global SNR uniform [-5, 5] dB",
            "yolo_matched_precision": shifted_yolo_stats,
            "yolo_matched_precision_with_recording_bootstrap": shifted_yolo_ci,
            "yolo_map50": shifted_map,
            "detector_confusion_matrix": shifted_confusion.tolist(),
            "detector_confusion_labels": confusion_names,
            "pam_subtype_confusion_matrix": shifted_pam_confusion.tolist(),
            "pam_subtype_confusion_labels": pam_names,
            "unresolved_detection_count": len(shifted_joint["unresolved_detections"]),
            "unresolved_detections": shifted_joint["unresolved_detections"],
            "yolo_per_class_with_recording_bootstrap": shifted_per_class,
            "yolo_by_global_snr_with_recording_bootstrap": shifted_by_snr,
            "energy_baseline_matched_precision": shifted_energy_stats,
            "energy_baseline_matched_precision_with_recording_bootstrap": shifted_energy_ci,
            "energy_baseline_by_global_snr_with_recording_bootstrap": shifted_baseline_by_snr,
            "pam_oracle_accuracy": shifted_oracle_accuracy,
            "pam_oracle_accuracy_by_global_snr": shifted_oracle_by_snr,
            "pam_oracle_accuracy_by_subtype": shifted_oracle_by_class,
            "pam_detector_box_accuracy": shifted_detector_box_accuracy,
            "pam_detector_box_accuracy_by_global_snr": shifted_detector_pam_by_snr,
            "pam_detector_box_accuracy_by_subtype": shifted_detector_pam_by_class,
            "end_to_end": shifted_joint["end_to_end"],
            "end_to_end_by_global_snr_with_recording_bootstrap": _end_to_end_by_snr(
                shifted_joint["e2e_rows"], args.bootstrap_iterations
            ),
            "non_pam_detections_classified_as_others": {
                "overall": _binary_rate_summary(
                    shifted_joint["false_class_rows"], iterations=args.bootstrap_iterations
                ),
                "by_detector_class": {
                    name: _binary_rate_summary(
                        shifted_joint["false_class_rows"], class_filter=name,
                        iterations=args.bootstrap_iterations,
                    ) for name in NON_PAM
                },
                "by_global_snr": {
                    key: _binary_rate_summary(
                        shifted_joint["false_class_rows"], snr_bin=key,
                        iterations=args.bootstrap_iterations,
                    ) for key in sorted({row["snr_bin"] for row in shifted_joint["false_class_rows"]})
                },
            },
        },
        "support": {
            "main_test_recordings": len({item["recording_id"] for item in test_items}),
            "main_test_segments": len(test_items),
            "shifted_test_recordings": len({item["recording_id"] for item in shifted_items}),
            "shifted_test_segments": len(shifted_items),
            "low_support_rule": "Any per-class or per-SNR cell with fewer than 30 signals is marked low_support.",
            "pam_zero_db_bin": "[0, 1) dB because the generator draws a continuous SNR; inspect support before interpreting.",
        },
    }
    metrics_path = out_dir / "results" / "metrics.json"
    metrics["metrics_path"] = str(metrics_path)
    _save_json(metrics_path, metrics)

    _write_summary(metrics, out_dir / "results" / "summary.txt")
    print("\n" + (out_dir / "results" / "summary.txt").read_text(encoding="utf-8"), flush=True)
    print(f"All artifacts saved under {out_dir}", flush=True)


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    default_config = project_root / "configs" / "milestone2.yaml"
    defaults = _load_config(default_config)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-dir", required=True)
    parser.add_argument("--out-dir", required=True, help="Google Drive result folder")
    parser.add_argument("--data-work", required=True, help="Colab local-SSD temporary data folder")
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--recordings", type=int, default=defaults["data"]["main_recordings"])
    parser.add_argument("--shift-recordings", type=int, default=defaults["data"]["shifted_recordings"])
    parser.add_argument("--segments-per-recording", type=int, default=defaults["data"]["segments_per_recording"])
    parser.add_argument("--yolo-epochs", type=int, default=defaults["detector"]["epochs"])
    parser.add_argument("--yolo-batch", type=int, default=defaults["detector"]["batch_size"])
    parser.add_argument("--pam-epochs", type=int, default=defaults["pam_classifier"]["epochs"])
    parser.add_argument("--pam-batch", type=int, default=defaults["pam_classifier"]["batch_size"])
    parser.add_argument("--bootstrap-iterations", type=int, default=defaults["evaluation"]["bootstrap_iterations"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
