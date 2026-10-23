"""Detection, subtype, and recording-bootstrap metrics without external eval packages."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
import numpy as np


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_image(ground_truth, predictions, *, iou_threshold=0.5, class_agnostic=False):
    """Greedy confidence-ordered one-to-one matching within one image."""
    used_gt = set()
    matches = []
    for pred_idx in sorted(range(len(predictions)), key=lambda i: predictions[i]["confidence"], reverse=True):
        pred = predictions[pred_idx]
        choices = []
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in used_gt:
                continue
            if not class_agnostic and gt["class"] != pred["class"]:
                continue
            overlap = iou_xyxy(gt["xyxy"], pred["xyxy"])
            if overlap >= iou_threshold:
                choices.append((overlap, gt_idx))
        if choices:
            overlap, gt_idx = max(choices)
            used_gt.add(gt_idx)
            matches.append((gt_idx, pred_idx, overlap))
    used_pred = {pred_idx for _, pred_idx, _ in matches}
    return matches, [i for i in range(len(ground_truth)) if i not in used_gt], [
        i for i in range(len(predictions)) if i not in used_pred
    ]


def detection_stats(items, *, confidence_threshold, iou_threshold=0.5, class_agnostic=False):
    tp = fp = fn = 0
    overlaps = []
    for item in items:
        preds = [p for p in item["pred"] if p["confidence"] >= confidence_threshold]
        matches, missed_gt, extra_pred = match_image(
            item["gt"], preds, iou_threshold=iou_threshold, class_agnostic=class_agnostic
        )
        tp += len(matches)
        fp += len(extra_pred)
        fn += len(missed_gt)
        overlaps.extend(match[2] for match in matches)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "mean_iou_true_positives": float(np.mean(overlaps)) if overlaps else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def calibrate_confidence(items, *, target_precision=0.8, class_agnostic=True, iou_threshold=0.5):
    """Choose a validation confidence threshold and maximize recall at target precision."""
    scores = np.asarray(
        [float(p["confidence"]) for item in items for p in item["pred"]],
        dtype=np.float64,
    )
    if scores.size:
        # Quantile thresholds keep validation calibration bounded even for dense outputs.
        candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))[::-1]
        candidates = [1.01, *candidates.tolist(), 0.0]
    else:
        candidates = [1.01, 0.0]
    evaluated = []
    for threshold in candidates:
        stats = detection_stats(
            items,
            confidence_threshold=threshold,
            iou_threshold=iou_threshold,
            class_agnostic=class_agnostic,
        )
        evaluated.append((threshold, stats))
    feasible = [entry for entry in evaluated if entry[1]["precision"] >= target_precision]
    if feasible:
        closest_precision = min(abs(row[1]["precision"] - target_precision) for row in feasible)
        closest = [row for row in feasible if abs(row[1]["precision"] - target_precision) <= closest_precision + 1e-12]
        threshold, stats = max(closest, key=lambda row: row[1]["recall"])
        return {"threshold": threshold, "attained_target": True, **stats}
    threshold, stats = min(
        evaluated,
        key=lambda row: (abs(row[1]["precision"] - target_precision), -row[1]["recall"]),
    )
    return {"threshold": threshold, "attained_target": False, **stats}


def average_precision(items, class_name, *, iou_threshold=0.5):
    """VOC-style all-points AP for one class at a fixed IoU threshold."""
    gt_by_image = []
    total_gt = 0
    predictions = []
    for image_index, item in enumerate(items):
        gt = [g for g in item["gt"] if g["class"] == class_name]
        gt_by_image.append(gt)
        total_gt += len(gt)
        predictions.extend(
            (float(p["confidence"]), image_index, p)
            for p in item["pred"]
            if p["class"] == class_name
        )
    if total_gt == 0:
        return 0.0, 0, len(predictions)
    used = [set() for _ in items]
    tp = np.zeros(len(predictions), dtype=np.float64)
    fp = np.zeros(len(predictions), dtype=np.float64)
    predictions.sort(key=lambda row: row[0], reverse=True)
    for index, (_, image_index, pred) in enumerate(predictions):
        gt = gt_by_image[image_index]
        if not gt:
            fp[index] = 1
            continue
        overlaps = [iou_xyxy(candidate["xyxy"], pred["xyxy"]) for candidate in gt]
        best = int(np.argmax(overlaps))
        if overlaps[best] >= iou_threshold and best not in used[image_index]:
            tp[index] = 1
            used[image_index].add(best)
        else:
            fp[index] = 1
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    mrec = np.r_[0.0, recall, 1.0]
    mpre = np.r_[0.0, precision, 0.0]
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))
    return ap, total_gt, len(predictions)


def class_map50(items, class_names):
    per_class = {}
    aps = []
    for name in class_names:
        ap, n_gt, n_pred = average_precision(items, name)
        per_class[name] = {"ap50": ap, "signals": n_gt, "predictions": n_pred}
        if n_gt:
            aps.append(ap)
    return {"map50": float(np.mean(aps)) if aps else 0.0, "per_class": per_class}


def recording_bootstrap(groups, statistic, *, iterations=1000, seed=2026):
    """Percentile CI by resampling parent-recording groups, never individual signals."""
    if not groups:
        return 0.0, 0.0, 0.0
    estimate = float(statistic(list(groups)))
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        indices = rng.integers(0, len(groups), size=len(groups))
        samples[i] = statistic([groups[int(j)] for j in indices])
    low, high = np.quantile(samples, [0.025, 0.975])
    return estimate, float(low), float(high)


def summarize_by_recording(items, metric, *, confidence_threshold, iou_threshold=0.5, class_agnostic=False):
    """Return overall estimate/CI and recording/signal support for a detection metric."""
    grouped = defaultdict(list)
    for item in items:
        grouped[item["recording_id"]].append(item)
    groups = [{"recording_id": record_id, "items": rows} for record_id, rows in grouped.items()]

    def statistic(sampled_groups):
        sampled = [item for group in sampled_groups for item in group["items"]]
        return float(
            detection_stats(
                sampled,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                class_agnostic=class_agnostic,
            )[metric]
        )

    estimate, low, high = recording_bootstrap(groups, statistic)
    signals = sum(len(item["gt"]) for item in items)
    return {
        "estimate": estimate,
        "ci95": [low, high],
        "recordings": len(groups),
        "signals": signals,
        "low_support": signals < 30,
    }
