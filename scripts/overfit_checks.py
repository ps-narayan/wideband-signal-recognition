"""Train-only memorization checks for the detector and PAM CNN."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml

from wideband_recognition.dataset import write_segment_artifacts
from wideband_recognition.geometry import to_yolo
from wideband_recognition.metrics import class_map50, detection_stats
from wideband_recognition.pam_model import PAMCNN, PAM_LABELS
from wideband_recognition.signals import DETECTOR_CLASSES
from wideband_recognition.yolo_schedule import expected_optimizer_schedule, temporary_yolov5_schedule
from run_milestone2 import _main_manifest, _build_frame_set, _save_json

IMAGE_SIZE = 640


def _run(command: list[str], *, cwd: Path, log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - start
    if result.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{tail}")
    return elapsed


def _draw_gt_overlays(data_dir: Path, rows: list[dict], output: Path, count: int = 4) -> list[dict]:
    directory = output / "ground_truth_boxes"
    directory.mkdir(parents=True, exist_ok=True)
    index = []
    colors = [(255, 80, 40), (60, 220, 80), (40, 160, 255), (255, 210, 40), (210, 70, 255), (40, 230, 230)]
    for row in rows[:count]:
        image_path = data_dir / "images" / "train" / f"{row['segment_id']}.png"
        label_path = data_dir / "labels" / "train" / f"{row['segment_id']}.txt"
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        objects = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_id, cx, cy, width, height = map(float, line.split())
            name = DETECTOR_CLASSES[int(class_id)]
            box = [
                (cx - width / 2) * image.width,
                (cy - height / 2) * image.height,
                (cx + width / 2) * image.width,
                (cy + height / 2) * image.height,
            ]
            color = colors[int(class_id) % len(colors)]
            draw.rectangle(box, outline=color, width=2)
            draw.text((max(0, box[0]), max(0, box[1] - 11)), name, fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
            objects.append({"class_id": int(class_id), "class": name, "xyxy_pixels": [round(v, 2) for v in box]})
        target = directory / f"{row['segment_id']}.png"
        image.save(target)
        index.append({"recording_id": row["recording_id"], "segment_id": row["segment_id"], "file": target.name, "boxes": objects})
    (directory / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def _parse_label(path: Path, *, predictions: bool, image_size: int = IMAGE_SIZE) -> list[dict]:
    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        values = list(map(float, line.split()))
        if len(values) not in ((6,) if predictions else (5,)):
            raise ValueError(f"Unexpected YOLO label row in {path}: {line}")
        class_id, cx, cy, width, height = values[:5]
        if int(class_id) not in range(len(DETECTOR_CLASSES)):
            raise ValueError(f"Unexpected class ID {class_id} in {path}")
        boxes.append({
            "class": DETECTOR_CLASSES[int(class_id)],
            "xyxy": [
                (cx - width / 2) * image_size, (cy - height / 2) * image_size,
                (cx + width / 2) * image_size, (cy + height / 2) * image_size,
            ],
            **({"confidence": values[5]} if predictions else {}),
        })
    return boxes


def _check_label_coordinates(data_dir: Path, rows: list[dict], image_size: int = IMAGE_SIZE) -> dict:
    max_abs_norm_error = 0.0
    box_count = 0
    class_counts = Counter()
    for row in rows:
        metadata = json.loads((data_dir / "metadata" / "train" / f"{row['segment_id']}.json").read_text(encoding="utf-8"))
        labels = (data_dir / "labels" / "train" / f"{row['segment_id']}.txt").read_text(encoding="utf-8").splitlines()
        if len(labels) != len(metadata["signals"]):
            raise AssertionError(f"Label/signal count mismatch for {row['segment_id']}")
        for signal, line in zip(metadata["signals"], labels):
            values = list(map(float, line.split()))
            class_id = DETECTOR_CLASSES.index(signal["detector_class"])
            expected = to_yolo(
                __import__("wideband_recognition.dataset", fromlist=["metadata_box"]).metadata_box(signal),
                class_id, int(metadata["covered_samples"]), image_size, image_size,
            )
            if int(values[0]) != class_id:
                raise AssertionError(f"Class mapping mismatch for {signal['detector_class']}")
            max_abs_norm_error = max(max_abs_norm_error, max(abs(values[i + 1] - expected[i + 1]) for i in range(4)))
            class_counts[signal["detector_class"]] += 1
            box_count += 1
    return {
        "label_rows": box_count,
        "class_counts": dict(class_counts),
        "max_abs_normalized_coordinate_error": max_abs_norm_error,
        "class_mapping_matches_generator": True,
        "coordinates_match_generator_parameters": max_abs_norm_error <= 2e-7,
        "image_orientation": "image columns are time and rows are shifted frequency, matching YOLO x/y box convention",
    }


def _training_box_dimensions(data_dir: Path, rows: list[dict], image_size: int) -> dict:
    widths, heights = [], []
    for row in rows:
        label_path = data_dir / "labels" / "train" / f"{row['segment_id']}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            _, _, _, width, height = map(float, line.split())
            widths.append(width * image_size)
            heights.append(height * image_size)
    return {
        "box_count": len(widths),
        "width_pixels_min_median_max": [float(np.min(widths)), float(np.median(widths)), float(np.max(widths))] if widths else [],
        "height_pixels_min_median_max": [float(np.min(heights)), float(np.median(heights)), float(np.max(heights))] if heights else [],
        "boxes_with_either_dimension_below_stride8": sum(w < 8 or h < 8 for w, h in zip(widths, heights)),
        "yolov5_detection_strides": [8, 16, 32],
    }


def _pam_overfit(data_dir: Path, rows: list[dict], *, seed: int, max_epochs: int, frames_per_class: int, output: Path) -> dict:
    frames, labels, frame_rows = _build_frame_set(rows, data_dir, jitter=True)
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(len(PAM_LABELS)):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) > frames_per_class:
            indices = np.sort(rng.choice(indices, size=frames_per_class, replace=False))
        selected.extend(indices.tolist())
    selected = np.asarray(sorted(selected), dtype=np.int64)
    frames, labels = frames[selected], labels[selected]
    frame_rows = [frame_rows[int(i)] for i in selected]
    if len(frames) < 2:
        raise RuntimeError("Need at least two generated I/Q frames for the PAM memorization check")

    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = PAMCNN().cpu()
    x = torch.from_numpy(frames.transpose(0, 2, 1)).float()
    y = torch.from_numpy(labels).long()
    counts = np.bincount(labels, minlength=len(PAM_LABELS)).astype(np.float32)
    weights = np.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    perfect_streak = 0
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            train_logits = model(x)
            train_loss = float(criterion(train_logits, y))
            train_accuracy = float((train_logits.argmax(1) == y).float().mean())
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_frame_accuracy": train_accuracy})
        perfect_streak = perfect_streak + 1 if train_accuracy == 1.0 and train_loss < 0.05 else 0
        if perfect_streak >= 5:
            break
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        predictions = model(x).argmax(1).cpu().numpy()
    per_class = {}
    for class_id, name in enumerate(PAM_LABELS):
        mask = labels == class_id
        if mask.any():
            per_class[name] = float(np.mean(predictions[mask] == labels[mask]))
    return {
        "frames": len(frames), "classes_present": [PAM_LABELS[i] for i in sorted(set(labels.tolist()))],
        "frames_per_class": {PAM_LABELS[i]: int(counts[i]) for i in range(len(PAM_LABELS)) if counts[i]},
        "frame_accuracy_same_training_frames": float(np.mean(predictions == labels)),
        "per_class_frame_accuracy": per_class,
        "epochs_run": len(history), "max_epochs": max_epochs,
        "stopped_after_five_perfect_epochs": perfect_streak >= 5,
        "wall_seconds": elapsed, "history": history,
        "scope": "training-only memorization check; no validation/test data used",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("work/overfit"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/overfit"))
    parser.add_argument("--recordings", type=int, default=18)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--yolo-epochs", type=int, default=100)
    parser.add_argument("--pam-epochs", type=int, default=250)
    parser.add_argument("--pam-frames-per-class", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--max-det", type=int, default=3000, help="Maximum detections per image for the same-image diagnostic; high cap avoids truncating low-confidence candidates.")
    parser.add_argument("--save-period", type=int, default=20, help="YOLO checkpoint save period in epochs; use 0 to keep only best/last checkpoints.")
    parser.add_argument("--model-size", choices=("n", "s", "m", "l"), default="n")
    parser.add_argument("--min-bandwidth", type=float, default=0.025)
    parser.add_argument("--min-duration", type=int, default=100_000)
    parser.add_argument("--generator-config", type=Path, help="YAML file containing optional min_bandwidth_cycles_per_sample and min_duration_samples constraints.")
    parser.add_argument("--optimizer", choices=("SGD", "Adam", "AdamW"), default="SGD")
    parser.add_argument("--nominal-batch-size", type=int, default=4,
                        help="Keep YOLOv5 nominal batch size equal to the four-image diagnostic micro-batch.")
    parser.add_argument("--warmup-min-iterations", type=int, default=2,
                        help="Short YOLOv5 warmup floor so the tiny-set fit receives optimizer updates after warmup.")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--skip-pam", action="store_true", help="Skip the independent PAM check for detector-only experiment variants.")
    parser.add_argument("--reuse-yolo-checkpoint", action="store_true", help="Evaluate an existing 8-image training checkpoint without retraining it.")
    args = parser.parse_args()
    generator_config = {}
    if args.generator_config:
        generator_config = yaml.safe_load(args.generator_config.read_text(encoding="utf-8")) or {}
        args.min_bandwidth = float(generator_config.get("min_bandwidth_cycles_per_sample", args.min_bandwidth))
        args.min_duration = int(generator_config.get("min_duration_samples", args.min_duration))
    if args.image_size <= 0 or args.image_size % 32:
        raise ValueError("image_size must be a positive multiple of 32")
    if args.nominal_batch_size < 1 or args.warmup_min_iterations < 1 or args.save_period < 0:
        raise ValueError("nominal_batch_size and warmup_min_iterations must be positive; save_period must be non-negative")
    yolo_dir, work_dir, results_dir = (p.resolve() for p in (args.yolo_dir, args.work_dir, args.results_dir))
    if work_dir.exists() and not args.reuse_yolo_checkpoint:
        shutil.rmtree(work_dir)
    if args.reuse_yolo_checkpoint and not work_dir.exists():
        raise FileNotFoundError(f"Cannot reuse checkpoint; work directory is missing: {work_dir}")
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    # Pinned YOLOv5 v7 writes its settings under %APPDATA% on Windows.
    # Keep that cache inside the diagnostic workspace for sandboxed runs.
    appdata = work_dir / "appdata"
    config_dir = work_dir / "ultralytics-config"
    appdata.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["APPDATA"] = str(appdata)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLOV5_CONFIG_DIR"] = str(config_dir)
    os.environ["MPLCONFIGDIR"] = str(work_dir / "mpl-config")
    yolo_dir = yolo_dir.resolve()
    from run_milestone2 import _check_yolo
    yolo_info = _check_yolo(yolo_dir)

    manifest, _ = _main_manifest(args.recordings, 1, args.seed)
    train_rows = sorted((row for row in manifest if row["split"] == "train"), key=lambda row: row["segment_id"])
    if args.images > len(train_rows):
        raise ValueError(f"Requested {args.images} training images but split has only {len(train_rows)}")
    train_rows = train_rows[:args.images]
    for row in train_rows:
        print(f"Generating training image {row['segment_id']} at {args.image_size}", flush=True)
        write_segment_artifacts(
            data_dir, row["segment_id"], row["recording_id"], "train", row["seed"], "train",
            n_samples=1_000_000, image_size=args.image_size,
            min_bandwidth=args.min_bandwidth, min_duration=args.min_duration,
            pam_subtype_index=row.get("pam_subtype_index"),
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    overlay_index = _draw_gt_overlays(data_dir, train_rows, results_dir)
    label_check = _check_label_coordinates(data_dir, train_rows, args.image_size)
    box_dimensions = _training_box_dimensions(data_dir, train_rows, args.image_size)

    names = ", ".join(f"'{name}'" for name in DETECTOR_CLASSES)
    data_yaml = work_dir / "overfit.yaml"
    data_yaml.write_text(
        f"path: {data_dir.as_posix()}\ntrain: images/train\nval: images/train\nnc: {len(DETECTOR_CLASSES)}\nnames: [{names}]\n",
        encoding="utf-8",
    )
    # YOLOv5 downloads its optional label font if missing. Reuse a local system
    # font when available so offline CPU runs can start without changing training.
    font_path = config_dir / "Arial.ttf"
    if not font_path.exists():
        font_candidates = [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
        local_font = next((candidate for candidate in font_candidates if candidate.is_file()), None)
        if local_font is not None:
            shutil.copyfile(local_font, font_path)

    hyp = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "yolov5_hyp.yaml").read_text(encoding="utf-8"))
    for key, value in {"mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "degrees": 0.0, "translate": 0.0, "scale": 0.0, "fliplr": 0.0, "flipud": 0.0}.items():
        hyp[key] = value
    hyp["lr0"] = args.lr0
    hyp["warmup_epochs"] = args.warmup_epochs
    hyp_path = work_dir / "hyp_no_augmentation.yaml"
    hyp_path.write_text(yaml.safe_dump(hyp, sort_keys=False), encoding="utf-8")
    train_log = work_dir / "yolo_train.log"
    yolo_project = work_dir / "checkpoints" / "yolo"
    train_command = [
        sys.executable, str(yolo_dir / "train.py"), "--img", str(args.image_size),
        "--batch-size", "4", "--epochs", str(args.yolo_epochs), "--data", str(data_yaml),
        "--weights", "", "--cfg", str(yolo_dir / "models" / f"yolov5{args.model_size}.yaml"),
        "--hyp", str(hyp_path), "--optimizer", args.optimizer, "--device", "cpu", "--workers", "0",
        "--project", str(yolo_project), "--name", "memorize8", "--exist-ok", "--noval",
        "--seed", str(args.seed), "--save-period", str(-1 if args.save_period == 0 else args.save_period),
    ]
    checkpoint = yolo_project / "memorize8" / "weights" / "best.pt"
    if args.reuse_yolo_checkpoint:
        if not checkpoint.is_file():
            checkpoint = yolo_project / "memorize8" / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"No existing YOLO checkpoint to reuse under {yolo_project}")
        yolo_seconds = None
    else:
        print(f"Starting YOLOv5{args.model_size} training: {args.yolo_epochs} epochs, {args.image_size}px", flush=True)
        train_script = yolo_dir / "train.py"
        with temporary_yolov5_schedule(
            train_script,
            nominal_batch_size=args.nominal_batch_size,
            warmup_min_iterations=args.warmup_min_iterations,
        ):
            yolo_seconds = _run(train_command, cwd=yolo_dir, log_path=train_log)
    if not checkpoint.is_file():
        checkpoint = yolo_project / "memorize8" / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"YOLO training produced no checkpoint: {checkpoint}")

    sys.path.insert(0, str(yolo_dir))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_obj = ckpt.get("ema") or ckpt.get("model")
    names_obj = getattr(model_obj, "names", None)
    saved_names = [names_obj[i] for i in range(len(names_obj))] if isinstance(names_obj, dict) else list(names_obj or [])
    mapping_matches = saved_names == list(DETECTOR_CLASSES)
    detect_dir = work_dir / "predictions"
    # YOLOv5 appends save-txt output when --exist-ok reuses a run directory.
    # Delete the prior inference outputs so threshold/cap reruns stay independent.
    if detect_dir.exists():
        shutil.rmtree(detect_dir)
    detect_log = work_dir / "yolo_detect.log"
    detect_command = [
        sys.executable, str(yolo_dir / "detect.py"), "--weights", str(checkpoint),
        "--source", str(data_dir / "images" / "train"), "--img", str(args.image_size),
        "--conf-thres", "0.001", "--iou-thres", "0.6", "--max-det", str(args.max_det),
        "--device", "cpu", "--save-txt", "--save-conf", "--project", str(detect_dir.parent),
        "--name", detect_dir.name, "--exist-ok", "--nosave",
    ]
    detect_seconds = _run(detect_command, cwd=yolo_dir, log_path=detect_log)
    prediction_labels = detect_dir / "labels"
    items = []
    for row in train_rows:
        segment_id = row["segment_id"]
        items.append({
            "recording_id": row["recording_id"],
            "gt": _parse_label(data_dir / "labels" / "train" / f"{segment_id}.txt", predictions=False, image_size=args.image_size),
            "pred": _parse_label(prediction_labels / f"{segment_id}.txt", predictions=True, image_size=args.image_size),
        })
    confidence_results = {}
    for threshold in (0.001, 0.01, 0.1, 0.25):
        confidence_results[str(threshold)] = {
            "class_agnostic": detection_stats(items, confidence_threshold=threshold, class_agnostic=True),
            "class_aware": detection_stats(items, confidence_threshold=threshold, class_agnostic=False),
            "class_map50": class_map50(items, DETECTOR_CLASSES),
        }
    prediction_confidences = [p["confidence"] for item in items for p in item["pred"]]

    pam_metrics = None if args.skip_pam else _pam_overfit(
        data_dir, train_rows, seed=args.seed, max_epochs=args.pam_epochs,
        frames_per_class=args.pam_frames_per_class, output=work_dir,
    )
    report = {
        "scope": "training-only memorization check; any final YOLO internal validation reuses these same training images; no held-out validation or test recordings used",
        "recordings": [row["recording_id"] for row in train_rows],
        "images": [row["segment_id"] for row in train_rows],
        "image_count": len(train_rows),
        "yolo_revision": yolo_info["commit"],
        "experiment_config": {
            "image_size": args.image_size, "model_size": args.model_size,
            "min_bandwidth": args.min_bandwidth, "min_duration": args.min_duration,
            "optimizer": args.optimizer, "lr0": args.lr0,
            "nominal_batch_size": args.nominal_batch_size,
            "warmup_min_iterations": args.warmup_min_iterations,
            "warmup_epochs_argument": args.warmup_epochs,
            "augmentation": {"mosaic": 0.0, "mixup": 0.0, "geometric": 0.0},
        },
        "label_and_coordinate_checks": label_check,
        "training_box_dimensions": box_dimensions,
        "training_setup": {
            "optimizer": args.optimizer, "lr0": args.lr0,
            "batches_per_epoch": math.ceil(len(train_rows) / 4),
            **expected_optimizer_schedule(
                epochs=args.yolo_epochs,
                batches_per_epoch=math.ceil(len(train_rows) / 4),
                batch_size=4,
                nominal_batch_size=args.nominal_batch_size,
                warmup_epochs=args.warmup_epochs,
                warmup_min_iterations=args.warmup_min_iterations,
            ),
            "warmup_min_iterations": args.warmup_min_iterations,
            "loss_gains_after_pinned_yolov5_scaling": {
                "box": float(hyp["box"] * 3 / 3),
                "cls": float(hyp["cls"] * len(DETECTOR_CLASSES) / 80 * 3 / 3),
                "obj": float(hyp["obj"] * (args.image_size / 640) ** 2 * 3 / 3),
            },
            "augmentation": {"mosaic": hyp["mosaic"], "mixup": hyp["mixup"], "copy_paste": hyp["copy_paste"], "geometry": 0.0, "hsv": [hyp["hsv_h"], hyp["hsv_s"], hyp["hsv_v"]]},
            "source_note": "The pinned train.py defaults to nbs=64 and a 100-iteration warmup floor. Non-default values are temporary, exact-string-checked diagnostic patches restored after training. YOLOv5 scales loss gains in train.py as box*3/nl, cls*nc/80*3/nl, and obj*(imgsz/640)^2*3/nl.",
        },
        "yolo": {
            "epochs_requested": args.yolo_epochs, "wall_seconds": yolo_seconds,
            "checkpoint_reused_without_retraining": args.reuse_yolo_checkpoint,
            "internal_validation_skipped_during_training_epochs": True,
            "final_epoch_validation_scope": "same eight training images; not a held-out split; direct detect.py output is the reported memorization metric",
            "same_image_confidence_threshold": 0.001, "same_image_max_detections_per_image": args.max_det,
            "prediction_directory_cleared_before_inference": True,
            "model_parameters": int(sum(parameter.numel() for parameter in model_obj.parameters())),
            "same_image_inference_seconds": detect_seconds,
            "autoanchor_log_lines": [line.strip() for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines() if "AutoAnchor:" in line],
            "checkpoint": "training checkpoint used on the identical 8 training images",
            "saved_class_names": saved_names, "class_mapping_matches": mapping_matches,
            "prediction_count_at_conf_0_001": len(prediction_confidences),
            "max_prediction_confidence": max(prediction_confidences) if prediction_confidences else None,
            "mean_prediction_confidence": float(np.mean(prediction_confidences)) if prediction_confidences else None,
            "same_image_metrics_by_confidence_threshold": confidence_results,
            "memorization_pass": (
                confidence_results["0.001"]["class_agnostic"]["recall"] is not None
                and confidence_results["0.001"]["class_agnostic"]["recall"] >= 0.95
                and confidence_results["0.001"]["class_map50"]["map50"] >= 0.95
                and mapping_matches
            ),
            "training_log": f"work/{work_dir.name}/yolo_train.log (local log, not committed)",
            "logged_training_completion": next(
                (line.strip() for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines()
                 if "epochs completed in" in line), "not available"
            ),
        "inference_log": f"work/{work_dir.name}/yolo_detect.log (local log, not committed)",
        },
        "pam_cnn": pam_metrics,
        "ground_truth_overlay_files": [entry["file"] for entry in overlay_index],
        "note": "Thresholds shown here are diagnostic thresholds on train images; none were calibrated from validation or test data.",
    }
    output_json = results_dir / "overfit_metrics.json"
    _save_json(output_json, report)
    pam_text = "The PAM memorization check is included in the JSON." if pam_metrics is not None else "This detector-only variant skipped the PAM check."
    (results_dir / "README.md").write_text(
        "# Training-only memorization check\n\n"
        f"These {len(train_rows)} spectrograms use generator-derived labels; the first four have boxes drawn in color, with class names in `ground_truth_boxes/index.json`. The detector check uses only the listed training recordings, never validation or test data. {pam_text} Exact measured outcomes are in `overfit_metrics.json`.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output_json), "recordings": report["recordings"],
        "label_check": label_check,
        "yolo_pass": report["yolo"]["memorization_pass"],
        "yolo_at_0_001": confidence_results["0.001"]["class_agnostic"],
        "pam_train_frames": None if pam_metrics is None else pam_metrics["frames"],
        "pam_classes": None if pam_metrics is None else pam_metrics["classes_present"],
        "pam_train_accuracy": None if pam_metrics is None else pam_metrics["frame_accuracy_same_training_frames"],
        "pam_epochs": None if pam_metrics is None else pam_metrics["epochs_run"],
    }, indent=2))


if __name__ == "__main__":
    main()

