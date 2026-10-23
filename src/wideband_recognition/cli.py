"""File-driven inference CLI for SigMF, NumPy, and raw complex-I/Q recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .extraction import extract_frames, read_complex_iq
from .geometry import SignalBox, box_to_pixels, pixels_to_box
from .layout import resolution_statuses
from .signals import DETECTOR_CLASSES
from .spectrogram import log_spectrogram, resize_spectrogram

YOLOV5_COMMIT = "f9578796b8a64889aa9e66dc06914d19f40d338e"
IMAGE_SIZE = 640
DEFAULT_SEGMENT_SAMPLES = 1_000_000


def _sigmf_pair(path: Path) -> tuple[Path, Path]:
    """Return metadata and data paths for either member of a conforming pair."""
    if path.suffix == ".sigmf-meta":
        meta_path = path
    elif path.suffix == ".sigmf-data":
        meta_path = path.with_suffix(".sigmf-meta")
    else:
        raise ValueError("expected a .sigmf-meta or .sigmf-data path")
    if not meta_path.is_file():
        raise FileNotFoundError(f"SigMF metadata file not found: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    data_name = metadata.get("global", {}).get("core:dataset")
    if data_name and (Path(data_name).name != data_name or "/" in data_name or "\\" in data_name):
        raise ValueError("SigMF core:dataset must name a file in the metadata directory")
    data_path = (meta_path.parent / data_name).resolve() if data_name else meta_path.with_suffix(".sigmf-data")
    if not data_path.is_file():
        raise FileNotFoundError(f"SigMF data file not found: {data_path}")
    return meta_path, data_path


def _read_sigmf_samples(data_path: Path, datatype: str) -> np.ndarray:
    """Read supported interleaved SigMF complex datatypes; fail for others."""
    formats = {
        "cf32_le": ("<f4", "float"), "cf32_be": (">f4", "float"),
        "cf64_le": ("<f8", "float"), "cf64_be": (">f8", "float"),
        "ci16_le": ("<i2", "signed"), "ci16_be": (">i2", "signed"),
        "cu16_le": ("<u2", "unsigned"), "cu16_be": (">u2", "unsigned"),
        "ci8": ("i1", "signed"), "cu8": ("u1", "unsigned"),
    }
    if datatype not in formats:
        raise ValueError(f"unsupported SigMF core:datatype {datatype!r}; supported: {', '.join(formats)}")
    dtype_text, numeric_kind = formats[datatype]
    raw = np.fromfile(data_path, dtype=np.dtype(dtype_text))
    if raw.size % 2:
        raise ValueError("SigMF complex data must contain an even number of I/Q components")
    components = raw.reshape(-1, 2).astype(np.float32)
    component_bits = np.dtype(dtype_text).itemsize * 8
    if numeric_kind == "signed":
        components /= float(2 ** (component_bits - 1))
    elif numeric_kind == "unsigned":
        maximum = float(2**component_bits - 1)
        components = (components - maximum / 2.0) / (maximum / 2.0)
    return (components[:, 0] + 1j * components[:, 1]).astype(np.complex64)


def read_iq_input(
    path: str | Path,
    *,
    sample_rate_hz: float | None = None,
    raw_dtype: str = "complex64",
) -> tuple[np.ndarray, float | None, dict[str, Any]]:
    """Read supported input types and return samples, rate, and provenance."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    info: dict[str, Any] = {"input_path": source.name, "format": source.suffix.lstrip(".") or "raw"}
    rate_from_metadata = None
    if source.suffix in {".sigmf-data", ".sigmf-meta"}:
        meta_path, data_path = _sigmf_pair(source)
        sigmf = json.loads(meta_path.read_text(encoding="utf-8"))
        global_meta = sigmf.get("global", {})
        if int(global_meta.get("core:num_channels", 1)) != 1:
            raise ValueError("multi-channel SigMF input is not supported; provide one complex-IQ channel")
        captures = sigmf.get("captures", [])
        if int(global_meta.get("core:trailing_bytes", 0)) != 0 or any(int(c.get("core:header_bytes", 0)) != 0 for c in captures):
            raise ValueError("SigMF non-conforming header/trailing bytes are not supported")
        if any(int(c.get("core:global_index", c.get("core:sample_start", 0))) != int(c.get("core:sample_start", 0)) for c in captures):
            raise ValueError("SigMF recordings with sample discontinuities are not supported")
        datatype = global_meta.get("core:datatype")
        if not datatype:
            raise ValueError("SigMF metadata is missing required global.core:datatype")
        iq = _read_sigmf_samples(data_path, datatype)
        rate_from_metadata = global_meta.get("core:sample_rate")
        info.update({"format": "SigMF", "sigmf_datatype": datatype,
                     "metadata_path": meta_path.name, "data_path": data_path.name})
        center_hz = global_meta.get("core:frequency")
        captures = sigmf.get("captures", [])
        if center_hz is None and captures:
            frequencies = {float(c["core:frequency"]) for c in captures if "core:frequency" in c}
            if len(frequencies) == 1 and all("core:frequency" in c for c in captures):
                center_hz = frequencies.pop()
        if center_hz is not None:
            info["center_frequency_hz"] = float(center_hz)
    else:
        iq = read_complex_iq(source, raw_dtype=raw_dtype)
    if sample_rate_hz is not None and sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if rate_from_metadata is not None and float(rate_from_metadata) <= 0:
        raise ValueError("SigMF core:sample_rate must be positive")
    if rate_from_metadata is not None and sample_rate_hz is not None:
        error = abs(float(rate_from_metadata) - sample_rate_hz) / float(rate_from_metadata)
        if error > 1e-9:
            raise ValueError("--sample-rate-hz conflicts with SigMF core:sample_rate")
    rate = float(rate_from_metadata) if rate_from_metadata is not None else sample_rate_hz
    if rate is not None and rate <= 0:
        raise ValueError("sample rate from metadata or command line must be positive")
    if rate is not None:
        info["sample_rate_hz"] = float(rate)
    info["samples"] = int(iq.size)
    return iq, rate, info


def make_segments(iq: np.ndarray, segment_samples: int) -> list[tuple[int, np.ndarray]]:
    """Bound memory by splitting input; omit only tails shorter than an STFT window."""
    if segment_samples < 512:
        raise ValueError("segment_samples must be at least the 512-sample STFT window")
    chunks = [
        (start, iq[start : start + segment_samples])
        for start in range(0, len(iq), segment_samples)
        if len(iq[start : start + segment_samples]) >= 512
    ]
    if not chunks:
        raise ValueError("input has fewer than 512 usable samples")
    return chunks


def _save_detector_image(resized: np.ndarray, destination: Path) -> None:
    pixels = np.clip((resized + 3.0) / 6.0 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(destination)


def _read_yolo_labels(path: Path) -> list[dict[str, Any]]:
    predictions = []
    if not path.is_file():
        return predictions
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = list(map(float, line.split()))
        if len(parts) != 6:
            raise ValueError(f"expected YOLO label with confidence on line {line_number} in {path}")
        class_id, cx, cy, width, height, confidence = parts
        class_index = int(class_id)
        if class_index != class_id or not 0 <= class_index < len(DETECTOR_CLASSES):
            raise ValueError(f"invalid detector class id {class_id} in {path}")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"invalid normalized YOLO box in {path}: {line.strip()}")
        x1, y1 = (cx - width / 2) * IMAGE_SIZE, (cy - height / 2) * IMAGE_SIZE
        x2, y2 = (cx + width / 2) * IMAGE_SIZE, (cy + height / 2) * IMAGE_SIZE
        if not (0 <= x1 < x2 <= IMAGE_SIZE and 0 <= y1 < y2 <= IMAGE_SIZE):
            raise ValueError(f"YOLO box falls outside image in {path}: {line.strip()}")
        predictions.append({"class": DETECTOR_CLASSES[class_index], "confidence": float(confidence),
                            "xyxy": (x1, y1, x2, y2)})
    return predictions


def _load_pam_model(checkpoint_path: Path, device: str):
    import torch
    from .pam_model import PAMCNN, PAM_LABELS

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    labels = tuple(checkpoint.get("labels", ()))
    if labels != PAM_LABELS or checkpoint.get("frame_samples") != 1024:
        raise ValueError("PAM checkpoint labels or frame size do not match this CLI")
    model = PAMCNN(classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, labels


def extract_cli_frames(iq: np.ndarray, box: SignalBox) -> np.ndarray:
    """Deterministic, seed-free extraction used by deployment inference."""
    return extract_frames(iq, box, frame_samples=1024, n_frames=2, jitter=False)


def _classify_pam_box(iq: np.ndarray, box: SignalBox, model, labels: tuple[str, ...], device: str):
    import torch

    frames = extract_cli_frames(iq, box)
    with torch.no_grad():
        predictions = model(torch.from_numpy(frames.transpose(0, 2, 1)).to(device)).argmax(1).cpu().tolist()
    counts: dict[int, int] = {}
    for label in predictions:
        counts[label] = counts.get(label, 0) + 1
    chosen = max(counts, key=lambda value: (counts[value], -predictions.index(value)))
    return labels[chosen]


def _frequency_output(box: SignalBox, sample_rate_hz: float | None, center_hz: float | None):
    result: dict[str, Any] = {"normalized_cycles_per_sample": [float(box.f_low), float(box.f_high)]}
    if sample_rate_hz is not None:
        offsets = [float(box.f_low * sample_rate_hz), float(box.f_high * sample_rate_hz)]
        result["offset_hz_from_baseband_center"] = offsets
        if center_hz is not None:
            result["absolute_hz"] = [float(center_hz + offsets[0]), float(center_hz + offsets[1])]
    return result


def _annotate(resized: np.ndarray, detections: list[dict[str, Any]], covered: int, destination: Path):
    pixels = np.clip((resized + 3.0) / 6.0 * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    for row in detections:
        coords = box_to_pixels(row["_box"], covered, IMAGE_SIZE, IMAGE_SIZE)
        draw.rectangle(coords, outline=(255, 65, 65), width=2)
        score = (f"{row['confidence']:.2f}" if row.get("confidence") is not None
                 else f"z={row['energy_score']:.2f}")
        label = f"{row['detector_class']} {score}"
        if row.get("pam_subtype"):
            label += f" / {row['pam_subtype']}"
        draw.text((int(coords[0]), max(0, int(coords[1]) - 12)), label, fill=(255, 255, 0))
    image.save(destination)


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    """Run detector and optional subtype model on actual samples; never use a seed."""
    source = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    backend = getattr(args, "backend", "yolo")
    if backend not in {"yolo", "energy"}:
        raise ValueError("backend must be yolo or energy")
    sha, weights, detect_script = None, None, None
    if backend == "yolo":
        if not args.detector_weights or not args.yolo_dir:
            raise ValueError("YOLO backend requires --detector-weights and --yolo-dir")
        yolo_dir = Path(args.yolo_dir).expanduser().resolve()
        weights = Path(args.detector_weights).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"detector weights not found: {weights}")
        sha = subprocess.check_output(["git", "-C", str(yolo_dir), "rev-parse", "HEAD"], text=True).strip()
        if sha != YOLOV5_COMMIT:
            raise RuntimeError(f"YOLOv5 revision mismatch: expected {YOLOV5_COMMIT}, found {sha}")
        detect_script = yolo_dir / "detect.py"
        if not detect_script.is_file():
            raise FileNotFoundError(f"detect.py not found in YOLOv5 checkout: {yolo_dir}")
    if not 0 <= args.confidence_threshold <= 1.01:
        raise ValueError("confidence threshold must be in [0, 1.01]")
    if backend == "energy" and args.pam_weights:
        raise ValueError("PAM subtype inference requires YOLO PAM boxes; omit --pam-weights for energy mode")
    if args.device == "auto" and (backend == "yolo" or args.pam_weights):
        import torch
        use_cuda = torch.cuda.is_available()
        detector_device, pam_device = ("0", "cuda:0") if use_cuda else ("cpu", "cpu")
    elif args.device == "cpu" or backend == "energy":
        detector_device, pam_device = "cpu", "cpu"
    else:
        detector_device, pam_device = args.device, f"cuda:{args.device}"

    iq, sample_rate_hz, source_info = read_iq_input(
        source, sample_rate_hz=args.sample_rate_hz, raw_dtype=args.raw_dtype
    )
    segments = make_segments(iq, args.segment_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    pam_model, pam_labels = None, None
    if args.pam_weights:
        pam_path = Path(args.pam_weights).expanduser().resolve()
        if not pam_path.is_file():
            raise FileNotFoundError(f"PAM checkpoint not found: {pam_path}")
        pam_model, pam_labels = _load_pam_model(pam_path, pam_device)

    with tempfile.TemporaryDirectory(prefix="wideband-cli-") as temporary:
        temp = Path(temporary)
        image_dir = temp / "images"
        image_dir.mkdir()
        rendered = {}
        for index, (start, chunk) in enumerate(segments):
            spectrum, covered = log_spectrogram(chunk)
            resized = resize_spectrogram(spectrum, IMAGE_SIZE, IMAGE_SIZE)
            stem = f"segment_{index:05d}"
            _save_detector_image(resized, image_dir / f"{stem}.png")
            rendered[stem] = (start, chunk, covered, resized)
        label_dir = None
        if backend == "yolo":
            detect_output = temp / "detect"
            command = [
                sys.executable, str(detect_script), "--weights", str(weights),
                "--source", str(image_dir), "--img", str(IMAGE_SIZE),
                "--conf-thres", "0.001", "--iou-thres", "0.6", "--max-det", "300",
                "--device", detector_device, "--save-txt", "--save-conf", "--project",
                str(detect_output), "--name", "result", "--exist-ok", "--nosave",
            ]
            subprocess.run(command, cwd=yolo_dir, check=True)
            label_dir = detect_output / "result" / "labels"
        else:
            from .baseline import energy_proposals
        detections = []
        for stem, (segment_start, chunk, covered, resized) in rendered.items():
            if backend == "yolo":
                predictions = [
                    prediction for prediction in _read_yolo_labels(label_dir / f"{stem}.txt")
                    if prediction["confidence"] >= args.confidence_threshold
                ]
            else:
                proposals = energy_proposals(
                    image_dir / f"{stem}.png", mask_threshold=args.energy_mask_threshold
                )
                predictions = [{
                    "class": "unknown", "confidence": None,
                    "energy_score": proposal["confidence"], "xyxy": tuple(proposal["xyxy"]),
                } for proposal in proposals]
            boxes = [pixels_to_box(prediction["xyxy"], covered, class_name=prediction["class"])
                     for prediction in predictions]
            statuses = resolution_statuses(boxes)
            annotated_rows = []
            for prediction, box, status in zip(predictions, boxes, statuses):
                row: dict[str, Any] = {
                    "detector_class": prediction["class"],
                    "confidence": prediction["confidence"],
                    "time_sample_range": [float(segment_start + box.start), float(segment_start + box.end)],
                    "frequency_range": _frequency_output(
                        box, sample_rate_hz, source_info.get("center_frequency_hz")
                    ),
                    "resolution_status": status,
                    "pam_subtype": None,
                    "pam_classifier_status": "not_run",
                    "_box": box,
                }
                if backend == "energy":
                    row["energy_score"] = prediction["energy_score"]
                    row["energy_score_type"] = "peak normalized spectrogram score; not a probability"
                    row["pam_classifier_status"] = "not_available_from_geometry_baseline"
                if sample_rate_hz is not None:
                    row["time_seconds"] = [
                        float((segment_start + box.start) / sample_rate_hz),
                        float((segment_start + box.end) / sample_rate_hz),
                    ]
                if prediction["class"] == "PAM" and pam_model is not None:
                    if status == "unresolved":
                        row["pam_classifier_status"] = "skipped_unresolved"
                    else:
                        try:
                            row["pam_subtype"] = _classify_pam_box(
                                chunk, box, pam_model, pam_labels, pam_device
                            )
                            row["pam_classifier_status"] = "classified"
                        except (ValueError, RuntimeError) as exc:
                            row["pam_classifier_status"] = f"unavailable: {exc}"
                annotated_rows.append(row)
            _annotate(resized, annotated_rows, covered, output_dir / f"annotated_{stem}.png")
            for row in annotated_rows:
                row.pop("_box")
            detections.extend(annotated_rows)

    report = {
        "schema_version": 1,
        "run_notice": (
            "reduced run: CPU, small data, not comparable to target"
            if backend == "energy" else None
        ),
        "input": source_info,
        "sample_rate_hz": sample_rate_hz,
        "time_convention": "Sample ranges use indices relative to the input file; seconds are elapsed from the start of that file.",
        "frequency_convention": (
            "Normalized baseband cycles/sample is in [-0.5, 0.5]. Hz values are offsets from "
            "baseband center; absolute_hz requires SigMF core:frequency."
        ),
        "detector": {
            "backend": backend,
            "architecture": "YOLOv5n" if backend == "yolo" else "Energy geometry baseline",
            "source_commit": sha, "weights": weights.name if weights else None,
            "confidence_threshold": args.confidence_threshold if backend == "yolo" else None,
            "energy_mask_threshold": args.energy_mask_threshold if backend == "energy" else None,
        },
        "pam_classifier": {
            "enabled": pam_model is not None,
            "weights": Path(args.pam_weights).name if args.pam_weights else None,
            "labels": list(pam_labels) if pam_labels else [],
        },
        "segment_samples": args.segment_samples,
        "segments_processed": len(segments),
        "covered_samples_per_segment": [int(rendered[f"segment_{i:05d}"][2]) for i in range(len(segments))],
        "samples_processed": int(sum(rendered[f"segment_{i:05d}"][2] for i in range(len(segments)))),
        "samples_not_processed": int(iq.size - sum(rendered[f"segment_{i:05d}"][2] for i in range(len(segments)))),
        "annotated_images": [f"annotated_segment_{index:05d}.png" for index in range(len(segments))],
        "detections": detections,
        "seed_dependent": False,
    }
    (output_dir / "detections.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="SigMF pair, .npy, .npz, or raw complex-I/Q file")
    parser.add_argument("--backend", choices=("yolo", "energy"), default="yolo",
                        help="trained YOLO detector or class-agnostic geometry baseline")
    parser.add_argument("--detector-weights", help="Milestone 2 YOLOv5n best.pt; required for yolo backend")
    parser.add_argument("--yolo-dir", help="checkout at the pinned YOLOv5 revision; required for yolo backend")
    parser.add_argument("--energy-mask-threshold", type=float, default=1.0,
                        help="normalized spectrogram mask threshold for energy backend")
    parser.add_argument("--output-dir", default="outputs/cli", help="directory for JSON and annotated PNG files")
    parser.add_argument("--pam-weights", help="optional Milestone 2 PAM CNN checkpoint")
    parser.add_argument("--sample-rate-hz", type=float, help="optional for raw/NumPy; required for Hz and seconds")
    parser.add_argument("--raw-dtype", default="complex64", help="raw file dtype, default complex64")
    parser.add_argument("--segment-samples", type=int, default=DEFAULT_SEGMENT_SAMPLES)
    parser.add_argument("--confidence-threshold", type=float, default=0.25,
                        help="use the validation-calibrated threshold from metrics.json")
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device index such as 0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_inference(args)
    print(json.dumps({
        "status": "ok",
        "samples": report["input"]["samples"],
        "segments_processed": report["segments_processed"],
        "detections": len(report["detections"]),
        "json": str(Path(args.output_dir).resolve() / "detections.json"),
        "annotated_images": report["annotated_images"],
    }, indent=2))


if __name__ == "__main__":
    main()








