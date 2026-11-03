"""Convert annotated SigMF I/Q recordings into paper-taxonomy YOLO segments.

This is an ingestion path, not a claim that the external dataset has been
downloaded or evaluated.  SigMF labels are mapped conservatively: PSK/QAM/APSK
families become the merged detector class PAM, while generic GMSK is rejected
because the reference paper separates GMSK1/GMSK2 by spectrogram shape.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .geometry import SignalBox, to_yolo
from .signals import PAPER_DETECTOR_CLASSES
from .spectrogram import log_spectrogram, resize_spectrogram


_SIGMF_FORMATS: dict[str, tuple[str, str]] = {
    "cf32_le": ("<f4", "float"),
    "cf32_be": (">f4", "float"),
    "cf64_le": ("<f8", "float"),
    "cf64_be": (">f8", "float"),
    "ci16_le": ("<i2", "signed"),
    "ci16_be": (">i2", "signed"),
    "cu16_le": ("<u2", "unsigned"),
    "cu16_be": (">u2", "unsigned"),
    "ci8": ("i1", "signed"),
    "cu8": ("u1", "unsigned"),
}


def _canonical_label(label: str) -> str:
    """Map an explicit modulation label to one of the paper's 12 classes."""
    original = label
    name = re.sub(r"[\s_]+", "-", label.strip()).upper()
    name = re.sub(r"-+", "-", name)
    aliases = {
        "AMDSB": "AM-DSB",
        "AMSSB": "AM-SSB",
        "2FSK": "2-FSK",
        "4FSK": "4-FSK",
        "2CPFSK": "2-CPFSK",
        "4CPFSK": "4-CPFSK",
        "GMSK-1": "GMSK1",
        "GMSK-2": "GMSK2",
    }
    name = aliases.get(name, name)
    if name in PAPER_DETECTOR_CLASSES:
        return name
    # PSK, QAM and APSK subtypes are deliberately merged in stage 1.
    if name in {"PAM", "BPSK", "QPSK", "OQPSK"} or re.fullmatch(
        r"(?:(?:M|\d+)-?)?(?:PSK|QAM|APSK)", name
    ):
        return "PAM"
    if name in {"GMSK", "GAUSSIAN-MSK"}:
        raise ValueError(
            "generic GMSK is ambiguous: assign GMSK1 or GMSK2 from the "
            "reference paper's spectrogram-shape labels"
        )
    raise ValueError(f"unsupported SigMF modulation label {original!r}")


def _effective_center_frequency(metadata: dict[str, Any], sample_start: int) -> float:
    global_meta = metadata.get("global", {})
    center = float(global_meta.get("core:frequency", 0.0))
    captures = sorted(
        metadata.get("captures", []), key=lambda item: int(item.get("core:sample_start", 0))
    )
    for capture in captures:
        if int(capture.get("core:sample_start", 0)) > sample_start:
            break
        if "core:frequency" in capture:
            center = float(capture["core:frequency"])
    return center


def _active_capture_end(metadata: dict[str, Any], sample_start: int, n_samples: int) -> int:
    """Return the next capture boundary, as SigMF annotation defaults require."""
    starts = sorted(int(item.get("core:sample_start", 0)) for item in metadata.get("captures", []))
    later = [start for start in starts if start > sample_start]
    return later[0] if later else n_samples


def load_sigmf_boxes(meta_path: str | Path, *, n_samples: int | None = None) -> list[SignalBox]:
    """Read ground-truth boxes from SigMF annotation metadata.

    ``core:sample_start``/``core:sample_count`` define time. Standard SigMF
    frequency edges are interpreted as RF Hz around the active capture center,
    or baseband Hz when the recording has no RF center, then normalized by
    ``core:sample_rate`` to cycles/sample.
    """
    path = Path(meta_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    global_meta = metadata.get("global", {})
    sample_rate = float(global_meta.get("core:sample_rate", 0.0))
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("SigMF metadata requires a positive global.core:sample_rate")
    if n_samples is None:
        n_samples = metadata.get("global", {}).get("x-sample-count")
    if n_samples is not None:
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

    boxes: list[SignalBox] = []
    for index, annotation in enumerate(metadata.get("annotations", [])):
        raw_label = annotation.get("core:label", annotation.get("modulation"))
        if raw_label is None:
            # SigMF permits annotations that describe other metadata.  They are
            # not detector labels and are ignored rather than guessed.
            continue
        class_name = _canonical_label(str(raw_label))
        start = int(annotation.get("core:sample_start", 0))
        count = annotation.get("core:sample_count")
        if count is None:
            if n_samples is None:
                raise ValueError(
                    f"annotation {index} omits core:sample_count and dataset length is unknown"
                )
            count = _active_capture_end(metadata, start, n_samples) - start
        count = int(count)
        end = start + count
        if start < 0 or count <= 0 or (n_samples is not None and end > n_samples):
            raise ValueError(f"annotation {index} has an invalid sample range")

        low_hz = annotation.get("core:freq_lower_edge")
        high_hz = annotation.get("core:freq_upper_edge")
        if (low_hz is None) != (high_hz is None):
            raise ValueError(f"annotation {index} must provide both frequency edges")
        if low_hz is not None:
            center_hz = _effective_center_frequency(metadata, start)
            low = (float(low_hz) - center_hz) / sample_rate
            high = (float(high_hz) - center_hz) / sample_rate
        elif "freq_low_normalized" in annotation and "freq_high_normalized" in annotation:
            low = float(annotation["freq_low_normalized"])
            high = float(annotation["freq_high_normalized"])
        else:
            raise ValueError(f"annotation {index} has no complete frequency-edge pair")
        if not (-0.5 <= low < high <= 0.5):
            raise ValueError(
                f"annotation {index} frequency edges normalize outside [-0.5, 0.5] cycles/sample"
            )
        boxes.append(SignalBox(float(start), float(end), low, high, class_name))
    return boxes


def _sigmf_paths(meta_path: Path) -> tuple[dict[str, Any], Path, str]:
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    global_meta = metadata.get("global", {})
    datatype = global_meta.get("core:datatype")
    if datatype not in _SIGMF_FORMATS:
        raise ValueError(f"unsupported SigMF datatype {datatype!r}")
    dataset_name = global_meta.get("core:dataset")
    if dataset_name and Path(dataset_name).name != dataset_name:
        raise ValueError("SigMF core:dataset must name a file beside its metadata")
    data_path = (meta_path.parent / dataset_name) if dataset_name else meta_path.with_suffix(".sigmf-data")
    if not data_path.is_file():
        raise FileNotFoundError(f"SigMF data file not found: {data_path}")
    if int(global_meta.get("core:num_channels", 1)) != 1:
        raise ValueError("multi-channel SigMF recordings are not supported")
    captures = metadata.get("captures", [])
    if int(global_meta.get("core:trailing_bytes", 0)) or any(
        int(capture.get("core:header_bytes", 0))
        or int(capture.get("core:global_index", capture.get("core:sample_start", 0)))
        != int(capture.get("core:sample_start", 0))
        for capture in captures
    ):
        raise ValueError("non-conforming SigMF headers or sample discontinuities are unsupported")
    return metadata, data_path, str(datatype)


def _read_sigmf_segment(data_path: Path, datatype: str, start: int, end: int) -> np.ndarray:
    dtype_text, numeric_kind = _SIGMF_FORMATS[datatype]
    dtype = np.dtype(dtype_text)
    components = np.fromfile(
        data_path,
        dtype=dtype,
        count=2 * (end - start),
        offset=2 * start * dtype.itemsize,
    )
    if components.size != 2 * (end - start):
        raise ValueError("SigMF data file ended before the declared segment")
    components = components.reshape(-1, 2).astype(np.float32)
    bits = dtype.itemsize * 8
    if numeric_kind == "signed":
        components /= float(2 ** (bits - 1))
    elif numeric_kind == "unsigned":
        maximum = float(2**bits - 1)
        components = (components - maximum / 2.0) / (maximum / 2.0)
    return (components[:, 0] + 1j * components[:, 1]).astype(np.complex64)


def _segment_boxes(
    boxes: list[SignalBox], segment_start: int, covered_samples: int
) -> list[SignalBox]:
    segment_end = segment_start + covered_samples
    result = []
    for box in boxes:
        start = max(float(segment_start), box.start)
        end = min(float(segment_end), box.end)
        if start < end:
            result.append(
                SignalBox(start - segment_start, end - segment_start, box.f_low, box.f_high, box.class_name)
            )
    return result


def convert_sigmf_recording(
    meta_path: str | Path,
    output_root: str | Path,
    *,
    recording_id: str,
    split: str,
    segment_samples: int = 1_000_000,
    image_size: int = 640,
) -> dict[str, Any]:
    """Stream a SigMF recording into spectrograms, YOLO labels, and manifests.

    The split is supplied at recording level so callers can keep every segment
    from a recording together. Raw I/Q is read one segment at a time and never
    copied into the generated dataset.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if segment_samples < 512 or image_size <= 0 or image_size % 32:
        raise ValueError("segment_samples must be >=512 and image_size a positive multiple of 32")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", recording_id):
        raise ValueError("recording_id may contain only letters, numbers, dot, underscore, and dash")

    metadata_path = Path(meta_path)
    metadata, data_path, datatype = _sigmf_paths(metadata_path)
    dtype = np.dtype(_SIGMF_FORMATS[datatype][0])
    if data_path.stat().st_size % (2 * dtype.itemsize):
        raise ValueError("SigMF complex data has an incomplete I/Q sample")
    n_samples = data_path.stat().st_size // (2 * dtype.itemsize)
    boxes = load_sigmf_boxes(metadata_path, n_samples=n_samples)
    class_ids = {name: index for index, name in enumerate(PAPER_DETECTOR_CLASSES)}
    root = Path(output_root)
    counts = {"segments": 0, "labeled_segments": 0, "signals": len(boxes), "samples": n_samples}

    for segment_index, start in enumerate(range(0, n_samples, segment_samples)):
        stop = min(start + segment_samples, n_samples)
        if stop - start < 512:
            continue
        iq = _read_sigmf_segment(data_path, datatype, start, stop)
        spectrogram, covered = log_spectrogram(iq)
        resized = resize_spectrogram(spectrogram, image_size, image_size)
        local_boxes = _segment_boxes(boxes, start, covered)
        segment_id = f"{recording_id}-seg-{segment_index:05d}"
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        metadata_dir = root / "metadata" / split
        for directory in (image_dir, label_dir, metadata_dir):
            directory.mkdir(parents=True, exist_ok=True)

        pixels = np.clip((resized + 3.0) / 6.0 * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(pixels, mode="L").save(image_dir / f"{segment_id}.png")
        labels = [
            to_yolo(box, class_ids[box.class_name], covered, image_size, image_size)
            for box in local_boxes
        ]
        (label_dir / f"{segment_id}.txt").write_text(
            "".join(f"{c} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}\n" for c, cx, cy, w, h in labels),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "source_metadata": metadata_path.name,
            "recording_id": recording_id,
            "segment_id": segment_id,
            "split": split,
            "segment_start": start,
            "segment_samples": stop - start,
            "covered_samples": covered,
            "classes": list(PAPER_DETECTOR_CLASSES),
            "boxes": [
                {
                    "start": box.start,
                    "end": box.end,
                    "f_low": box.f_low,
                    "f_high": box.f_high,
                    "class": box.class_name,
                }
                for box in local_boxes
            ],
        }
        (metadata_dir / f"{segment_id}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        counts["segments"] += 1
        counts["labeled_segments"] += bool(local_boxes)
    return counts


def write_sigmf_dataset_yaml(output_root: str | Path, path: str | Path) -> Path:
    """Write YOLOv5 dataset YAML using the paper's explicit 12-class order."""
    root = Path(output_root).resolve()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    names = ", ".join(f"'{name}'" for name in PAPER_DETECTOR_CLASSES)
    target.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\nnc: {len(PAPER_DETECTOR_CLASSES)}\nnames: [{names}]\n",
        encoding="utf-8",
    )
    (root / "class_names.json").write_text(
        json.dumps({"names": list(PAPER_DETECTOR_CLASSES)}, indent=2), encoding="utf-8"
    )
    return target
