"""Seed metadata and YOLO-ready spectrogram generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import SignalBox, to_yolo
from .layout import resolution_statuses
from .signals import DETECTOR_CLASSES, GeneratedRecording, SignalSpec, generate_recording
from .spectrogram import log_spectrogram, resize_spectrogram
from .synthetic_data import approximate_inband_snr_db, sample_specs


def grouped_split(
    n_recordings: int = 90,
    train_fraction: float = 2 / 3,
    validation_fraction: float = 1 / 9,
    seed: int = 2026,
) -> dict[str, str]:
    """Assign parent recordings before any per-segment sampling."""
    if n_recordings <= 0 or train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split sizes and fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")
    ids = [f"main-{i:03d}" for i in range(n_recordings)]
    np.random.default_rng(seed).shuffle(ids)
    n_train = int(round(n_recordings * train_fraction))
    n_val = int(round(n_recordings * validation_fraction))
    if n_train + n_val >= n_recordings:
        n_val = n_recordings - n_train - 1
    result = {key: "train" for key in ids[:n_train]}
    result.update({key: "val" for key in ids[n_train : n_train + n_val]})
    result.update({key: "test" for key in ids[n_train + n_val :]})
    return result


def _boxes(specs: list[SignalSpec]) -> list[SignalBox]:
    return [
        SignalBox(spec.start, spec.end, spec.f_low, spec.f_high, spec.detector_class)
        for spec in specs
    ]


def write_segment_artifacts(
    root: str | Path,
    segment_id: str,
    recording_id: str,
    split: str,
    seed: int,
    profile: str,
    *,
    n_samples: int = 1_000_000,
    image_size: int = 640,
) -> dict:
    """Regenerate one segment and write only image, YOLO labels, and replay JSON."""
    root = Path(root)
    specs, requested_snr = sample_specs(seed, profile=profile, n_samples=n_samples)
    generated = generate_recording(specs, n_samples, seed, requested_snr)
    normalized, covered = log_spectrogram(generated.iq)
    resized = resize_spectrogram(normalized, image_size, image_size)
    boxes = _boxes(specs)

    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    meta_dir = root / "metadata" / split
    for directory in (image_dir, label_dir, meta_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pixels = np.clip((resized + 3.0) / 6.0 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(image_dir / f"{segment_id}.png")
    class_to_id = {name: i for i, name in enumerate(DETECTOR_CLASSES)}
    labels = [
        to_yolo(box, class_to_id[box.class_name], covered, image_size, image_size)
        for box in boxes
    ]
    (label_dir / f"{segment_id}.txt").write_text(
        "".join(f"{c} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}\n" for c, cx, cy, w, h in labels),
        encoding="utf-8",
    )

    signal_meta = []
    for spec in specs:
        data = spec.as_dict()
        data["approximate_inband_snr_db"] = approximate_inband_snr_db(
            generated.noise_power, spec.bandwidth
        )
        signal_meta.append(data)
    metadata = {
        "schema_version": 2,
        "segment_id": segment_id,
        "recording_id": recording_id,
        "split": split,
        "seed": int(seed),
        "profile": profile,
        "n_samples": int(n_samples),
        "covered_samples": int(covered),
        "normalized_sampling_rate_sps": 1.0,
        "requested_global_snr_db": requested_snr,
        "measured_global_snr_db": generated.measured_global_snr_db,
        "signal_power": generated.signal_power,
        "noise_power": generated.noise_power,
        "signals": signal_meta,
        "resolution_status": resolution_statuses(boxes),
        "unresolved_pairs": [list(pair) for pair in generated.unresolved_pairs],
    }
    (meta_dir / f"{segment_id}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def write_dataset_yaml(root: str | Path, path: str | Path) -> Path:
    """Create the version-7 YOLO data file with paths relative to dataset root."""
    root = Path(root).resolve()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    names = ", ".join(f"'{name}'" for name in DETECTOR_CLASSES)
    target.write_text(
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(DETECTOR_CLASSES)}\n"
        f"names: [{names}]\n",
        encoding="utf-8",
    )
    return target


def metadata_box(item: dict, class_name: str | None = None) -> SignalBox:
    return SignalBox(
        float(item["start"]),
        float(item["end"]),
        float(item["f_low"]),
        float(item["f_high"]),
        str(class_name or item["detector_class"]),
    )


def segment_snr_bin(snr: float) -> str:
    edges = [-5, 0, 5, 10, 15, 20, 25]
    for low, high in zip(edges, edges[1:]):
        if low <= snr < high:
            return f"[{low},{high})"
    return f"<{edges[0]}" if snr < edges[0] else f">={edges[-2]}"
