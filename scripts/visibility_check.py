"""Run the 50-seed minimum-box visibility check at a chosen global SNR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wideband_recognition.geometry import SignalBox, box_to_pixels
from wideband_recognition.signals import generate_recording, make_visibility_specs
from wideband_recognition.spectrogram import (
    local_contrast_sd,
    log_spectrogram,
    resize_spectrogram,
    save_annotated_spectrogram,
)


def run(seeds: int, snr_db: float, n_samples: int, output_dir: Path, seed_start: int = 0) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | int]] = []
    worst: tuple[float, int, np.ndarray, list[SignalBox], int] | None = None
    min_width_px = float("inf")
    min_height_px = float("inf")

    for seed in range(seed_start, seed_start + seeds):
        specs = make_visibility_specs(seed, n_samples)
        generated = generate_recording(specs, n_samples, seed + 100_000, snr_db)
        raw, covered = log_spectrogram(generated.iq)
        resized = resize_spectrogram(raw)
        minimum = min(specs, key=lambda item: item.bandwidth * item.duration)
        box = SignalBox(minimum.start, minimum.end, minimum.f_low, minimum.f_high, minimum.detector_class)
        x1, y1, x2, y2 = box_to_pixels(box, covered, 640, 640)
        width_px = x2 - x1
        height_px = y2 - y1
        contrast = local_contrast_sd(resized, box, covered, margin=8)
        min_width_px = min(min_width_px, width_px)
        min_height_px = min(min_height_px, height_px)
        item = {
            "seed": seed,
            "global_snr_db_measured": generated.measured_global_snr_db,
            "minimum_box_contrast_sd": contrast,
            "box_width_px": width_px,
            "box_height_px": height_px,
        }
        records.append(item)
        if worst is None or contrast < worst[0]:
            boxes = [
                SignalBox(s.start, s.end, s.f_low, s.f_high, s.detector_class)
                for s in specs
            ]
            worst = (contrast, seed, resized, boxes, covered)
        print(
            f"seed={seed:02d} global_snr_db={generated.measured_global_snr_db:.3f} "
            f"min_box_contrast_sd={contrast:.3f} size={width_px:.2f}x{height_px:.2f}px"
        )

    assert worst is not None
    passed = worst[0] >= 1.0 and min_width_px >= 5.0 and min_height_px >= 5.0
    save_annotated_spectrogram(
        worst[2], worst[3], worst[4], str(output_dir / "worst_seed_spectrogram.png")
    )
    summary = {
        "requested_global_snr_db": snr_db,
        "snr_definition": "10*log10(total clean signal power / total added noise power over the full I/Q segment)",
        "seeds_tested": seeds,
        "worst_seed": worst[1],
        "worst_local_contrast_sd_above_local_median": worst[0],
        "minimum_box_width_px": min_width_px,
        "minimum_box_height_px": min_height_px,
        "acceptance_threshold_local_sd": 1.0,
        "passed": passed,
        "per_seed": records,
    }
    (output_dir / "visibility_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"WORST seed={worst[1]} contrast={worst[0]:.3f} local SD "
        f"({min_width_px:.2f}x{min_height_px:.2f}px); "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--snr-db", type=float, default=-5.0)
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visibility"))
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    summary = run(args.seeds, args.snr_db, args.n_samples, args.output_dir, args.seed_start)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
