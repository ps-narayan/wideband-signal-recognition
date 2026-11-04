"""Write a small, reproducible synthetic SigMF capture for CLI demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wideband_recognition.signals import SignalSpec, generate_recording
from wideband_recognition.synthetic_data import sample_specs


def make_specs(layout: str, seed: int, n_samples: int) -> list[SignalSpec]:
    if layout == "multi":
        specs, _ = sample_specs(
            seed, profile="train", n_samples=n_samples, min_bandwidth=0.025,
            min_duration=100_000, pam_subtype_index=3,
        )
        return specs
    rng = np.random.default_rng(seed)
    return [SignalSpec(
        modulation="16-QAM",
        start=50_000,
        duration=min(900_000, (n_samples // 512) * 512 - 50_000),
        center_frequency=float(rng.uniform(-0.12, 0.12)),
        bandwidth=0.04,
        carrier_phase=float(rng.uniform(-np.pi, np.pi)),
        symbol_timing=int(rng.integers(0, 34)),
        rolloff=float(rng.uniform(0.2, 0.5)),
    )]


def write_synthetic_sigmf(
    output_meta: Path,
    *,
    layout: str = "multi",
    seed: int = 2026,
    n_samples: int = 1_000_000,
    global_snr_db: float = 15.0,
    sample_rate_hz: float = 1_000_000.0,
    center_frequency_hz: float = 915_000_000.0,
) -> dict:
    output_meta = output_meta.expanduser().resolve()
    if output_meta.suffix != ".sigmf-meta":
        raise ValueError("output path must end in .sigmf-meta")
    specs = make_specs(layout, seed, n_samples)
    generated = generate_recording(specs, n_samples, seed, global_snr_db)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    data_path = output_meta.with_suffix(".sigmf-data")
    iq_components = np.empty((n_samples, 2), dtype="<f4")
    iq_components[:, 0] = generated.iq.real
    iq_components[:, 1] = generated.iq.imag
    iq_components.tofile(data_path)
    covered = (n_samples // 512) * 512
    annotations = []
    for spec in specs:
        start = max(0, int(spec.start))
        end = min(covered, int(spec.end))
        annotations.append({
            "core:sample_start": start,
            "core:sample_count": end - start,
            "core:label": spec.modulation,
            "freq_low_normalized": float(spec.f_low),
            "freq_high_normalized": float(spec.f_high),
            "detector_class": spec.detector_class,
        })
    metadata = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": float(sample_rate_hz),
            "core:frequency": float(center_frequency_hz),
            "core:version": "1.0.0",
            "core:dataset": data_path.name,
            "core:description": "Reproducible synthetic complex-baseband SigMF capture; generator parameters are included in wideband:provenance.",
            "wideband:provenance": {
                "generator": "scripts/generate_synthetic_sigmf.py",
                "seed": int(seed),
                "layout": layout,
                "requested_global_snr_db": float(global_snr_db),
                "measured_global_snr_db": generated.measured_global_snr_db,
                "snr_definition": "10*log10(total clean signal power / total added noise power across the full I/Q recording)",
                "normalized_sampling_rate_sps": 1.0,
                "covered_samples_for_spectrogram": int(covered),
                "signals": [spec.as_dict() for spec in specs],
            },
        },
        "captures": [{
            "core:sample_start": 0,
            "core:frequency": float(center_frequency_hz),
        }],
        "annotations": annotations,
    }
    output_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "metadata_path": str(output_meta),
        "data_path": str(data_path),
        "samples": int(n_samples),
        "signals": len(specs),
        "requested_global_snr_db": float(global_snr_db),
        "measured_global_snr_db": generated.measured_global_snr_db,
        "seed": int(seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_meta", type=Path, help="Destination ending in .sigmf-meta")
    parser.add_argument("--layout", choices=("pam16qam", "multi"), default="multi")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--global-snr-db", type=float, default=15.0)
    parser.add_argument("--sample-rate-hz", type=float, default=1_000_000.0)
    parser.add_argument("--center-frequency-hz", type=float, default=915_000_000.0)
    args = parser.parse_args()
    if args.samples < 150_000 or args.sample_rate_hz <= 0:
        parser.error("--samples must be at least 150000 and sample rate must be positive")
    print(json.dumps(write_synthetic_sigmf(
        args.output_meta, layout=args.layout, seed=args.seed, n_samples=args.samples,
        global_snr_db=args.global_snr_db, sample_rate_hz=args.sample_rate_hz,
        center_frequency_hz=args.center_frequency_hz,
    ), indent=2))


if __name__ == "__main__":
    main()
