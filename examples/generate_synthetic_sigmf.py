"""Create a small, deterministic SigMF file for the trained CLI cascade demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wideband_recognition.signals import SignalSpec, generate_recording


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("work/synthetic-sigmf"))
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()

    sample_rate_hz = 1_000_000.0
    center_frequency_hz = 915_000_000.0
    spec = SignalSpec(
        "16-QAM", start=100_000, duration=600_000,
        center_frequency=0.15, bandwidth=0.04,
        carrier_phase=0.73, symbol_timing=3, rolloff=0.25,
    )
    recording = generate_recording(
        [spec], 1_000_000, seed=args.seed, global_snr_db=10.0
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_name = "synthetic.sigmf-data"
    metadata_name = "synthetic.sigmf-meta"

    interleaved = np.empty((recording.iq.size, 2), dtype="<f4")
    interleaved[:, 0] = recording.iq.real
    interleaved[:, 1] = recording.iq.imag
    interleaved.tofile(args.output_dir / data_name)

    metadata = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": sample_rate_hz,
            "core:frequency": center_frequency_hz,
            "core:version": "1.0.0",
            "core:dataset": data_name,
            "core:description": (
                "Generated one-signal CLI demonstration, not a benchmark"
            ),
        },
        "captures": [{
            "core:sample_start": 0,
            "core:frequency": center_frequency_hz,
        }],
        "annotations": [{
            "core:sample_start": spec.start,
            "core:sample_count": spec.duration,
            "core:freq_lower_edge": center_frequency_hz + spec.f_low * sample_rate_hz,
            "core:freq_upper_edge": center_frequency_hz + spec.f_high * sample_rate_hz,
            "core:label": spec.modulation,
        }],
    }
    (args.output_dir / metadata_name).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "metadata": str(args.output_dir / metadata_name),
        "samples": int(recording.iq.size),
        "requested_global_snr_db": 10.0,
        "measured_global_snr_db": recording.measured_global_snr_db,
        "signal_class": spec.modulation,
        "seed": args.seed,
    }, indent=2))


if __name__ == "__main__":
    main()
