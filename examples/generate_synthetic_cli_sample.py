"""Generate a tiny deterministic IQ file for the energy-only CLI example."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wideband_recognition.signals import SignalSpec, generate_recording


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="work/synthetic_cli_sample.npy")
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()

    spec = SignalSpec(
        modulation="2-PSK",
        start=0,
        duration=8192,
        center_frequency=0.10,
        bandwidth=0.04,
        carrier_phase=0.7,
        symbol_timing=2,
        rolloff=0.35,
    )
    recording = generate_recording([spec], 8192, seed=args.seed, global_snr_db=10.0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, recording.iq, allow_pickle=False)
    print(f"Saved {recording.iq.size} complex samples to {output}")
    print(f"Requested global SNR: 10.0 dB; measured: {recording.measured_global_snr_db:.4f} dB")


if __name__ == "__main__":
    main()
