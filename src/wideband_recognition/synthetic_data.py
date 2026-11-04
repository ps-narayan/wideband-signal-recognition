"""Deterministic synthetic recording layouts for the training/evaluation pipeline."""

from __future__ import annotations

import math

import numpy as np

from .geometry import SignalBox
from .layout import pair_remains_resolved_after_edge_jitter
from .signals import DETECTOR_CLASSES, PAM_SUBTYPES, SignalSpec


N_SAMPLES = 1_000_000
MIN_BANDWIDTH = 0.025
MIN_DURATION = 100_000


def sample_specs(
    seed: int,
    *,
    profile: str = "train",
    n_samples: int = N_SAMPLES,
    min_bandwidth: float = MIN_BANDWIDTH,
    min_duration: int = MIN_DURATION,
    snr_override_db: float | None = None,
    pam_subtype_index: int | None = None,
) -> tuple[list[SignalSpec], float]:
    """Make one signal from each detector class with a guard-safe frequency layout."""
    if profile not in {"train", "shifted"}:
        raise ValueError("profile must be 'train' or 'shifted'")
    if min_bandwidth <= 0 or min_bandwidth > 0.04:
        raise ValueError("min_bandwidth must be in (0, 0.04] for the main profile")
    if min_duration < MIN_DURATION or min_duration > 250_000:
        raise ValueError("min_duration must be in [100000, 250000]")
    covered_samples = (n_samples // 512) * 512
    if min_duration > covered_samples:
        raise ValueError("min_duration cannot exceed the STFT-covered recording length")
    if covered_samples < MIN_DURATION:
        raise ValueError("recording does not contain one fully covered minimum-duration signal")
    rng = np.random.default_rng(seed)
    if pam_subtype_index is not None and not 0 <= pam_subtype_index < len(PAM_SUBTYPES):
        raise ValueError(f"pam_subtype_index must be in [0, {len(PAM_SUBTYPES) - 1}]")
    specs: list[SignalSpec] = []
    widths = []
    for _ in DETECTOR_CLASSES:
        if profile == "train":
            width = float(np.exp(rng.uniform(np.log(min_bandwidth), np.log(0.04))))
        else:
            width = float(rng.uniform(max(0.03, min_bandwidth), 0.05))
        widths.append(width)

    classes: list[str] = []
    for detector_class in DETECTOR_CLASSES:
        if detector_class == "PAM":
            subtype = (
                PAM_SUBTYPES[pam_subtype_index]
                if pam_subtype_index is not None else str(rng.choice(PAM_SUBTYPES))
            )
            classes.append(subtype)
        else:
            classes.append(detector_class)

    # Place widest signals first so rejection sampling remains reliable.
    order = np.argsort(widths)[::-1]
    for index in order:
        width = widths[int(index)]
        duration = int(np.exp(rng.uniform(np.log(min_duration), np.log(250_000))))
        duration = min(duration, covered_samples)
        start = int(rng.integers(0, covered_samples - duration + 1))
        placed = None
        for _attempt in range(2000):
            low = -0.5 + width / 2
            high = 0.5 - width / 2
            center = float(rng.uniform(low, high))
            candidate = SignalSpec(
                modulation=classes[int(index)],
                start=start,
                duration=duration,
                center_frequency=center,
                bandwidth=width,
                carrier_phase=float(rng.uniform(-math.pi, math.pi)),
                symbol_timing=int(rng.integers(0, max(2, int(1.5 / width)))),
                rolloff=float(rng.uniform(0.2, 0.5)),
            )
            box = SignalBox(candidate.start, candidate.end, candidate.f_low, candidate.f_high, candidate.detector_class)
            if all(
                pair_remains_resolved_after_edge_jitter(
                    SignalBox(other.start, other.end, other.f_low, other.f_high, other.detector_class),
                    box,
                )
                for other in specs
            ):
                placed = candidate
                break
        if placed is None:
            raise RuntimeError("could not sample a guard-safe signal layout")
        specs.append(placed)
    # Keep manifests and images in stable class order.
    specs.sort(key=lambda item: DETECTOR_CLASSES.index(item.detector_class))
    if any(spec.end > covered_samples or spec.duration < min_duration for spec in specs):
        raise AssertionError("sampled signal falls outside the STFT-covered time range")

    if snr_override_db is not None:
        low_snr, high_snr = (-5.0, 20.0) if profile == "train" else (-5.0, 5.0)
        if not low_snr <= snr_override_db <= high_snr:
            raise ValueError(f"snr_override_db must be in [{low_snr}, {high_snr}] for {profile} profile")
        global_snr_db = float(snr_override_db)
    elif profile == "train":
        global_snr_db = float(rng.uniform(-5.0, 20.0))
    else:
        global_snr_db = float(rng.uniform(-5.0, 5.0))
    return specs, global_snr_db


def approximate_inband_snr_db(noise_power: float, bandwidth: float) -> float:
    """Estimate SNR for a unit-power signal integrated over its occupied bandwidth."""
    if noise_power <= 0 or bandwidth <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / (noise_power * bandwidth)))
