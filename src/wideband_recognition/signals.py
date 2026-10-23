"""Seeded complex-baseband signal generation for the first milestone."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from .geometry import SignalBox
from .layout import assert_main_layout, resolution_statuses, unresolved_pairs

PAM_SUBTYPES = (
    "2-PSK",
    "4-PSK",
    "8-PSK",
    "16-QAM",
    "64-QAM",
    "256-QAM",
)
NON_PAM = ("AM-DSB", "FM", "2-FSK", "OOK")
DETECTOR_CLASSES = ("AM-DSB", "FM", "OFDM", "2-FSK", "OOK", "PAM")


@dataclass(frozen=True)
class SignalSpec:
    modulation: str
    start: int
    duration: int
    center_frequency: float
    bandwidth: float
    carrier_phase: float = 0.0
    symbol_timing: int = 0
    rolloff: float = 0.35
    amplitude: float = 1.0

    @property
    def end(self) -> int:
        return self.start + self.duration

    @property
    def f_low(self) -> float:
        return self.center_frequency - self.bandwidth / 2

    @property
    def f_high(self) -> float:
        return self.center_frequency + self.bandwidth / 2

    @property
    def detector_class(self) -> str:
        return "PAM" if self.modulation in PAM_SUBTYPES else self.modulation

    def as_dict(self) -> dict[str, str | int | float]:
        result = asdict(self)
        result["detector_class"] = self.detector_class
        result["end"] = self.end
        result["f_low"] = self.f_low
        result["f_high"] = self.f_high
        return result


@dataclass(frozen=True)
class GeneratedRecording:
    iq: np.ndarray
    measured_global_snr_db: float
    signal_power: float
    noise_power: float
    unresolved_pairs: tuple[tuple[int, int], ...] = ()


def make_visibility_specs(seed: int, n_samples: int = 1_000_000) -> list[SignalSpec]:
    """Six simultaneous, frequency-separated signals, including the minimum box."""
    rng = np.random.default_rng(seed)
    rolloff = float(rng.uniform(0.2, 0.5))
    sps = max(2, int((1.0 + rolloff) / 0.008))
    start = 100_000
    specs = [
        SignalSpec(
            "2-PSK", start + int(rng.integers(0, 513)), 8192, -0.400, 0.008,
            carrier_phase=float(rng.uniform(-math.pi, math.pi)),
            symbol_timing=int(rng.integers(0, sps)),
            rolloff=rolloff,
        ),
        SignalSpec("AM-DSB", start, 240_000, -0.220, 0.040,
                   carrier_phase=float(rng.uniform(-math.pi, math.pi))),
        SignalSpec("FM", start, 320_000, -0.060, 0.040,
                   carrier_phase=float(rng.uniform(-math.pi, math.pi))),
        SignalSpec("OFDM", start, 260_000, 0.100, 0.040,
                   carrier_phase=float(rng.uniform(-math.pi, math.pi))),
        SignalSpec("2-FSK", start, 220_000, 0.260, 0.040,
                   carrier_phase=float(rng.uniform(-math.pi, math.pi))),
        SignalSpec("OOK", start, 280_000, 0.420, 0.040,
                   carrier_phase=float(rng.uniform(-math.pi, math.pi))),
    ]
    if n_samples < max(spec.end for spec in specs):
        raise ValueError("recording is too short for the visibility layout")
    assert_main_layout([_as_box(s) for s in specs])
    return specs


def _as_box(spec: SignalSpec) -> SignalBox:
    return SignalBox(spec.start, spec.end, spec.f_low, spec.f_high, spec.detector_class)


def _qam_symbols(order: int, rng: np.random.Generator, count: int) -> np.ndarray:
    side = int(round(math.sqrt(order)))
    levels = np.arange(-(side - 1), side, 2, dtype=np.float64)
    i = levels[rng.integers(0, side, count)]
    q = levels[rng.integers(0, side, count)]
    symbols = i + 1j * q
    return symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))


def _symbols(modulation: str, rng: np.random.Generator, count: int) -> np.ndarray:
    if modulation in ("2-PSK", "4-PSK", "8-PSK"):
        order = int(modulation.split("-")[0])
        phase = 2 * np.pi * rng.integers(0, order, count) / order
        return np.exp(1j * phase)
    if modulation in ("16-QAM", "64-QAM", "256-QAM"):
        return _qam_symbols(int(modulation.split("-")[0]), rng, count)
    if modulation == "OOK":
        return rng.integers(0, 2, count).astype(np.complex128)
    return rng.choice(np.array([-1.0, 1.0]), count).astype(np.complex128)


def _rrc_taps(rolloff: float, samples_per_symbol: int, span_symbols: int = 8) -> np.ndarray:
    beta = float(np.clip(rolloff, 0.05, 1.0))
    sps = max(2, int(samples_per_symbol))
    t = np.arange(-(span_symbols * sps) // 2, (span_symbols * sps) // 2 + 1) / sps
    taps = np.empty_like(t, dtype=np.float64)
    for idx, value in enumerate(t):
        if abs(value) < 1e-12:
            taps[idx] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif abs(abs(4.0 * beta * value) - 1.0) < 1e-10:
            taps[idx] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            numerator = np.sin(np.pi * value * (1.0 - beta))
            numerator += 4.0 * beta * value * np.cos(np.pi * value * (1.0 + beta))
            denominator = np.pi * value * (1.0 - (4.0 * beta * value) ** 2)
            taps[idx] = numerator / denominator
    return taps / np.sqrt(np.sum(taps**2))


def _fft_convolve(source: np.ndarray, taps: np.ndarray) -> np.ndarray:
    length = source.size + taps.size - 1
    fft_size = 1 << (length - 1).bit_length()
    result = np.fft.ifft(np.fft.fft(source, fft_size) * np.fft.fft(taps, fft_size))[:length]
    return result


def _pulse_shaped(
    modulation: str,
    duration: int,
    bandwidth: float,
    rolloff: float,
    timing: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sps = max(2, int((1.0 + rolloff) / bandwidth))
    n_symbols = math.ceil((duration + timing) / sps) + 12
    symbols = _symbols(modulation, rng, n_symbols)
    upsampled = np.zeros(n_symbols * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    taps = _rrc_taps(rolloff, sps)
    shaped = _fft_convolve(upsampled, taps)
    delay = (taps.size - 1) // 2
    start = delay + timing
    wave = shaped[start : start + duration]
    if wave.size != duration:
        wave = np.pad(wave, (0, duration - wave.size))
    return wave


def _make_baseband(spec: SignalSpec, rng: np.random.Generator) -> np.ndarray:
    n = spec.duration
    t = np.arange(n, dtype=np.float64)
    b = spec.bandwidth
    if spec.modulation in PAM_SUBTYPES or spec.modulation == "OOK":
        return _pulse_shaped(
            spec.modulation, n, b, spec.rolloff, spec.symbol_timing, rng
        )
    if spec.modulation == "AM-DSB":
        tone = np.cos(2 * np.pi * (b / 4.0) * t + rng.uniform(-np.pi, np.pi))
        return (1.0 + 0.8 * tone).astype(np.complex128)
    if spec.modulation == "FM":
        message_frequency = b / 12.0
        message = np.sin(2 * np.pi * message_frequency * t + rng.uniform(-np.pi, np.pi))
        instantaneous_offset = (b / 4.0) * message
        phase = 2 * np.pi * np.cumsum(instantaneous_offset)
        return np.exp(1j * phase)
    if spec.modulation == "2-FSK":
        sps = max(2, int(1.0 / b))
        bit_count = math.ceil(n / sps)
        offsets = rng.choice(np.array([-b / 4.0, b / 4.0]), bit_count)
        frequency = np.repeat(offsets, sps)[:n]
        return np.exp(1j * 2 * np.pi * np.cumsum(frequency))
    if spec.modulation == "OFDM":
        nfft = 4096
        n_active = max(8, int(np.floor(b * nfft)))
        n_active += n_active % 2
        half = n_active // 2
        active = np.r_[np.arange(1, half + 1), np.arange(nfft - half, nfft)]
        cp = nfft // 8
        blocks: list[np.ndarray] = []
        produced = 0
        while produced < n:
            bins = np.zeros(nfft, dtype=np.complex128)
            qpsk = np.exp(1j * (np.pi / 2) * rng.integers(0, 4, active.size))
            bins[active] = qpsk
            symbol = np.fft.ifft(bins) * np.sqrt(nfft / active.size)
            block = np.concatenate((symbol[-cp:], symbol))
            blocks.append(block)
            produced += block.size
        return np.concatenate(blocks)[:n]
    raise ValueError(f"unsupported modulation: {spec.modulation}")


def generate_recording(
    specs: list[SignalSpec],
    n_samples: int,
    seed: int,
    global_snr_db: float | None = -5.0,
    *,
    allow_overlap_stress: bool = False,
) -> GeneratedRecording:
    """Generate complex I/Q; the seed replays symbols, phases, and AWGN."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if any(s.start < 0 or s.duration <= 0 or s.end > n_samples for s in specs):
        raise ValueError("every signal must fit inside the recording")
    boxes = [_as_box(s) for s in specs]
    unresolved = tuple(unresolved_pairs(boxes))
    if unresolved and not allow_overlap_stress:
        raise ValueError(f"main recording contains unresolved pairs: {unresolved}")

    rng = np.random.default_rng(seed)
    clean = np.zeros(n_samples, dtype=np.complex128)
    for spec in specs:
        baseband = _make_baseband(spec, rng)
        power = float(np.mean(np.abs(baseband) ** 2))
        if power <= 0:
            raise ValueError(f"zero-power signal for {spec.modulation}")
        baseband = baseband / np.sqrt(power)
        local_t = np.arange(spec.duration, dtype=np.float64)
        shifted = baseband * np.exp(
            1j * (2 * np.pi * spec.center_frequency * local_t + spec.carrier_phase)
        )
        clean[spec.start : spec.end] += spec.amplitude * shifted

    signal_power = float(np.mean(np.abs(clean) ** 2))
    if global_snr_db is None:
        noise = np.zeros(n_samples, dtype=np.complex128)
        noise_power = 0.0
        measured_snr = float("inf")
    else:
        target_noise_power = signal_power / (10.0 ** (global_snr_db / 10.0))
        noise = np.sqrt(target_noise_power / 2.0) * (
            rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
        )
        noise_power = float(np.mean(np.abs(noise) ** 2))
        measured_snr = 10.0 * np.log10(signal_power / noise_power)
    return GeneratedRecording(
        iq=(clean + noise).astype(np.complex64),
        measured_global_snr_db=float(measured_snr),
        signal_power=signal_power,
        noise_power=noise_power,
        unresolved_pairs=unresolved,
    )


def save_generator_metadata(
    path: str | Path,
    specs: list[SignalSpec],
    n_samples: int,
    seed: int,
    requested_global_snr_db: float | None,
    generated: GeneratedRecording,
) -> None:
    """Store replay parameters without saving the large I/Q array."""
    boxes = [_as_box(spec) for spec in specs]
    metadata = {
        "schema_version": 1,
        "seed": int(seed),
        "n_samples": int(n_samples),
        "normalized_sampling_rate_sps": 1.0,
        "requested_global_snr_db": requested_global_snr_db,
        "measured_global_snr_db": (
            generated.measured_global_snr_db
            if math.isfinite(generated.measured_global_snr_db)
            else None
        ),
        "signal_power": generated.signal_power,
        "noise_power": generated.noise_power,
        "signals": [spec.as_dict() for spec in specs],
        "resolution_status": resolution_statuses(boxes),
        "unresolved_pairs": [list(pair) for pair in generated.unresolved_pairs],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
