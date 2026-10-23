"""Band-limit and frame signal boxes for the raw-I/Q PAM classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

from .geometry import SignalBox
from .layout import unresolved_pairs


@dataclass(frozen=True)
class FilterResponse:
    taps: np.ndarray
    passband_edge: float
    stopband_edge: float
    group_delay: int


def perturb_box(
    box: SignalBox,
    covered_samples: int,
    rng: np.random.Generator,
    edge_fraction: float = 0.05,
) -> SignalBox:
    """Jitter each time edge by 5% of duration and each frequency edge by 5% of bandwidth."""
    if not 0 <= edge_fraction < 0.5:
        raise ValueError("edge_fraction must be in [0, 0.5)")
    dt = box.duration * edge_fraction
    df = box.bandwidth * edge_fraction
    start = float(np.clip(box.start + rng.uniform(-dt, dt), 0, covered_samples))
    end = float(np.clip(box.end + rng.uniform(-dt, dt), 0, covered_samples))
    f_low = float(np.clip(box.f_low + rng.uniform(-df, df), -0.5, 0.5))
    f_high = float(np.clip(box.f_high + rng.uniform(-df, df), -0.5, 0.5))
    if end <= start:
        end = min(float(covered_samples), start + 1.0)
        start = max(0.0, end - 1.0)
    if f_high <= f_low:
        f_high = min(0.5, f_low + 1e-9)
        f_low = max(-0.5, f_high - 1e-9)
    return SignalBox(start, end, f_low, f_high, box.class_name)


def design_extraction_filter(
    bandwidth: float,
    numtaps: int = 1111,
    beta: float = 6.0,
) -> FilterResponse:
    """Design the agreed Kaiser low-pass after mixing the estimated center to DC."""
    if bandwidth <= 0 or bandwidth >= 0.4:
        raise ValueError("bandwidth must be in (0, 0.4)")
    if numtaps < 3 or numtaps % 2 == 0:
        raise ValueError("numtaps must be odd and >= 3")
    passband_edge = 5.0 * bandwidth / 8.0
    stopband_edge = 17.0 * bandwidth / 16.0
    cutoff = (passband_edge + stopband_edge) / 2.0
    taps = scipy_signal.firwin(
        numtaps,
        cutoff=cutoff,
        window=("kaiser", beta),
        pass_zero="lowpass",
        fs=1.0,
        scale=True,
    ).astype(np.float64)
    return FilterResponse(taps, passband_edge, stopband_edge, (numtaps - 1) // 2)


def measure_filter_response(
    response: FilterResponse,
    worN: int = 262144,
) -> tuple[float, float]:
    """Return passband ripple in dB and worst stopband magnitude in dB."""
    frequencies, h = scipy_signal.freqz(response.taps, worN=worN, fs=1.0)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    passband = magnitude_db[frequencies <= response.passband_edge]
    stopband = magnitude_db[frequencies >= response.stopband_edge]
    ripple_db = float(passband.max() - passband.min())
    worst_stop_db = float(stopband.max())
    return ripple_db, worst_stop_db


def extract_frames(
    iq: np.ndarray,
    box: SignalBox,
    *,
    frame_samples: int = 1024,
    n_frames: int = 2,
    seed: int = 0,
    jitter: bool = False,
    unresolved: bool = False,
    taps_count: int = 1111,
) -> np.ndarray:
    """Mix, filter, compensate group delay, and return [frames, samples, (I,Q)]."""
    source = np.asarray(iq, dtype=np.complex64)
    if source.ndim != 1 or not np.iscomplexobj(source):
        raise ValueError("iq must be one-dimensional complex I/Q")
    if frame_samples <= 0 or n_frames <= 0:
        raise ValueError("frame_samples and n_frames must be positive")
    if unresolved:
        raise ValueError("cannot extract a box marked unresolved")
    if not (0 <= box.start < box.end <= source.size):
        raise ValueError("box must fit inside the I/Q recording")
    rng = np.random.default_rng(seed)
    working_box = perturb_box(box, source.size, rng) if jitter else box
    start = max(0, int(np.floor(working_box.start)))
    end = min(source.size, int(np.ceil(working_box.end)))
    available = end - start
    if available < frame_samples:
        raise ValueError("box is shorter than one classifier frame")

    filt = design_extraction_filter(working_box.bandwidth, numtaps=taps_count)
    delay = filt.group_delay
    left = max(0, start - delay)
    right = min(source.size, end + delay)
    indices = np.arange(left, right, dtype=np.float64)
    center = (working_box.f_low + working_box.f_high) / 2.0
    mixed = source[left:right] * np.exp(-2j * np.pi * center * indices)
    convolution = scipy_signal.fftconvolve(mixed, filt.taps, mode="full")
    first = start - left + delay
    last = end - left + delay
    aligned = convolution[first:last]
    if aligned.size != available:
        raise RuntimeError("filter delay compensation returned the wrong sample count")

    offsets = np.linspace(0, available - frame_samples, n_frames, dtype=np.int64)
    frames = []
    for offset in offsets:
        frame = aligned[int(offset) : int(offset) + frame_samples].astype(np.complex64)
        scale = np.sqrt(np.mean(np.abs(frame) ** 2) + 1e-12)
        frame = frame / scale
        frames.append(np.stack((frame.real, frame.imag), axis=-1))
    return np.asarray(frames, dtype=np.float32)


def read_complex_iq(path: str | Path, *, raw_dtype: str = "complex64") -> np.ndarray:
    """Read .npy, .npz (key 'iq'), or an interleaved real/imag raw file."""
    path = Path(path)
    if path.suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "iq" not in archive:
                raise ValueError("npz input must contain an 'iq' array")
            value = archive["iq"]
    else:
        dtype = np.dtype(raw_dtype)
        raw = np.fromfile(path, dtype=dtype)
        if dtype.kind == "c":
            value = raw
        elif dtype.kind == "f":
            if raw.size % 2:
                raise ValueError("interleaved raw I/Q must contain an even number of floats")
            value = raw[0::2] + 1j * raw[1::2]
        else:
            raise ValueError("raw_dtype must be complex or floating point")
    value = np.asarray(value)
    if value.ndim != 1 or not np.iscomplexobj(value):
        raise ValueError("input file must contain a one-dimensional complex IQ array")
    return value.astype(np.complex64, copy=False)


def assert_boxes_are_separated(boxes: list[SignalBox]) -> None:
    """Raise when any simultaneously active pair is not filter-resolvable."""
    pairs = unresolved_pairs(boxes)
    if pairs:
        raise ValueError(f"unresolved signal pairs: {pairs}")
