"""Paper-style STFT and direct 640x640 spectrogram resizing."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .geometry import SignalBox, box_to_pixels


def covered_sample_count(n_samples: int, window_samples: int = 512) -> int:
    if n_samples < window_samples or window_samples <= 0:
        raise ValueError("recording must contain at least one positive-size window")
    return (n_samples // window_samples) * window_samples


def log_spectrogram(
    iq: np.ndarray,
    window_samples: int = 512,
    fft_bins: int = 4096,
) -> tuple[np.ndarray, int]:
    """Return normalized [time, shifted-frequency] log-magnitude and coverage."""
    iq = np.asarray(iq)
    if iq.ndim != 1 or not np.iscomplexobj(iq):
        raise ValueError("iq must be a one-dimensional complex array")
    if fft_bins < window_samples:
        raise ValueError("fft_bins must be >= window_samples")
    covered = covered_sample_count(iq.size, window_samples)
    frames = iq[:covered].reshape(-1, window_samples)
    window = np.hanning(window_samples).astype(np.float32)
    spectrum = np.fft.fft(frames * window[None, :], n=fft_bins, axis=1)
    magnitude = np.abs(np.fft.fftshift(spectrum, axes=1)[:, ::-1])
    logmag = 20.0 * np.log10(magnitude + 1e-12)
    mean = float(logmag.mean())
    std = float(logmag.std())
    if std <= 1e-12:
        raise ValueError("spectrogram standard deviation is zero")
    normalized = ((logmag - mean) / std).astype(np.float32)
    return normalized, covered


def resize_spectrogram(
    normalized: np.ndarray,
    width: int = 640,
    height: int = 640,
) -> np.ndarray:
    if normalized.ndim != 2 or width <= 0 or height <= 0:
        raise ValueError("expected a 2-D spectrogram and positive output size")
    # STFT is time-by-frequency; image columns are time and rows are frequency.
    image = Image.fromarray(np.asarray(normalized, dtype=np.float32).T, mode="F")
    resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def save_annotated_spectrogram(
    resized: np.ndarray,
    boxes: list[SignalBox],
    covered_samples: int,
    path: str,
) -> None:
    display = np.clip((resized + 3.0) / 6.0 * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(display, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        coords = box_to_pixels(box, covered_samples, image.width, image.height)
        draw.rectangle(coords, outline=(255, 64, 64), width=1)
    image.save(path)


def local_contrast_sd(
    resized: np.ndarray,
    box: SignalBox,
    covered_samples: int,
    margin: int = 8,
) -> float:
    """Median(box)-median(local annulus), divided by annulus standard deviation."""
    height, width = resized.shape
    x1, y1, x2, y2 = box_to_pixels(box, covered_samples, width, height)
    left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    right, bottom = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    target = resized[top:bottom, left:right]
    outer_left, outer_top = max(0, left - margin), max(0, top - margin)
    outer_right, outer_bottom = min(width, right + margin), min(height, bottom + margin)
    outer = resized[outer_top:outer_bottom, outer_left:outer_right]
    mask = np.ones(outer.shape, dtype=bool)
    mask[top - outer_top : bottom - outer_top, left - outer_left : right - outer_left] = False
    background = outer[mask]
    if target.size == 0 or background.size < 8:
        raise ValueError("box or local background is too small for contrast estimate")
    local_std = float(background.std())
    if local_std <= 1e-12:
        raise ValueError("local background has zero standard deviation")
    return float((np.median(target) - np.median(background)) / local_std)
