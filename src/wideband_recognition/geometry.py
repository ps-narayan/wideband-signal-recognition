"""Time/frequency boxes and mappings to the resized spectrogram image."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalBox:
    start: float
    end: float
    f_low: float
    f_high: float
    class_name: str = "PAM"

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def bandwidth(self) -> float:
        return self.f_high - self.f_low


def _validate_domain(box: SignalBox, covered_samples: int) -> None:
    if covered_samples <= 0:
        raise ValueError("covered_samples must be positive")
    if not (0 <= box.start < box.end <= covered_samples):
        raise ValueError("time box must satisfy 0 <= start < end <= covered_samples")
    if not (-0.5 <= box.f_low < box.f_high <= 0.5):
        raise ValueError("frequency box must satisfy -0.5 <= f_low < f_high <= 0.5")


def box_to_pixels(
    box: SignalBox,
    covered_samples: int,
    width: int = 640,
    height: int = 640,
) -> tuple[float, float, float, float]:
    """Return (x1, y1, x2, y2); image row 0 is the +0.5 frequency edge."""
    _validate_domain(box, covered_samples)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x1 = box.start / covered_samples * width
    x2 = box.end / covered_samples * width
    y1 = (0.5 - box.f_high) * height
    y2 = (0.5 - box.f_low) * height
    return x1, y1, x2, y2


def pixels_to_box(
    pixels: tuple[float, float, float, float],
    covered_samples: int,
    width: int = 640,
    height: int = 640,
    class_name: str = "PAM",
) -> SignalBox:
    """Invert box_to_pixels using the covered, not nominal, sample count."""
    if covered_samples <= 0 or width <= 0 or height <= 0:
        raise ValueError("covered_samples, width, and height must be positive")
    x1, y1, x2, y2 = pixels
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("pixel box must be valid and inside the image")
    return SignalBox(
        start=x1 / width * covered_samples,
        end=x2 / width * covered_samples,
        f_low=0.5 - y2 / height,
        f_high=0.5 - y1 / height,
        class_name=class_name,
    )


def to_yolo(
    box: SignalBox,
    class_id: int,
    covered_samples: int,
    width: int = 640,
    height: int = 640,
) -> tuple[int, float, float, float, float]:
    """Return normalized YOLO (class, center_x, center_y, width, height)."""
    x1, y1, x2, y2 = box_to_pixels(box, covered_samples, width, height)
    return (
        int(class_id),
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    )
