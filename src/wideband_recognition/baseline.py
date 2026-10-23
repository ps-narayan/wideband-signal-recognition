"""A class-agnostic spectrogram energy/connected-component geometry baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def energy_proposals(path: str | Path, *, mask_threshold: float = 1.0) -> list[dict]:
    """Threshold normalized log-power and return connected-component boxes."""
    pixels = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    normalized = pixels / 255.0 * 6.0 - 3.0
    mask = normalized >= mask_threshold
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    proposals = []
    height, width = normalized.shape
    for label_id, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        y_slice, x_slice = region
        x1, x2 = int(x_slice.start), int(x_slice.stop)
        y1, y2 = int(y_slice.start), int(y_slice.stop)
        area = int(np.count_nonzero(labels[y_slice, x_slice] == label_id))
        box_area = (x2 - x1) * (y2 - y1)
        if x2 - x1 < 2 or y2 - y1 < 2 or area < 4:
            continue
        if box_area > 0.5 * height * width:
            continue
        peak = float(normalized[y_slice, x_slice][labels[y_slice, x_slice] == label_id].max())
        proposals.append(
            {
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "class": "unknown",
                "confidence": peak,
            }
        )
    return proposals
