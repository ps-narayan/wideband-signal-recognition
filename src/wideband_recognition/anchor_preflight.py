"""Preflight YOLOv5 default anchors against the minimum rasterized target."""

from __future__ import annotations

import numpy as np


YOLOV5N_V7_ANCHORS_PX = np.asarray(
    [
        [10, 13],
        [16, 30],
        [33, 23],
        [30, 61],
        [62, 45],
        [59, 119],
        [116, 90],
        [156, 198],
        [373, 326],
    ],
    dtype=np.float64,
)


def best_possible_recall(
    box_wh_px: tuple[float, float],
    anchors_px: np.ndarray = YOLOV5N_V7_ANCHORS_PX,
    threshold: float = 4.0,
) -> float:
    """Match YOLOv5's anchor ratio metric for one width/height box."""
    box = np.asarray(box_wh_px, dtype=np.float64)
    anchors = np.asarray(anchors_px, dtype=np.float64)
    if box.shape != (2,) or np.any(box <= 0) or anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError("box_wh_px and anchors_px must contain positive width/height pairs")
    if threshold <= 1:
        raise ValueError("anchor threshold must be greater than one")
    ratio = box[None, :] / anchors
    score = np.minimum(ratio, 1.0 / ratio).min(axis=1)
    return float(np.mean(score.max() > 1.0 / threshold))
