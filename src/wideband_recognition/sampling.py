"""Deterministic class-wise caps for training-frame storage and CPU time."""
from __future__ import annotations

import numpy as np


def cap_frames_per_class(frames, labels, rows, max_per_class: int | None, *, seed: int = 2026):
    """Keep at most N training frames for each observed class, reproducibly."""
    frames = np.asarray(frames)
    labels = np.asarray(labels)
    if len(frames) != len(labels) or len(rows) != len(labels):
        raise ValueError("frames, labels, and rows must have equal length")
    if max_per_class is None:
        return frames, labels, list(rows)
    if max_per_class < 1:
        raise ValueError("max_per_class must be positive when supplied")
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) > max_per_class:
            indices = rng.choice(indices, size=max_per_class, replace=False)
        selected.extend(indices.tolist())
    selected.sort()
    return frames[selected], labels[selected], [rows[int(i)] for i in selected]
