"""Training-loop arithmetic for the pinned YOLOv5 v7 optimizer schedule."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np


def expected_optimizer_schedule(
    *,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    nominal_batch_size: int,
    warmup_epochs: float,
    warmup_min_iterations: int = 100,
) -> dict:
    """Mirror YOLOv5 v7's per-mini-batch accumulation and warmup loop."""
    if min(epochs, batches_per_epoch, batch_size, nominal_batch_size, warmup_min_iterations) < 1:
        raise ValueError("epoch, batch, and warmup counts must be positive")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")

    total_batches = epochs * batches_per_epoch
    warmup_batches = max(round(warmup_epochs * batches_per_epoch), warmup_min_iterations)
    last_opt_step = -1
    updates = 0
    updates_during_warmup = 0
    accumulate = max(round(nominal_batch_size / batch_size), 1)
    for ni in range(total_batches):
        if ni <= warmup_batches:
            value = np.interp(ni, [0, warmup_batches], [1, nominal_batch_size / batch_size])
            accumulate = max(1, int(np.rint(value)))
        if ni - last_opt_step >= accumulate:
            updates += 1
            updates_during_warmup += int(ni <= warmup_batches)
            last_opt_step = ni
    return {
        "micro_batches_total": total_batches,
        "warmup_micro_batches": min(warmup_batches + 1, total_batches),
        "nominal_batch_size": nominal_batch_size,
        "micro_batch_size": batch_size,
        "steady_state_accumulation": max(round(nominal_batch_size / batch_size), 1),
        "optimizer_updates_expected_from_pinned_loop": updates,
        "optimizer_updates_during_warmup": updates_during_warmup,
        "fraction_optimizer_updates_during_warmup": updates_during_warmup / updates if updates else None,
    }


@contextmanager
def temporary_yolov5_schedule(
    train_script: str | Path,
    *,
    nominal_batch_size: int,
    warmup_min_iterations: int,
):
    """Temporarily override the pinned YOLOv5 v7 schedule constants.

    Exact source markers make a revision/API change fail closed. The original
    bytes are restored even if training raises or is interrupted.
    """
    if nominal_batch_size < 1 or warmup_min_iterations < 1:
        raise ValueError("nominal_batch_size and warmup_min_iterations must be positive")
    path = Path(train_script)
    original = path.read_bytes()
    source = original.decode("utf-8")
    replacements = []
    if nominal_batch_size != 64:
        marker = "nbs = 64  # nominal batch size"
        if source.count(marker) != 1:
            raise RuntimeError("Pinned YOLOv5 train.py nominal-batch-size marker changed")
        replacements.append((marker, f"nbs = {nominal_batch_size}  # temporary nominal batch size"))
    if warmup_min_iterations != 100:
        marker = 'nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)'
        if source.count(marker) != 1:
            raise RuntimeError("Pinned YOLOv5 train.py warmup-floor marker changed")
        replacements.append((marker, f'nw = max(round(hyp["warmup_epochs"] * nb), {warmup_min_iterations})  # temporary warmup floor'))
    patched = source
    for old, new in replacements:
        patched = patched.replace(old, new, 1)
    if patched != source:
        path.write_text(patched, encoding="utf-8")
    try:
        yield
    finally:
        if patched != source:
            path.write_bytes(original)
