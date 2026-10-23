"""Signal separation rules shared by generation and extraction."""

from __future__ import annotations

from collections.abc import Sequence

from .geometry import SignalBox


def passband_half_width(bandwidth: float) -> float:
    """Box half-bandwidth plus a guard of B/8."""
    _positive_bandwidth(bandwidth)
    return 5.0 * bandwidth / 8.0


def transition_width(bandwidth: float) -> float:
    """Transition width is 7B/16, matching the minimum-bandwidth FIR design."""
    _positive_bandwidth(bandwidth)
    return 7.0 * bandwidth / 16.0


def stopband_half_width(bandwidth: float) -> float:
    return passband_half_width(bandwidth) + transition_width(bandwidth)


def required_guard_band(bandwidth_a: float, bandwidth_b: float) -> float:
    """Required edge-to-edge gap for two simultaneous signals."""
    _positive_bandwidth(bandwidth_a)
    _positive_bandwidth(bandwidth_b)
    return 9.0 / 16.0 * max(bandwidth_a, bandwidth_b)


def _positive_bandwidth(bandwidth: float) -> None:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")




def required_guard_band_after_edge_jitter(
    bandwidth_a: float, bandwidth_b: float, edge_fraction: float = 0.05,
) -> float:
    """Conservative frequency gap after both boxes' edges may jitter inward."""
    _positive_bandwidth(bandwidth_a)
    _positive_bandwidth(bandwidth_b)
    if not 0 <= edge_fraction < 0.5:
        raise ValueError("edge_fraction must be in [0, 0.5)")
    jittered_max = max(bandwidth_a, bandwidth_b) * (1.0 + 2.0 * edge_fraction)
    inward_edge_motion = edge_fraction * (bandwidth_a + bandwidth_b)
    return 9.0 / 16.0 * jittered_max + inward_edge_motion


def pair_remains_resolved_after_edge_jitter(
    a: SignalBox, b: SignalBox, edge_fraction: float = 0.05,
) -> bool:
    """Guarantee two boxes stay filter-resolvable after bounded time/frequency jitter."""
    if not 0 <= edge_fraction < 0.5:
        raise ValueError("edge_fraction must be in [0, 0.5)")
    if a.end <= b.start:
        time_gap = b.start - a.end
    elif b.end <= a.start:
        time_gap = a.start - b.end
    else:
        time_gap = -1.0
    time_jitter_budget = edge_fraction * (a.duration + b.duration)
    if time_gap > time_jitter_budget:
        return True
    if a.f_high <= b.f_low:
        frequency_gap = b.f_low - a.f_high
    elif b.f_high <= a.f_low:
        frequency_gap = a.f_low - b.f_high
    else:
        return False
    needed = required_guard_band_after_edge_jitter(
        a.bandwidth, b.bandwidth, edge_fraction=edge_fraction
    )
    return frequency_gap + 1e-12 >= needed


def pair_is_resolved(a: SignalBox, b: SignalBox) -> bool:
    """True if boxes are time-disjoint or have the filter-required frequency gap."""
    if min(a.end, b.end) <= max(a.start, b.start):
        return True
    if a.f_high <= b.f_low:
        gap = b.f_low - a.f_high
    elif b.f_high <= a.f_low:
        gap = a.f_low - b.f_high
    else:
        return False
    return gap + 1e-12 >= required_guard_band(a.bandwidth, b.bandwidth)


def unresolved_pairs(boxes: Sequence[SignalBox]) -> list[tuple[int, int]]:
    """Indices of simultaneous box pairs too close for the specified filter."""
    return [
        (i, j)
        for i, a in enumerate(boxes)
        for j in range(i + 1, len(boxes))
        if not pair_is_resolved(a, boxes[j])
    ]


def assert_main_layout(boxes: Sequence[SignalBox]) -> None:
    pairs = unresolved_pairs(boxes)
    if pairs:
        raise ValueError(f"main layout has unresolved/overlapping signal pairs: {pairs}")


def resolution_statuses(boxes: Sequence[SignalBox]) -> list[str]:
    """Return one resolved/unresolved status for each input signal box."""
    unresolved = {index for pair in unresolved_pairs(boxes) for index in pair}
    return ["unresolved" if index in unresolved else "resolved" for index in range(len(boxes))]
