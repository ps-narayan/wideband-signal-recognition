import unittest

from wideband_recognition.geometry import SignalBox
from wideband_recognition.layout import (
    passband_half_width,
    required_guard_band,
    required_guard_band_after_edge_jitter,
    pair_remains_resolved_after_edge_jitter,
    stopband_half_width,
    transition_width,
    unresolved_pairs,
    resolution_statuses,
)


class LayoutTests(unittest.TestCase):
    def test_filter_rule_matches_minimum_bandwidth_spec(self):
        bandwidth = 0.008
        self.assertAlmostEqual(passband_half_width(bandwidth), 0.005)
        self.assertAlmostEqual(transition_width(bandwidth), 0.0035)
        self.assertAlmostEqual(stopband_half_width(bandwidth), 0.0085)
        self.assertAlmostEqual(required_guard_band(bandwidth, bandwidth), 0.0045)

    def test_main_layout_enforces_guard_for_simultaneous_signals(self):
        a = SignalBox(0, 1000, -0.104, -0.096)
        b = SignalBox(0, 1000, -0.0915, -0.0835)
        self.assertEqual(unresolved_pairs([a, b]), [])

    def test_too_close_and_overlapping_boxes_are_unresolved(self):
        a = SignalBox(0, 1000, -0.104, -0.096)
        too_close = SignalBox(0, 1000, -0.100, -0.092)
        overlap = SignalBox(0, 1000, -0.101, -0.093)
        self.assertEqual(unresolved_pairs([a, too_close]), [(0, 1)])
        self.assertEqual(unresolved_pairs([a, overlap]), [(0, 1)])

    def test_jitter_safe_guard_includes_filter_transition_and_edge_motion(self):
        bandwidth = 0.008
        self.assertAlmostEqual(
            required_guard_band_after_edge_jitter(bandwidth, bandwidth), 0.00575
        )
        a = SignalBox(0, 1000, -0.104, -0.096)
        just_too_close = SignalBox(0, 1000, -0.0904, -0.0824)
        safe = SignalBox(0, 1000, -0.09025, -0.08225)
        self.assertFalse(pair_remains_resolved_after_edge_jitter(a, just_too_close))
        self.assertTrue(pair_remains_resolved_after_edge_jitter(a, safe))

    def test_time_disjoint_boxes_need_no_frequency_guard(self):
        a = SignalBox(0, 100, -0.104, -0.096)
        b = SignalBox(100, 200, -0.101, -0.093)
        self.assertEqual(unresolved_pairs([a, b]), [])


    def test_close_pairs_have_explicit_unresolved_status(self):
        a = SignalBox(0, 1000, -0.104, -0.096)
        b = SignalBox(0, 1000, -0.100, -0.092)
        self.assertEqual(resolution_statuses([a, b]), ["unresolved", "unresolved"])
