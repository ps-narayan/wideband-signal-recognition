"""Scaled wideband signal-recognition project."""

from .geometry import SignalBox, box_to_pixels, pixels_to_box, to_yolo
from .layout import resolution_statuses
from .signals import (
    GeneratedRecording,
    SignalSpec,
    generate_recording,
    make_visibility_specs,
    save_generator_metadata,
)

__all__ = [
    "GeneratedRecording",
    "SignalBox",
    "SignalSpec",
    "box_to_pixels",
    "generate_recording",
    "make_visibility_specs",
    "save_generator_metadata",
    "pixels_to_box",
    "resolution_statuses",
    "to_yolo",
]
