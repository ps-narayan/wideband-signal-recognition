#!/usr/bin/env python3
"""Prepare one recording-disjoint SigMF source as paper-taxonomy YOLO data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wideband_recognition.sigmf_dataset import (
    convert_sigmf_recording,
    write_sigmf_dataset_yaml,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True, help="annotated .sigmf-meta file")
    parser.add_argument("--output", required=True, help="generated dataset directory")
    parser.add_argument("--recording-id", required=True, help="stable parent recording ID")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True,
                        help="assign all segments of this source recording to one split")
    parser.add_argument("--segment-samples", type=int, default=1_000_000)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    report = convert_sigmf_recording(
        args.meta,
        args.output,
        recording_id=args.recording_id,
        split=args.split,
        segment_samples=args.segment_samples,
        image_size=args.image_size,
    )
    yaml_path = write_sigmf_dataset_yaml(args.output, Path(args.output) / "dataset.yaml")
    print(json.dumps({**report, "dataset_yaml": str(yaml_path), "run_status": "prepared_only_not_trained"}, indent=2))


if __name__ == "__main__":
    main()
