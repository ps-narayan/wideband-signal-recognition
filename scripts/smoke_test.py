"""Small CPU-only path check: synthesize, render, threshold, extract, and classify."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from wideband_recognition.baseline import energy_proposals
from wideband_recognition.extraction import extract_frames
from wideband_recognition.geometry import SignalBox
from wideband_recognition.pam_model import PAMCNN
from wideband_recognition.signals import SignalSpec, generate_recording
from wideband_recognition.spectrogram import log_spectrogram, resize_spectrogram


def main():
    import torch

    spec = SignalSpec(
        modulation="2-PSK", start=0, duration=8192, center_frequency=0.1,
        bandwidth=0.04, carrier_phase=0.7, symbol_timing=2, rolloff=0.35,
    )
    generated = generate_recording([spec], 8192, seed=42, global_snr_db=10.0)
    spectrum, covered = log_spectrogram(generated.iq)
    resized = resize_spectrogram(spectrum, 640, 640)
    box = SignalBox(0, 8192, 0.08, 0.12, "PAM")
    frames = extract_frames(generated.iq, box, n_frames=2, seed=11)
    model = PAMCNN().eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(frames.transpose(0, 2, 1)))
    with tempfile.TemporaryDirectory() as directory:
        image_path = Path(directory) / "smoke.png"
        pixels = np.clip((resized + 3) / 6 * 255, 0, 255).astype(np.uint8)
        Image.fromarray(pixels, mode="L").save(image_path)
        proposals = energy_proposals(image_path)
    report = {
        "status": "ok",
        "input_samples": int(generated.iq.size),
        "stft_covered_samples": int(covered),
        "resized_shape": list(resized.shape),
        "extracted_frame_shape": list(frames.shape),
        "classifier_logits_shape": list(logits.shape),
        "energy_proposal_count": len(proposals),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
