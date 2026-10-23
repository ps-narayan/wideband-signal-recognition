import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from wideband_recognition.baseline import energy_proposals


class BaselineTests(unittest.TestCase):
    def test_energy_detector_returns_geometry_for_bright_patch(self):
        image = np.zeros((64, 64), dtype=np.uint8)
        image[20:30, 10:40] = 255
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "spectrogram.png"
            Image.fromarray(image, mode="L").save(path)
            proposals = energy_proposals(path, mask_threshold=1.0)
        self.assertTrue(any(proposal["xyxy"][2] > proposal["xyxy"][0] for proposal in proposals))
