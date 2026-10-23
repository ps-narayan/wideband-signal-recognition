import unittest

import numpy as np

from wideband_recognition.spectrogram import covered_sample_count, log_spectrogram, resize_spectrogram


class SpectrogramTests(unittest.TestCase):
    def test_negative_frequency_is_near_bottom_of_image_axis(self):
        samples = np.arange(8192)
        iq = np.exp(2j * np.pi * -0.25 * samples)
        spectrogram, _ = log_spectrogram(iq)
        image = resize_spectrogram(spectrogram)
        peak_row = int(np.argmax(image.mean(axis=1)))
        self.assertGreater(peak_row, 450)
        self.assertLess(peak_row, 520)

    def test_time_is_horizontal_image_axis(self):
        samples = np.arange(8192)
        iq = np.zeros(8192, dtype=np.complex128)
        iq[:4096] = np.exp(2j * np.pi * 0.25 * samples[:4096])
        spectrogram, _ = log_spectrogram(iq)
        image = resize_spectrogram(spectrogram)
        self.assertGreater(float(image[:, :300].mean()), float(image[:, 340:].mean()))

    def test_one_million_sample_coverage_drops_only_final_64(self):
        self.assertEqual(covered_sample_count(1_000_000, 512), 999_936)
        self.assertEqual(1_000_000 - covered_sample_count(1_000_000, 512), 64)

    def test_stft_uses_full_shifted_fft_and_normalizes(self):
        rng = np.random.default_rng(1)
        iq = (rng.standard_normal(8192) + 1j * rng.standard_normal(8192)).astype(np.complex64)
        spectrogram, covered = log_spectrogram(iq)
        self.assertEqual(spectrogram.shape, (16, 4096))
        self.assertEqual(covered, 8192)
        self.assertAlmostEqual(float(spectrogram.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(spectrogram.std()), 1.0, places=5)
        self.assertEqual(resize_spectrogram(spectrogram).shape, (640, 640))
