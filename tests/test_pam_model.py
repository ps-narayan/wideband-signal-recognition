import unittest

try:
    import torch
except ImportError:  # the full suite installs the pinned CPU build
    torch = None

if torch is not None:
    from wideband_recognition.pam_model import PAMCNN
else:
    PAMCNN = None


@unittest.skipIf(torch is None, "PyTorch is installed by the ML/CI requirements")
class PAMModelTests(unittest.TestCase):
    def test_model_maps_iq_frame_to_seven_logits(self):
        model = PAMCNN()
        logits = model(torch.zeros(3, 2, 1024))
        self.assertEqual(tuple(logits.shape), (3, 7))
