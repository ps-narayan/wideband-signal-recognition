"""Small raw-I/Q frame classifier; torch is imported only when this module is used."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised in no-torch installations
    raise ImportError(
        "The PAM classifier requires the pinned PyTorch dependencies in requirements-colab.txt"
    ) from exc


PAM_LABELS = ("2-PSK", "4-PSK", "8-PSK", "16-QAM", "64-QAM", "256-QAM", "Others")


class PAMCNN(nn.Module):
    """Compact 1-D CNN for tensors shaped [batch, 2, 1024]."""

    def __init__(self, classes: int = len(PAM_LABELS)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, classes),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(frames))
