"""Compact regressors for frozen two-channel 15x15 NBR inputs."""
import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 3 * 3, 64), nn.ReLU(),
        )

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (2, 15, 15):
            raise ValueError('Expected images [batch,2,15,15]')
        return self.layers(images)


class CNNRegressor(nn.Module):
    def __init__(self, model_type='cnn_only'):
        super().__init__()
        if model_type not in ('cnn_only', 'fusion'):
            raise ValueError('Unknown model type')
        self.model_type = model_type
        self.encoder = ImageEncoder()
        self.tabular = nn.Sequential(nn.Linear(70, 32), nn.ReLU()) if model_type == 'fusion' else None
        self.head = nn.Sequential(nn.Linear(96 if self.tabular is not None else 64, 32),
                                  nn.ReLU(), nn.Linear(32, 1))

    def forward(self, images, tabular=None):
        features = self.encoder(images)
        if self.tabular is not None:
            if tabular is None or tabular.shape != (len(images), 70):
                raise ValueError('Expected fusion tabular [batch,70]')
            features = torch.cat((features, self.tabular(tabular)), dim=1)
        return self.head(features).squeeze(-1)  # Unbounded scalar regression; no target clipping.
