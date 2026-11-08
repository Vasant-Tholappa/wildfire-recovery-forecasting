"""Compact regressors for frozen two-channel 15x15 NBR inputs."""
import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, conv_widths=(16, 32), embedding_dim=64):
        super().__init__()
        c1, c2 = conv_widths
        self.layers = nn.Sequential(
            nn.Conv2d(2, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(c2 * 3 * 3, embedding_dim), nn.ReLU(),
        )

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (2, 15, 15):
            raise ValueError('Expected images [batch,2,15,15]')
        return self.layers(images)


class CNNRegressor(nn.Module):
    def __init__(self, model_type='cnn_only', *, conv_widths=(16, 32), embedding_dim=64,
                 head_dim=32, dropout=0.0, tabular_dim=32, tabular_dropout=0.0):
        super().__init__()
        if model_type not in ('cnn_only', 'fusion'):
            raise ValueError('Unknown model type')
        if (len(conv_widths) != 2 or any(int(v) != v or v < 1 for v in
                (*conv_widths, embedding_dim, head_dim, tabular_dim))
                or not 0 <= dropout < 1 or not 0 <= tabular_dropout < 1):
            raise ValueError('Invalid architecture settings')
        self.configuration = dict(model_type=model_type, conv_widths=list(conv_widths),
                                  embedding_dim=embedding_dim, head_dim=head_dim, dropout=dropout,
                                  tabular_dim=tabular_dim, tabular_dropout=tabular_dropout)
        self.model_type = model_type
        self.encoder = ImageEncoder(conv_widths, embedding_dim)
        self.image_dropout = nn.Dropout(dropout)
        self.tabular = nn.Sequential(nn.Linear(70, tabular_dim), nn.ReLU(),
                                    nn.Dropout(tabular_dropout)) if model_type == 'fusion' else None
        # Parameter-bearing indices remain compatible with original default checkpoints.
        self.head = nn.Sequential(nn.Linear(embedding_dim + (tabular_dim if self.tabular is not None else 0), head_dim),
                                  nn.ReLU(), nn.Linear(head_dim, 1))

    def forward(self, images, tabular=None):
        features = self.image_dropout(self.encoder(images))
        if self.tabular is not None:
            if tabular is None or tabular.shape != (len(images), 70):
                raise ValueError('Expected fusion tabular [batch,70]')
            features = torch.cat((features, self.tabular(tabular)), dim=1)
        return self.head(features).squeeze(-1)  # Unbounded scalar regression; no target clipping.
