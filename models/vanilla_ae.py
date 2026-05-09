"""Vanilla FC Autoencoder — baseline teacher model."""
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class VanillaAutoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = 64):
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 128),        nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, latent_dim),
        )
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 256),        nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, n_features), nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder_net(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder_net(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        return self.decode(z), z
