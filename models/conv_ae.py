"""1D Convolutional Autoencoder — CNN teacher model.

Uses AdaptiveAvgPool1d so the same code works for n_features=64, 100, 400.
Input shape: (batch, 1, n_features) — caller must add the channel dimension.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class ConvAutoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = 64):
        super().__init__()
        self.n_features = n_features

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1,   32,  kernel_size=5, stride=1, padding=2), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32,  64,  kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,  128, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.encoder_fc = nn.Linear(128 * 4, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 128 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64,  kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.ConvTranspose1d(64,  32,  kernel_size=5, stride=2, padding=2, output_padding=1), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.ConvTranspose1d(32,  1,   kernel_size=5, stride=1, padding=2),
            nn.AdaptiveAvgPool1d(n_features),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> Tensor:
        # x: (B, 1, N)
        h = self.encoder_conv(x).flatten(1)
        return self.encoder_fc(h)

    def decode(self, z: Tensor) -> Tensor:
        h = self.decoder_fc(z).view(-1, 128, 4)
        return self.decoder_conv(h)  # (B, 1, n_features)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z    = self.encode(x)
        recon = self.decode(z)
        return recon, z
