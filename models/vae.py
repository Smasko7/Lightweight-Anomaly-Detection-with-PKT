"""Variational Autoencoder teacher.

Same encoder/decoder backbone as VanillaAutoencoder for fair comparison.
The encoder trunk is shared; the bottleneck splits into mu and log_var heads.

forward() returns (recon, mu) — identical interface to other teachers so PKT
and evaluation code need no changes. mu is always used as the latent for PKT
(deterministic, no noise), which produces stable B×B similarity matrices.

During training use forward_with_kl() to also get log_var for the ELBO loss.
During eval, forward() decodes from mu directly (no reparameterization).
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class VAETeacher(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Shared encoder trunk — same as VanillaAutoencoder up to the bottleneck
        self.encoder_trunk = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 128),        nn.ReLU(), nn.BatchNorm1d(128),
        )
        self.mu_head      = nn.Linear(128, latent_dim)
        self.log_var_head = nn.Linear(128, latent_dim)

        # Decoder — identical to VanillaAutoencoder
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 256),        nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, n_features), nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (mu, log_var), each (B, latent_dim)."""
        h = self.encoder_trunk(x)
        return self.mu_head(h), self.log_var_head(h)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder_net(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (recon, mu).

        Training mode: decode from reparameterized z (stochastic).
        Eval mode:     decode from mu (deterministic) — used by PKT and evaluation.
        Always returns mu as the latent so callers (PKT, metrics) need no changes.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var) if self.training else mu
        return self.decode(z), mu

    def forward_with_kl(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (recon, mu, log_var) — for ELBO computation in the training loop."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
