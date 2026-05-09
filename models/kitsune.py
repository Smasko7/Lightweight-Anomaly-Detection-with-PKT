"""PyTorch Kitsune — faithful reimplementation of KitNET-py for KD training."""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, List


class SubAutoencoder(nn.Module):
    """Single-hidden-layer FC autoencoder for one feature group."""

    def __init__(self, n_visible: int, hidden_ratio: float = 0.75):
        super().__init__()
        n_hidden = max(1, int(n_visible * hidden_ratio))
        self.encoder = nn.Linear(n_visible, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_visible)
        self.n_hidden = n_hidden

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        hidden = torch.relu(self.encoder(x))
        recon  = torch.sigmoid(self.decoder(hidden))  # input is MinMax-scaled [0,1]
        return recon, hidden


class OutputAutoencoder(nn.Module):
    """Takes K normalized RMSE scores → final anomaly score via reconstruction RMSE."""

    def __init__(self, k: int, hidden_ratio: float = 0.75):
        super().__init__()
        n_hidden = max(1, int(k * hidden_ratio))
        self.encoder = nn.Linear(k, n_hidden)
        self.decoder = nn.Linear(n_hidden, k)

    def forward(self, rmses: Tensor) -> Tuple[Tensor, Tensor]:
        hidden = torch.relu(self.encoder(rmses))
        recon  = self.decoder(hidden)
        score  = torch.sqrt(torch.mean((rmses - recon) ** 2, dim=1))
        return score, recon


class KitsunePyTorch(nn.Module):
    """
    Kitsune ensemble of sub-autoencoders.

    Feature grouping: equal-size sequential splits (deterministic, no corClust).
    Pseudo-latent for PKT: concatenation of all sub-AE hidden activations.
    No projection layer — PKT is dimensionality-agnostic.

    RMSE normalization: per-component running min-max, matching original dA.train()
    normalization in KitNET-py. Tracked as non-parameter buffers; updated only
    during training (self.training=True).
    """

    def __init__(self, n_features: int, k_groups: int, hidden_ratio: float = 0.75):
        super().__init__()
        assert n_features % k_groups == 0, "n_features must be divisible by k_groups"
        self.n_features = n_features
        self.k_groups   = k_groups
        self.fpg        = n_features // k_groups  # features per group

        self.sub_aes = nn.ModuleList([
            SubAutoencoder(self.fpg, hidden_ratio) for _ in range(k_groups)
        ])
        self.output_ae = OutputAutoencoder(k_groups, hidden_ratio)

        self.pseudo_latent_dim = sum(ae.n_hidden for ae in self.sub_aes)

        # Running per-component min/max of the RMSE vector fed into output AE.
        # Matches dA.norm_min / dA.norm_max in the original KitNET-py.
        self.register_buffer('rmse_norm_min', torch.full((k_groups,),  float('inf')))
        self.register_buffer('rmse_norm_max', torch.full((k_groups,), float('-inf')))

    def _split(self, x: Tensor) -> List[Tensor]:
        return x.split(self.fpg, dim=1)

    def _normalize_rmse(self, rmse_vec: Tensor) -> Tensor:
        """Per-component min-max normalization of the RMSE vector.

        During training, updates running min/max from the current batch.
        At inference, uses the stored stats from training (self.training=False).
        Matches dA.train() normalization: x = (x - norm_min) / (norm_max - norm_min).
        """
        if self.training:
            batch_min = rmse_vec.detach().min(dim=0).values
            batch_max = rmse_vec.detach().max(dim=0).values
            self.rmse_norm_min = torch.minimum(self.rmse_norm_min, batch_min)
            self.rmse_norm_max = torch.maximum(self.rmse_norm_max, batch_max)
        return (rmse_vec - self.rmse_norm_min) / (self.rmse_norm_max - self.rmse_norm_min + 1e-16)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            anomaly_score: (batch,) RMSE-based score from output AE
            pseudo_latent: (batch, pseudo_latent_dim) concat of hidden activations
        """
        groups = self._split(x)
        rmses, hiddens = [], []
        for group, ae in zip(groups, self.sub_aes):
            recon, hidden = ae(group)
            rmse = torch.sqrt(torch.mean((group - recon) ** 2, dim=1, keepdim=True))
            rmses.append(rmse)
            hiddens.append(hidden)

        rmse_vec      = torch.cat(rmses, dim=1)             # (B, K)
        rmse_norm     = self._normalize_rmse(rmse_vec)      # (B, K), values in [0,1]
        pseudo_latent = torch.cat(hiddens, dim=1)           # (B, pseudo_latent_dim)
        anomaly_score, _ = self.output_ae(rmse_norm)        # (B,)
        return anomaly_score, pseudo_latent

    def get_anomaly_score(self, x: Tensor) -> Tensor:
        """Inference only."""
        with torch.no_grad():
            score, _ = self.forward(x)
        return score
