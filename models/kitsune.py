"""PyTorch Kitsune — faithful reimplementation of KitNET-py for KD training."""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, List, Optional


class SubAutoencoder(nn.Module):
    """Single-hidden-layer FC autoencoder for one feature group."""

    def __init__(self, n_visible: int, hidden_ratio: float = 0.75):
        super().__init__()
        n_hidden = max(1, int(n_visible * hidden_ratio))
        self.encoder = nn.Linear(n_visible, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_visible)
        self.n_hidden = n_hidden

    def encode(self, x: Tensor) -> Tensor:
        return torch.relu(self.encoder(x))

    def decode(self, h: Tensor) -> Tensor:
        return torch.sigmoid(self.decoder(h))  # input is MinMax-scaled [0,1]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encode(x)
        return self.decode(h), h


class CrossGroupMixer(nn.Module):
    """Token-mixing MLP across K groups, MLP-Mixer style (Tolstikhin et al. 2021).

    The K per-group hiddens are treated as K 'tokens' of width d = pseudo_latent_dim/K.
    The mixer applies a residual MLP on the K dimension only — each of the d feature
    channels is mixed across groups independently. Parameter cost is O(K²), so it
    stays bounded as the sub-AEs widen (hidden_ratio grows).

    Gives the otherwise-independent KitNet sub-AEs a single point of cross-group
    interaction, mirroring (in spirit) the global structure CNN/Transformer
    teachers provide via convolution/attention.
    """

    def __init__(self, k_groups: int, expansion: int = 4):
        super().__init__()
        self.k_groups = k_groups
        self.net = nn.Sequential(
            nn.Linear(k_groups, k_groups * expansion),
            nn.GELU(),
            nn.Linear(k_groups * expansion, k_groups),
        )

    def forward(self, concat_hiddens: Tensor) -> Tensor:
        # (B, K*d) → (B, K, d) → (B, d, K) → residual MLP over K → (B, K, d) → (B, K*d)
        B, total = concat_hiddens.shape
        d = total // self.k_groups
        x = concat_hiddens.view(B, self.k_groups, d).transpose(1, 2)  # (B, d, K)
        x = x + self.net(x)
        return x.transpose(1, 2).contiguous().view(B, total)


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

    def __init__(self, n_features: int, k_groups: int, hidden_ratio: float = 0.75,
                 mixer_type: Optional[str] = None):
        super().__init__()
        assert n_features % k_groups == 0, "n_features must be divisible by k_groups"
        self.n_features = n_features
        self.k_groups   = k_groups
        self.fpg        = n_features // k_groups  # features per group
        self.mixer_type = mixer_type

        self.sub_aes = nn.ModuleList([
            SubAutoencoder(self.fpg, hidden_ratio) for _ in range(k_groups)
        ])
        self.output_ae = OutputAutoencoder(k_groups, hidden_ratio)

        self.pseudo_latent_dim = sum(ae.n_hidden for ae in self.sub_aes)
        self._hidden_sizes = [ae.n_hidden for ae in self.sub_aes]

        if mixer_type is None:
            self.mixer = None
        elif mixer_type == 'mlp':
            self.mixer = CrossGroupMixer(k_groups, expansion=4)
        else:
            raise ValueError(f"Unknown mixer_type: {mixer_type}")

        # Running per-component min/max of the RMSE vector fed into output AE.
        # Matches dA.norm_min / dA.norm_max in the original KitNET-py.
        self.register_buffer('rmse_norm_min', torch.full((k_groups,),  float('inf')))
        self.register_buffer('rmse_norm_max', torch.full((k_groups,), float('-inf')))

    def _split(self, x: Tensor) -> List[Tensor]:
        return x.split(self.fpg, dim=1)

    def _apply_mixer(self, hiddens: List[Tensor]) -> List[Tensor]:
        """If mixer is configured, mix the K hiddens and re-split. Otherwise no-op."""
        if self.mixer is None:
            return hiddens
        mixed = self.mixer(torch.cat(hiddens, dim=1))
        return list(mixed.split(self._hidden_sizes, dim=1))

    def _encode_groups(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Run per-group encoders. Returns (groups, hiddens_post_mixer)."""
        groups  = self._split(x)
        hiddens = [ae.encode(g) for g, ae in zip(groups, self.sub_aes)]
        hiddens = self._apply_mixer(hiddens)
        return list(groups), hiddens

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
            pseudo_latent: (batch, pseudo_latent_dim) concat of (post-mixer) hidden activations
        """
        groups, hiddens = self._encode_groups(x)
        rmses = []
        for g, h, ae in zip(groups, hiddens, self.sub_aes):
            recon = ae.decode(h)
            rmses.append(torch.sqrt(torch.mean((g - recon) ** 2, dim=1, keepdim=True)))

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

    def encode_per_subae(self, x: Tensor) -> List[Tensor]:
        """List of K per-group (post-mixer) hidden activations. Used by PKT teacher path."""
        _, hiddens = self._encode_groups(x)
        return hiddens

    def direct_rmse_sum(self, x: Tensor) -> Tensor:
        """Anomaly score = sum of per-group RMSE. Bypasses output AE entirely.
        Used to score a bigkit teacher (no trained output AE) and as a diagnostic.
        Uses post-mixer hiddens for decoding when mixer is present.
        """
        groups, hiddens = self._encode_groups(x)
        rmses = [
            torch.sqrt(torch.mean((g - ae.decode(h)) ** 2, dim=1))
            for g, h, ae in zip(groups, hiddens, self.sub_aes)
        ]
        return torch.stack(rmses, dim=1).sum(dim=1)
