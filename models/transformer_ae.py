"""Transformer Autoencoder — teacher model.

Tokenization uses the SAME k_groups / features_per_group as KitsunePyTorch
to improve PKT alignment between teacher and student.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features: int, k_groups: int, latent_dim: int = 64,
                 nhead: int = 4, dim_feedforward: int = 128, num_layers: int = 2):
        super().__init__()
        assert n_features % k_groups == 0
        self.k_groups         = k_groups
        self.features_per_group = n_features // k_groups
        self.latent_dim       = latent_dim
        d_model = latent_dim  # 64

        self.input_proj  = nn.Linear(self.features_per_group, d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(1, k_groups, d_model))

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               batch_first=True)
        self.encoder_tf = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.latent_proj = nn.Identity()  # mean-pool is the bottleneck

        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               batch_first=True)
        self.decoder_tf  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, self.features_per_group)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _tokenize(self, x: Tensor) -> Tensor:
        # x: (B, N)  →  (B, K, fpg)
        return x.view(x.size(0), self.k_groups, self.features_per_group)

    def encode(self, x: Tensor) -> Tensor:
        tokens = self._tokenize(x)                          # (B, K, fpg)
        tokens = self.input_proj(tokens) + self.pos_embed   # (B, K, d_model)
        enc    = self.encoder_tf(tokens)                    # (B, K, d_model)
        return enc.mean(dim=1)                              # (B, d_model)

    def encode_tokens(self, x: Tensor) -> Tensor:
        """Returns per-group token representations before mean-pool: (B, K, d_model).
        Token i corresponds to the same feature group as Kitsune sub-AE i."""
        tokens = self._tokenize(x)
        tokens = self.input_proj(tokens) + self.pos_embed
        return self.encoder_tf(tokens)                      # (B, K, d_model)

    def decode(self, z: Tensor) -> Tensor:
        # Repeat latent to K query tokens
        queries = z.unsqueeze(1).expand(-1, self.k_groups, -1)  # (B, K, d_model)
        queries = queries + self.pos_embed
        memory  = z.unsqueeze(1).expand(-1, self.k_groups, -1)  # (B, K, d_model)
        dec     = self.decoder_tf(queries, memory)               # (B, K, d_model)
        tokens  = self.output_proj(dec)                          # (B, K, fpg)
        return tokens.reshape(tokens.size(0), -1)                # (B, N)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z    = self.encode(x)
        recon = self.decode(z)
        return recon, z
