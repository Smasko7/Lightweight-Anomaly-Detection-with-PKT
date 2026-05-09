"""Probabilistic Knowledge Transfer loss (Passalis & Tefas, ECCV 2018).

Operates on BxB pairwise similarity matrices — dimensionality-agnostic,
so teacher and student latent dims need not match. No projection layer.

Matches the official implementation at:
https://github.com/passalis/probabilistic_kt/blob/master/nn/pkt.py
"""
import torch
import torch.nn as nn
from torch import Tensor

_EPS = 1e-7


class PKTLoss(nn.Module):
    def forward(self, teacher_latent: Tensor, student_latent: Tensor) -> Tensor:
        """
        Args:
            teacher_latent: (B, D_t) — should be detached before calling
            student_latent: (B, D_s) — D_s != D_t is fine
        Returns:
            scalar KL divergence D_KL(P_teacher || P_student)
        """
        t = _cosine_probs(teacher_latent.detach())
        s = _cosine_probs(student_latent)
        return torch.mean(t * torch.log((t + _EPS) / (s + _EPS)))


def _cosine_probs(x: Tensor) -> Tensor:
    """L2-normalize, compute cosine similarity, scale to [0,1], row-normalize."""
    norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    x = x / (norm + _EPS)
    x[x != x] = 0  # zero out NaNs
    sim = torch.mm(x, x.t())
    sim = (sim + 1.0) / 2.0                                    # scale to [0, 1]
    sim = sim / (torch.sum(sim, dim=1, keepdim=True) + _EPS)   # row-normalize
    return sim
