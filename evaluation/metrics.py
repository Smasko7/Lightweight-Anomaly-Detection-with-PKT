"""Evaluation metrics: AUC-ROC, AUC-PR, F1, params, inference time."""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    model_type: str = 'kitsune',  # 'kitsune' | 'bigkit' | 'vanilla' | 'cnn' | 'transformer' | 'vae'
    phi: Optional[float] = None,  # paper III-E threshold: alert if score >= phi
) -> dict:
    device = next(model.parameters()).device
    model.eval()

    all_scores, all_labels = [], []
    total_time, total_samples = 0.0, 0

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1]
            n = x.size(0)

            t0 = time.perf_counter()
            if model_type == 'kitsune':
                scores = model.get_anomaly_score(x)
            elif model_type == 'bigkit':
                # Standalone bigkit teacher: output AE is untrained → direct RMSE sum
                scores = model.direct_rmse_sum(x)
            elif model_type == 'cnn':
                recon, _ = model(x.unsqueeze(1))
                scores = torch.mean((x - recon.squeeze(1)) ** 2, dim=1)
            elif model_type == 'vae':
                # ELBO-based score: reconstruction MSE (from mu) + KL divergence
                recon, mu = model(x)                          # eval: decode from mu
                _, log_var = model.encode(x)
                kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
                scores = torch.mean((x - recon) ** 2, dim=1) + kl
            else:
                recon, _ = model(x)
                scores = torch.mean((x - recon) ** 2, dim=1)
            elapsed = time.perf_counter() - t0

            all_scores.append(scores.cpu().numpy())
            all_labels.append(y.numpy())
            total_time    += elapsed
            total_samples += n

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    auc_roc = roc_auc_score(labels, scores)
    auc_pr  = average_precision_score(labels, scores)

    precision, recall, _ = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    f1_oracle = float(f1_scores.max())

    # F1 at paper threshold phi (no label leakage — phi comes from train set)
    f1_phi = None
    if phi is not None:
        preds = (scores >= phi).astype(int)
        f1_phi = float(f1_score(labels, preds, zero_division=0))

    params = sum(p.numel() for p in model.parameters())
    inference_time_ms = (total_time / total_samples) * 1000

    result = {
        'auc_roc':           float(auc_roc),
        'auc_pr':            float(auc_pr),
        'f1_oracle':         f1_oracle,   # best possible F1 (uses test labels to pick threshold)
        'params':            params,
        'inference_time_ms': inference_time_ms,
    }
    if phi is not None:
        result['phi']     = phi
        result['f1_phi']  = f1_phi        # F1 at paper threshold (deployment-realistic)
    return result
