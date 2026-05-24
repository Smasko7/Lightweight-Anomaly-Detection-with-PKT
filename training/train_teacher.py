"""Train teacher model (vanilla, CNN, or Transformer) on normal data only."""
import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path
from tqdm import tqdm

from data.loader import load_dataset
from models.vanilla_ae import VanillaAutoencoder
from models.conv_ae import ConvAutoencoder
from models.transformer_ae import TransformerAutoencoder
from models.vae import VAETeacher
from models.kitsune import KitsunePyTorch
from evaluation.metrics import evaluate
from training.utils import set_seed, get_device


def _elbo(recon: torch.Tensor, x: torch.Tensor,
          mu: torch.Tensor, log_var: torch.Tensor, beta: float) -> torch.Tensor:
    """ELBO = MSE reconstruction + beta * KL(N(mu,sigma^2) || N(0,I)).
    KL per sample = -0.5 * sum(1 + log_var - mu^2 - exp(log_var)).
    """
    recon_loss = nn.functional.mse_loss(recon, x)
    kl = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
    return recon_loss + beta * kl


def _bigkit_loss(model: KitsunePyTorch, x: torch.Tensor) -> torch.Tensor:
    """Sum of per-group MSE across all sub-AEs. Output AE is not involved."""
    groups = x.split(model.fpg, dim=1)
    return sum(nn.functional.mse_loss(ae(g)[0], g) for g, ae in zip(groups, model.sub_aes))


def train_teacher(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    save_path: str = 'checkpoints/teacher.pt',
    model_type: str = 'vanilla',
    beta: float = 1.0,
) -> dict:
    device    = get_device()
    model     = model.to(device)
    # bigkit teacher trains only its sub-AEs; output AE stays at init and is unused.
    params    = model.sub_aes.parameters() if model_type == 'bigkit' else model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)
    mse       = nn.MSELoss()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    best_val, no_improve, history = float('inf'), 0, {'train': [], 'val': []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (x,) in train_loader:
            x = x.to(device)
            if model_type == 'cnn':
                recon, _ = model(x.unsqueeze(1))
                loss = mse(recon.squeeze(1), x)
            elif model_type == 'vae':
                recon, mu, log_var = model.forward_with_kl(x)
                loss = _elbo(recon, x, mu, log_var, beta)
            elif model_type == 'bigkit':
                loss = _bigkit_loss(model, x)
            else:
                recon, _ = model(x)
                loss = mse(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                if model_type == 'cnn':
                    recon, _ = model(x.unsqueeze(1))
                    loss = mse(recon.squeeze(1), x)
                elif model_type == 'vae':
                    recon, mu, log_var = model.forward_with_kl(x)
                    loss = _elbo(recon, x, mu, log_var, beta)
                elif model_type == 'bigkit':
                    loss = _bigkit_loss(model, x)
                else:
                    recon, _ = model(x)
                    loss = mse(recon, x)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={train_loss:.5f} | val={val_loss:.5f}")

    model.load_state_dict(torch.load(save_path, map_location=device))

    # Compute phi = max anomaly score on training set (naive threshold)
    model.eval()
    phi = 0.0
    with torch.no_grad():
        for (x,) in train_loader:
            x = x.to(device)
            if model_type == 'cnn':
                recon, _ = model(x.unsqueeze(1))
                scores = torch.mean((x - recon.squeeze(1)) ** 2, dim=1)
            elif model_type == 'vae':
                recon, mu = model(x)                          # eval: decode from mu
                _, log_var = model.encode(x)
                kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
                scores = torch.mean((x - recon) ** 2, dim=1) + kl
            elif model_type == 'bigkit':
                scores = model.direct_rmse_sum(x)
            else:
                recon, _ = model(x)
                scores = torch.mean((x - recon) ** 2, dim=1)
            phi = max(phi, scores.max().item())
    torch.save({'state_dict': model.state_dict(), 'phi': phi}, save_path)
    print(f"  phi (max train score) = {phi:.5f}")

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['optdigits', 'mnist', 'landsat', 'backdoor'])
    parser.add_argument('--model',   required=True, choices=['vanilla', 'cnn', 'transformer', 'vae', 'bigkit'])
    parser.add_argument('--beta',    type=float, default=1.0,
                        help='KL weight for VAE ELBO loss (default 1.0)')
    parser.add_argument('--hidden_ratio', type=float, default=4.0,
                        help='Sub-AE hidden ratio for bigkit teacher (ignored for other models)')
    parser.add_argument('--mixer', choices=['none', 'mlp'], default='none',
                        help='Cross-group mixer for bigkit teacher (default none; "mlp" adds a residual MLP mixer)')
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--epochs',  type=int, default=50)
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    set_seed(args.seed)
    train_loader, val_loader, test_loader, cfg = load_dataset(
        args.dataset, batch_size=args.batch_size, seed=args.seed)

    n  = cfg['n_features']
    kg = cfg['k_groups']

    if args.model == 'vanilla':
        model = VanillaAutoencoder(n_features=n)
    elif args.model == 'cnn':
        model = ConvAutoencoder(n_features=n)
    elif args.model == 'vae':
        model = VAETeacher(n_features=n)
    elif args.model == 'bigkit':
        mixer_type = None if args.mixer == 'none' else args.mixer
        model = KitsunePyTorch(n_features=n, k_groups=kg,
                               hidden_ratio=args.hidden_ratio, mixer_type=mixer_type)
    else:
        model = TransformerAutoencoder(n_features=n, k_groups=kg)

    if args.model == 'bigkit':
        mix_tag  = '_mix' if args.mixer != 'none' else ''
        model_id = f"bigkit{mix_tag}_r{int(args.hidden_ratio)}"
    else:
        model_id = args.model
    save_path = f"checkpoints/{args.dataset}_{model_id}_seed{args.seed}.pt"
    if args.model == 'vae':
        tag = f" (beta={args.beta})"
    elif args.model == 'bigkit':
        tag = f" (hidden_ratio={args.hidden_ratio}, mixer={args.mixer})"
    else:
        tag = ""
    print(f"Training {args.model} teacher{tag} on {args.dataset} (seed={args.seed})")
    history = train_teacher(model, train_loader, val_loader,
                            epochs=args.epochs, lr=args.lr,
                            save_path=save_path, model_type=args.model,
                            beta=args.beta)
    print(f"Best val loss: {min(history['val']):.5f}")

    ckpt_data = torch.load(save_path, map_location='cpu')
    phi = ckpt_data['phi']

    print(f"Evaluating on test set...")
    metrics = evaluate(model, test_loader, model_type=args.model, phi=phi)
    metrics.update({'dataset': args.dataset, 'model': model_id, 'seed': args.seed})
    print(f"  AUC-ROC={metrics['auc_roc']:.4f} | AUC-PR={metrics['auc_pr']:.4f} | "
          f"F1_oracle={metrics['f1_oracle']:.4f} | F1_phi={metrics['f1_phi']:.4f} | "
          f"phi={phi:.5f} | params={metrics['params']}")

    Path('results').mkdir(exist_ok=True)
    out = f"results/{args.dataset}_{model_id}_seed{args.seed}.json"
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved to {out}")


if __name__ == '__main__':
    main()
