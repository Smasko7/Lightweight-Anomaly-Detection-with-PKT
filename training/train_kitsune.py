"""Train Kitsune student — vanilla (no KD) or with PKT knowledge distillation.

Two-phase training:
  Phase 1 — sub-AEs only (+ PKT if teacher provided). Output AE frozen.
  Phase 2 — output AE only, sub-AEs frozen. RMSE norm buffers are reset at
             the start so they calibrate to the now-stable sub-AE outputs.

This prevents the output AE from training on a moving RMSE distribution
(caused by PKT continuously reshaping sub-AE representations), which was the
root cause of PKT improving sub-AE reconstruction quality but the output AE
then destroying the signal.
"""
import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path

from data.loader import load_dataset
from models.kitsune import KitsunePyTorch
from models.vanilla_ae import VanillaAutoencoder
from models.conv_ae import ConvAutoencoder
from models.transformer_ae import TransformerAutoencoder
from models.vae import VAETeacher
from losses.pkt import PKTLoss
from evaluation.metrics import evaluate
from training.utils import set_seed, get_device


def _get_rmse_vec(kitsune: KitsunePyTorch, x: torch.Tensor):
    """Compute per-group RMSE vector and concat hiddens."""
    groups = x.split(kitsune.fpg, dim=1)
    rmses, hiddens, recon_losses = [], [], []
    for group, ae in zip(groups, kitsune.sub_aes):
        recon, hidden = ae(group)
        rmses.append(torch.sqrt(torch.mean((group - recon) ** 2, dim=1, keepdim=True)))
        hiddens.append(hidden)
        recon_losses.append(nn.functional.mse_loss(recon, group))
    loss_L1  = sum(recon_losses)
    rmse_vec = torch.cat(rmses, dim=1)   # (B, K)
    return loss_L1, rmse_vec, hiddens


def _compute_pkt_term(pkt_fn, teacher, teacher_type, pkt_mode, x, hiddens, recon_loss, alpha):
    """Compute scaled PKT loss for any mode. Teacher forward runs under no_grad."""
    x_in = x.unsqueeze(1) if teacher_type == 'cnn' else x

    if teacher_type == 'bigkit':
        with torch.no_grad():
            t_hiddens = teacher.encode_per_subae(x_in)
            t_concat  = torch.cat(t_hiddens, dim=1).detach()
            t_hiddens = [h.detach() for h in t_hiddens]
        if pkt_mode == 'concat':
            pkt_raw = pkt_fn(t_concat, torch.cat(hiddens, dim=1))
        elif pkt_mode == 'per_subae':
            pkt_raw = sum(pkt_fn(t_concat, h) for h in hiddens)
        else:  # subae_paired
            pkt_raw = sum(pkt_fn(t_h, s_h) for t_h, s_h in zip(t_hiddens, hiddens))
    else:
        with torch.no_grad():
            if pkt_mode in ('concat', 'per_subae'):
                _, teacher_latent = teacher(x_in)
                teacher_latent = teacher_latent.detach()
            else:  # token
                token_latents = teacher.encode_tokens(x_in).detach()  # (B, K, d_model)

        if pkt_mode == 'concat':
            pseudo_latent = torch.cat(hiddens, dim=1)
            pkt_raw = pkt_fn(teacher_latent, pseudo_latent)

        elif pkt_mode == 'per_subae':
            pkt_raw = sum(pkt_fn(teacher_latent, h) for h in hiddens)

        else:  # token
            pkt_raw = sum(pkt_fn(token_latents[:, i, :], h) for i, h in enumerate(hiddens))

    pkt_scale = recon_loss.detach() / (pkt_raw.detach() + 1e-8)
    return alpha * pkt_scale * pkt_raw


def _run_phase1(
    kitsune, train_loader, val_loader, teacher, teacher_type, pkt_mode,
    alpha, epochs, lr, patience, device, pkt_fn,
) -> dict:
    """Phase 1: train sub-AEs (+ PKT). Output AE is not updated."""
    optimizer = torch.optim.Adam(kitsune.sub_aes.parameters(), lr=lr)
    best_val, no_improve, history = float('inf'), 0, {'train': [], 'val': []}
    best_state = None
    _diag_done = False

    for epoch in range(1, epochs + 1):
        kitsune.train()
        train_loss = 0.0
        for (x,) in train_loader:
            x = x.to(device)
            loss_L1, rmse_vec, hiddens = _get_rmse_vec(kitsune, x)
            # Update RMSE norm stats (side-effect of _normalize_rmse in train mode)
            kitsune._normalize_rmse(rmse_vec)
            loss = loss_L1

            if teacher is not None:
                pkt_term = _compute_pkt_term(
                    pkt_fn, teacher, teacher_type, pkt_mode, x, hiddens, loss_L1, alpha)
                if not _diag_done:
                    print(f"  [diag p1] L1={loss_L1.item():.5f} | PKT(scaled)={pkt_term.item():.5f}")
                    _diag_done = True
                loss = loss + pkt_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        kitsune.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                loss_L1, rmse_vec, hiddens = _get_rmse_vec(kitsune, x)
                loss = loss_L1
                if teacher is not None:
                    pkt_term = _compute_pkt_term(
                        pkt_fn, teacher, teacher_type, pkt_mode, x, hiddens, loss_L1, alpha)
                    loss = loss + pkt_term
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.clone() for k, v in kitsune.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [Phase 1] Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Phase 1] Epoch {epoch:3d} | train={train_loss:.5f} | val={val_loss:.5f}")

    # Restore best sub-AE weights (output AE weights in best_state are whatever,
    # they will be re-trained in phase 2)
    kitsune.load_state_dict(best_state)
    return history


def _run_phase2(
    kitsune, train_loader, val_loader, epochs, lr, patience, device,
) -> dict:
    """Phase 2: train output AE only. Sub-AEs frozen. RMSE norm buffers reset first."""
    # Reset norm buffers so they re-calibrate on stable sub-AE outputs
    kitsune.rmse_norm_min.fill_(float('inf'))
    kitsune.rmse_norm_max.fill_(float('-inf'))

    optimizer = torch.optim.Adam(kitsune.output_ae.parameters(), lr=lr)
    best_val, no_improve, history = float('inf'), 0, {'train': [], 'val': []}
    best_outae_state = None

    for epoch in range(1, epochs + 1):
        kitsune.train()
        train_loss = 0.0
        for (x,) in train_loader:
            x = x.to(device)
            with torch.no_grad():
                # sub-AEs frozen — run without grad to save memory
                _, rmse_vec, _ = _get_rmse_vec(kitsune, x)
            rmse_norm = kitsune._normalize_rmse(rmse_vec)  # updates buffers
            _, rmse_recon = kitsune.output_ae(rmse_norm)
            loss_L2 = nn.functional.mse_loss(rmse_recon, rmse_norm.detach())

            optimizer.zero_grad()
            loss_L2.backward()
            optimizer.step()
            train_loss += loss_L2.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        kitsune.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                _, rmse_vec, _ = _get_rmse_vec(kitsune, x)
                rmse_norm = kitsune._normalize_rmse(rmse_vec)
                _, rmse_recon = kitsune.output_ae(rmse_norm)
                loss_L2 = nn.functional.mse_loss(rmse_recon, rmse_norm.detach())
                val_loss += loss_L2.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            best_outae_state = {k: v.clone() for k, v in kitsune.output_ae.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [Phase 2] Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Phase 2] Epoch {epoch:3d} | train={train_loss:.5f} | val={val_loss:.5f}")

    kitsune.output_ae.load_state_dict(best_outae_state)
    return history


def train_kitsune(
    kitsune: KitsunePyTorch,
    train_loader: DataLoader,
    val_loader: DataLoader,
    teacher: Optional[nn.Module] = None,
    teacher_type: str = 'vanilla',
    pkt_mode: str = 'concat',
    alpha: float = 0.5,
    epochs: int = 50,
    phase2_epochs: int = 25,
    lr: float = 1e-3,
    patience: int = 10,
    save_path: str = 'checkpoints/kitsune.pt',
) -> dict:
    device  = get_device()
    kitsune = kitsune.to(device)
    pkt     = PKTLoss()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    if teacher is not None:
        teacher = teacher.to(device)
        teacher.eval()

    print(f"  Phase 1: sub-AE training ({epochs} epochs, patience={patience})")
    h1 = _run_phase1(kitsune, train_loader, val_loader,
                     teacher, teacher_type, pkt_mode,
                     alpha, epochs, lr, patience, device, pkt)

    print(f"  Phase 2: output AE training ({phase2_epochs} epochs, patience={patience})")
    h2 = _run_phase2(kitsune, train_loader, val_loader,
                     phase2_epochs, lr, patience, device)

    history = {
        'train': h1['train'] + h2['train'],
        'val':   h1['val']   + h2['val'],
    }

    # Compute phi = max anomaly score on training set
    kitsune.eval()
    phi = 0.0
    with torch.no_grad():
        for (x,) in train_loader:
            x = x.to(device)
            scores = kitsune.get_anomaly_score(x)
            phi = max(phi, scores.max().item())

    torch.save({'state_dict': kitsune.state_dict(), 'phi': phi}, save_path)
    print(f"  phi (max train RMSE) = {phi:.5f}")

    return history


def _load_teacher(model_name: str, cfg: dict, dataset: str, seed: int,
                  device: torch.device, teacher_hidden_ratio: float = 0.75,
                  teacher_mixer: str = 'none'):
    n, kg = cfg['n_features'], cfg['k_groups']
    if model_name == 'bigkit':
        r          = teacher_hidden_ratio
        mixer_type = None if teacher_mixer == 'none' else teacher_mixer
        mix_tag    = '_mix' if teacher_mixer != 'none' else ''
        ckpt = f"checkpoints/{dataset}_bigkit{mix_tag}_r{int(r)}_seed{seed}.pt"
        model = KitsunePyTorch(n_features=n, k_groups=kg, hidden_ratio=r, mixer_type=mixer_type)
    else:
        ckpt = f"checkpoints/{dataset}_{model_name}_seed{seed}.pt"
        if model_name == 'vanilla':
            model = VanillaAutoencoder(n_features=n)
        elif model_name == 'cnn':
            model = ConvAutoencoder(n_features=n)
        elif model_name == 'vae':
            model = VAETeacher(n_features=n)
        else:
            model = TransformerAutoencoder(n_features=n, k_groups=kg)
    ckpt_data = torch.load(ckpt, map_location=device)
    state = ckpt_data.get('state_dict', ckpt_data) if isinstance(ckpt_data, dict) else ckpt_data
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['optdigits', 'mnist', 'landsat', 'backdoor'])
    parser.add_argument('--mode',    required=True, choices=['vanilla', 'pkt'])
    parser.add_argument('--teacher', choices=['vanilla', 'cnn', 'transformer', 'vae', 'bigkit'], default=None)
    parser.add_argument('--pkt_mode', choices=['concat', 'per_subae', 'token', 'subae_paired'], default='concat',
                        help='concat: global→concat | per_subae: global→each sub-AE | '
                             'token: transformer token i → sub-AE i (requires --teacher transformer) | '
                             'subae_paired: teacher.sub_ae[i] → student.sub_ae[i] (requires --teacher bigkit)')
    parser.add_argument('--teacher_hidden_ratio', type=float, default=0.75,
                        help='Hidden ratio for bigkit teacher (required when --teacher bigkit)')
    parser.add_argument('--teacher_mixer', choices=['none', 'mlp'], default='none',
                        help='Mixer type used when the bigkit teacher was trained (default none)')
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--phase2_epochs',type=int,   default=25,
                        help='Epochs for phase 2 (output AE only). Default 25.')
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--alpha',        type=float, default=0.5)
    parser.add_argument('--batch_size',   type=int,   default=256)
    parser.add_argument('--tag',          type=str,   default='',
                        help='Optional suffix appended to checkpoint/results filenames')
    parser.add_argument('--out_dir',      type=str,   default='results')
    args = parser.parse_args()

    if args.mode == 'pkt' and args.teacher is None:
        parser.error('--teacher required when --mode pkt')
    if args.pkt_mode == 'token' and args.teacher != 'transformer':
        parser.error('--pkt_mode token requires --teacher transformer')
    if args.pkt_mode == 'subae_paired' and args.teacher != 'bigkit':
        parser.error('--pkt_mode subae_paired requires --teacher bigkit')

    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader, test_loader, cfg = load_dataset(
        args.dataset, batch_size=args.batch_size, seed=args.seed)

    kitsune = KitsunePyTorch(cfg['n_features'], cfg['k_groups'])

    teacher = None
    if args.mode == 'pkt':
        teacher = _load_teacher(args.teacher, cfg, args.dataset, args.seed, device,
                                teacher_hidden_ratio=args.teacher_hidden_ratio,
                                teacher_mixer=args.teacher_mixer)

    label = f"{args.dataset}_kitsune_{args.mode}"
    if args.teacher:
        label += f"_{args.teacher}"
        if args.teacher == 'bigkit':
            if args.teacher_mixer != 'none':
                label += "_mix"
            label += f"_r{int(args.teacher_hidden_ratio)}"
    if args.pkt_mode != 'concat':
        label += f"_{args.pkt_mode}"
    label += f"_seed{args.seed}"
    if args.tag:
        label += f"_{args.tag}"
    save_path = f"checkpoints/{label}.pt"

    print(f"Training Kitsune [{args.mode}, pkt_mode={args.pkt_mode}] on {args.dataset} (seed={args.seed})")
    history = train_kitsune(kitsune, train_loader, val_loader,
                            teacher=teacher,
                            teacher_type=args.teacher or 'vanilla',
                            pkt_mode=args.pkt_mode,
                            alpha=args.alpha,
                            epochs=args.epochs,
                            phase2_epochs=args.phase2_epochs,
                            lr=args.lr,
                            save_path=save_path)
    print(f"Best combined val loss: {min(history['val']):.5f}")

    ckpt_data = torch.load(save_path, map_location='cpu')
    phi = ckpt_data['phi']

    print(f"Evaluating on test set...")
    metrics = evaluate(kitsune, test_loader, model_type='kitsune', phi=phi)
    model_label = f"kitsune_{args.mode}" + (f"_{args.teacher}" if args.teacher else "")
    if args.pkt_mode != 'concat':
        model_label += f"_{args.pkt_mode}"
    metrics.update({'dataset': args.dataset, 'model': model_label, 'seed': args.seed})
    print(f"  AUC-ROC={metrics['auc_roc']:.4f} | AUC-PR={metrics['auc_pr']:.4f} | "
          f"F1_oracle={metrics['f1_oracle']:.4f} | F1_phi={metrics['f1_phi']:.4f} | "
          f"phi={phi:.5f} | params={metrics['params']}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = f"{args.out_dir}/{label}.json"
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved to {out}")


if __name__ == '__main__':
    main()
