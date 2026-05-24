"""
Evaluate all trained checkpoints and save results to results/*.json.
Can be run standalone after training, or after run_all.sh.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pathlib import Path

from data.loader import load_dataset, DATASET_CONFIGS
from models.kitsune import KitsunePyTorch
from models.vanilla_ae import VanillaAutoencoder
from models.conv_ae import ConvAutoencoder
from models.transformer_ae import TransformerAutoencoder
from models.vae import VAETeacher
from evaluation.metrics import evaluate
from training.utils import set_seed, get_device

RESULTS_DIR = Path('results')
CKPT_DIR    = Path('checkpoints')
SEEDS       = [42, 123, 456]
DATASETS    = ['optdigits', 'mnist', 'landsat', 'backdoor']


def _parse_bigkit_ratio(name: str) -> float:
    """Extract hidden_ratio from a bigkit-bearing name. Handles both mixer-less
    ('bigkit_r4', 'kitsune_pkt_bigkit_r2_subae_paired') and mixer-bearing
    ('bigkit_mix_r4', 'kitsune_pkt_bigkit_mix_r4_subae_paired') variants."""
    # Find '_r<digits>' anchored after 'bigkit'
    bk = name.index('bigkit')
    rest = name[bk:]
    # The ratio always lives at the first '_r' that's followed by a digit
    for k in range(len(rest) - 1):
        if rest[k:k+2] == '_r' and k + 2 < len(rest) and rest[k+2].isdigit():
            i = k + 2
            j = i
            while j < len(rest) and (rest[j].isdigit() or rest[j] == '.'):
                j += 1
            return float(rest[i:j])
    raise ValueError(f"Could not parse hidden_ratio from name: {name}")


def load_model(model_name: str, cfg: dict, ckpt_path: Path, device: torch.device):
    n, kg = cfg['n_features'], cfg['k_groups']
    if model_name.startswith('bigkit'):
        # Standalone bigkit teacher — wider sub-AEs, no trained output AE.
        # 'bigkit_r{R}' has no mixer; 'bigkit_mix_r{R}' has the MLP cross-group mixer.
        r          = _parse_bigkit_ratio(model_name)
        mixer_type = 'mlp' if '_mix_' in model_name else None
        model = KitsunePyTorch(n_features=n, k_groups=kg, hidden_ratio=r, mixer_type=mixer_type)
    elif model_name.startswith('kitsune'):
        # Student Kitsune (always default hidden_ratio = 0.75, no mixer) — covers
        # both baseline (kitsune_vanilla) and PKT variants (kitsune_pkt_*).
        model = KitsunePyTorch(n_features=n, k_groups=kg)
    elif model_name == 'vanilla':
        model = VanillaAutoencoder(n_features=n)
    elif model_name == 'cnn':
        model = ConvAutoencoder(n_features=n)
    elif model_name == 'transformer':
        model = TransformerAutoencoder(n_features=n, k_groups=kg)
    elif model_name == 'vae':
        model = VAETeacher(n_features=n)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    ckpt_data = torch.load(ckpt_path, map_location=device)
    state = ckpt_data['state_dict'] if 'state_dict' in ckpt_data else ckpt_data
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def model_type_str(model_name: str) -> str:
    if model_name.startswith('bigkit'):
        return 'bigkit'         # direct RMSE sum scoring (no trained output AE)
    if model_name.startswith('kitsune'):
        return 'kitsune'        # output AE scoring
    return model_name  # 'vanilla', 'cnn', 'transformer', 'vae'


def ckpt_filename(dataset: str, model_name: str, seed: int) -> str:
    if model_name.startswith('kitsune'):
        # e.g. kitsune_vanilla → kitsune_vanilla, kitsune_pkt_cnn → kitsune_pkt_cnn
        return f'{dataset}_{model_name}_seed{seed}.pt'
    else:
        return f'{dataset}_{model_name}_seed{seed}.pt'


# All model variants to evaluate
MODEL_VARIANTS = [
    # Kitsune baseline (no KD)
    'kitsune_vanilla',
    # Standalone teachers
    'vanilla', 'cnn', 'transformer', 'vae',
    # Kitsune + PKT — concat routing (the default; checkpoint name has no routing suffix)
    'kitsune_pkt_vanilla', 'kitsune_pkt_cnn', 'kitsune_pkt_transformer', 'kitsune_pkt_vae',
    # Kitsune + PKT — per_subae routing
    'kitsune_pkt_vanilla_per_subae', 'kitsune_pkt_cnn_per_subae',
    'kitsune_pkt_transformer_per_subae', 'kitsune_pkt_vae_per_subae',
    # Kitsune + PKT — token routing (transformer teacher only)
    'kitsune_pkt_transformer_token',

    # Big-Kitsune teacher sweep (hidden_ratio ∈ {2,4,8,16,32}) and PKT variants
    'bigkit_r2',  'bigkit_r4',  'bigkit_r8',  'bigkit_r16',  'bigkit_r32',
    'kitsune_pkt_bigkit_r2',               'kitsune_pkt_bigkit_r4',               'kitsune_pkt_bigkit_r8',               'kitsune_pkt_bigkit_r16',               'kitsune_pkt_bigkit_r32',
    'kitsune_pkt_bigkit_r2_per_subae',     'kitsune_pkt_bigkit_r4_per_subae',     'kitsune_pkt_bigkit_r8_per_subae',     'kitsune_pkt_bigkit_r16_per_subae',     'kitsune_pkt_bigkit_r32_per_subae',
    'kitsune_pkt_bigkit_r2_subae_paired',  'kitsune_pkt_bigkit_r4_subae_paired',  'kitsune_pkt_bigkit_r8_subae_paired',  'kitsune_pkt_bigkit_r16_subae_paired',  'kitsune_pkt_bigkit_r32_subae_paired',

    # Mixer variant: bigkit + cross-group MLP mixer
    'bigkit_mix_r2',  'bigkit_mix_r4',  'bigkit_mix_r8',  'bigkit_mix_r16',  'bigkit_mix_r32',
    'kitsune_pkt_bigkit_mix_r2',               'kitsune_pkt_bigkit_mix_r4',               'kitsune_pkt_bigkit_mix_r8',               'kitsune_pkt_bigkit_mix_r16',               'kitsune_pkt_bigkit_mix_r32',
    'kitsune_pkt_bigkit_mix_r2_per_subae',     'kitsune_pkt_bigkit_mix_r4_per_subae',     'kitsune_pkt_bigkit_mix_r8_per_subae',     'kitsune_pkt_bigkit_mix_r16_per_subae',     'kitsune_pkt_bigkit_mix_r32_per_subae',
    'kitsune_pkt_bigkit_mix_r2_subae_paired',  'kitsune_pkt_bigkit_mix_r4_subae_paired',  'kitsune_pkt_bigkit_mix_r8_subae_paired',  'kitsune_pkt_bigkit_mix_r16_subae_paired',  'kitsune_pkt_bigkit_mix_r32_subae_paired',
]


def run_evaluate_all():
    RESULTS_DIR.mkdir(exist_ok=True)
    device = get_device()
    total, skipped = 0, 0

    for dataset in DATASETS:
        _, _, test_loader, cfg = load_dataset(dataset, batch_size=256, seed=42)

        for model_name in MODEL_VARIANTS:
            for seed in SEEDS:
                ckpt = CKPT_DIR / ckpt_filename(dataset, model_name, seed)
                if not ckpt.exists():
                    skipped += 1
                    continue

                set_seed(seed)
                model = load_model(model_name, cfg, ckpt, device)
                mtype = model_type_str(model_name)

                ckpt_data = torch.load(ckpt, map_location=device)
                phi = ckpt_data.get('phi') if isinstance(ckpt_data, dict) else None

                metrics = evaluate(model, test_loader, model_type=mtype, phi=phi)
                metrics.update({'dataset': dataset, 'model': model_name, 'seed': seed})

                out = RESULTS_DIR / f'{dataset}_{model_name}_seed{seed}.json'
                with open(out, 'w') as f:
                    json.dump(metrics, f, indent=2)

                f1_phi_str = f'{metrics["f1_phi"]:.4f}' if phi is not None else 'N/A'
                print(f'  {dataset} | {model_name:30s} | seed={seed} | '
                      f'AUC-ROC={metrics["auc_roc"]:.4f} | '
                      f'AUC-PR={metrics["auc_pr"]:.4f} | '
                      f'F1_oracle={metrics["f1_oracle"]:.4f} | '
                      f'F1_phi={f1_phi_str} | '
                      f'params={metrics["params"]}')
                total += 1

    print(f'\nDone: {total} evaluated, {skipped} skipped (checkpoint not found).')


if __name__ == '__main__':
    run_evaluate_all()
