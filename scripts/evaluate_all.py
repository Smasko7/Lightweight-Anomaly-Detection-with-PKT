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
from evaluation.metrics import evaluate
from training.utils import set_seed, get_device

RESULTS_DIR = Path('results')
CKPT_DIR    = Path('checkpoints')
SEEDS       = [42, 123, 456]
DATASETS    = ['optdigits', 'speech', 'mnist']


def load_model(model_name: str, cfg: dict, ckpt_path: Path, device: torch.device):
    n, kg = cfg['n_features'], cfg['k_groups']
    if model_name == 'kitsune_vanilla' or model_name == 'kitsune_pkt_cnn' or \
       model_name == 'kitsune_pkt_transformer' or model_name == 'kitsune_pkt_vanilla':
        model = KitsunePyTorch(n_features=n, k_groups=kg)
    elif model_name == 'vanilla':
        model = VanillaAutoencoder(n_features=n)
    elif model_name == 'cnn':
        model = ConvAutoencoder(n_features=n)
    elif model_name == 'transformer':
        model = TransformerAutoencoder(n_features=n, k_groups=kg)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    ckpt_data = torch.load(ckpt_path, map_location=device)
    state = ckpt_data['state_dict'] if 'state_dict' in ckpt_data else ckpt_data
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def model_type_str(model_name: str) -> str:
    if model_name.startswith('kitsune'):
        return 'kitsune'
    return model_name  # 'vanilla', 'cnn', 'transformer'


def ckpt_filename(dataset: str, model_name: str, seed: int) -> str:
    if model_name.startswith('kitsune'):
        # e.g. kitsune_vanilla → kitsune_vanilla, kitsune_pkt_cnn → kitsune_pkt_cnn
        return f'{dataset}_{model_name}_seed{seed}.pt'
    else:
        return f'{dataset}_{model_name}_seed{seed}.pt'


# All model variants to evaluate
MODEL_VARIANTS = [
    'kitsune_vanilla',
    'vanilla',
    'cnn',
    'transformer',
    'kitsune_pkt_vanilla',
    'kitsune_pkt_cnn',
    'kitsune_pkt_transformer',
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
