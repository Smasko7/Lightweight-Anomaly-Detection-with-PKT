"""ADBench dataset loading and preprocessing."""
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(__file__).parent / "raw"

DATASET_CONFIGS = {
    'optdigits': {
        'file': '26_optdigits.npz',
        'n_features': 64,
        'k_groups': 8,
        'features_per_group': 8,
    },
    'mnist': {
        'file': '24_mnist.npz',
        'n_features': 100,
        'k_groups': 10,
        'features_per_group': 10,
    },
    'landsat': {
        'file': '19_landsat.npz',
        'n_features': 36,
        'k_groups': 6,
        'features_per_group': 6,
    },
    'backdoor': {
        'file': '3_backdoor.npz',
        'n_features': 196,
        'k_groups': 14,
        'features_per_group': 14,
    },
}


def load_dataset(name: str, batch_size: int = 256, seed: int = 42):
    """
    Returns:
        train_loader: DataLoader (normal data only, 80%)
        val_loader:   DataLoader (normal data only, 20%)
        test_loader:  DataLoader (all data with labels)
        config:       dict with n_features, k_groups, features_per_group
    """
    cfg = DATASET_CONFIGS[name]
    path = DATA_DIR / cfg['file']
    data = np.load(path, allow_pickle=True)
    X, y = data['X'].astype(np.float32), data['y'].astype(np.int64)

    X_normal = X[y == 0]

    X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=seed)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val)),
        batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y)),
        batch_size=batch_size, shuffle=False,
    )

    return train_loader, val_loader, test_loader, cfg
