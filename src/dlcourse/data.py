"""Datasets, seeded train/validation/test splits and DataLoaders."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split

from .reproducibility import seed_worker


def make_synthetic(n: int = 2000, n_features: int = 20, n_classes: int = 3, seed: int = 0) -> TensorDataset:
    """Overlapping Gaussian blobs: fast and offline. Class overlap is tuned so a small MLP
    reaches roughly 85-90% test accuracy, which leaves visible variation across seeds."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, n_features, generator=g) * 0.5
    y = torch.randint(0, n_classes, (n,), generator=g)
    x = centers[y] + torch.randn(n, n_features, generator=g)
    return TensorDataset(x, y)


def dataset_info(name: str) -> dict[str, Any]:
    if name == "synthetic":
        return {"in_features": 20, "n_classes": 3}
    if name == "mnist":
        return {"in_features": 28 * 28, "n_classes": 10}
    raise ValueError(f"Unknown dataset {name!r}")


def load_splits(cfg: dict[str, Any]) -> tuple[Dataset, Dataset, Dataset]:
    """Return (train, val, test).

    The split depends only on ``cfg['seed']``, so every run with the same seed sees the same
    examples in each split. The test split is never used for model selection.
    """
    d = cfg["data"]
    g = torch.Generator().manual_seed(cfg["seed"])
    if d["dataset"] == "synthetic":
        full = make_synthetic(seed=0)  # fixed data; only the split depends on the seed
        n_test = int(len(full) * d["test_fraction"])
        n_val = int(len(full) * d["val_fraction"])
        train, val, test = random_split(full, [len(full) - n_val - n_test, n_val, n_test], generator=g)
        return train, val, test
    if d["dataset"] == "mnist":
        from torchvision import datasets, transforms

        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        full_train = datasets.MNIST(d["root"], train=True, download=True, transform=tf)
        test = datasets.MNIST(d["root"], train=False, download=True, transform=tf)
        n_val = int(len(full_train) * d["val_fraction"])
        train, val = random_split(full_train, [len(full_train) - n_val, n_val], generator=g)
        return train, val, test
    raise ValueError(f"Unknown dataset {d['dataset']!r}")


def split_indices(ds: Dataset) -> list[int]:
    """Underlying indices of a ``random_split`` subset (used by the leakage test)."""
    return list(ds.indices) if isinstance(ds, Subset) else list(range(len(ds)))


def make_loaders(train: Dataset, val: Dataset, test: Dataset, cfg: dict[str, Any]) -> dict[str, DataLoader]:
    d = cfg["data"]
    g = torch.Generator().manual_seed(cfg["seed"])
    common = {"batch_size": d["batch_size"], "num_workers": d["num_workers"], "worker_init_fn": seed_worker}
    return {
        # shuffle the training set every epoch (seeded); never shuffle evaluation sets
        "train": DataLoader(train, shuffle=True, generator=g, **common),
        "val": DataLoader(val, shuffle=False, **common),
        "test": DataLoader(test, shuffle=False, **common),
    }
