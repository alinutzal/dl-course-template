"""Train one model from a config file and record everything needed to reproduce it.

Usage:
    python scripts/train.py --config configs/default.yaml [key=value ...]

Each run creates runs/<name>/seed<seed>-<timestamp>/ containing:
    config.yaml   the fully resolved config (including overrides)
    env.json      Python/PyTorch versions, device, git commit
    metrics.csv   per-epoch train/validation loss and accuracy
    best.pt       weights of the epoch with the lowest validation loss
    results.json  test metrics of the best epoch, parameter count, run time
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from dlcourse.config import load_config, save_config
from dlcourse.data import dataset_info, load_splits, make_loaders
from dlcourse.models import build_model, build_optimizer, count_parameters
from dlcourse.reproducibility import environment_info, get_device, set_seed
from dlcourse.training import fit, load_weights, run_epoch


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value, e.g. optim.lr=0.01 seed=2")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["seed"], cfg.get("deterministic", False))
    device = get_device(cfg.get("device", "auto"))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = Path(cfg["output"]["runs_dir"]) / cfg["name"] / f"seed{cfg['seed']}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "env.json").write_text(json.dumps(environment_info(device), indent=2), encoding="utf-8")
    print(f"run directory: {run_dir}\ndevice: {device}")

    train, val, test = load_splits(cfg)
    loaders = make_loaders(train, val, test, cfg)
    info = dataset_info(cfg["data"]["dataset"])
    model = build_model(cfg, info["in_features"], info["n_classes"]).to(device)
    optimizer = build_optimizer(cfg, model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    history = fit(model, loaders, optimizer, cfg, run_dir, device)
    train_seconds = time.perf_counter() - t0

    load_weights(model, run_dir / "best.pt", device)
    test_loss, test_acc = run_epoch(model, loaders["test"], device)
    best = min(history, key=lambda r: r["val_loss"])
    results = {
        "name": cfg["name"],
        "seed": cfg["seed"],
        "best_epoch": best["epoch"],
        "val_loss": best["val_loss"],
        "val_acc": best["val_acc"],
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "n_parameters": count_parameters(model),
        "train_seconds": round(train_seconds, 2),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
        if device.type == "cuda"
        else None,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return run_dir


if __name__ == "__main__":
    main()
