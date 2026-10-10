"""Training loop with per-epoch logging, best-checkpoint saving and early stopping."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One pass over ``loader``. Trains if an optimizer is given, otherwise evaluates.

    Returns (mean loss, accuracy) weighted by example count, so a smaller final batch is
    not over-counted.
    """
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                (loss / len(y)).backward()  # mean loss for the gradient step
                optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def fit(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    run_dir: Path,
    device: torch.device,
) -> list[dict[str, float]]:
    """Train for ``cfg['train']['epochs']`` epochs, selecting the best epoch on validation loss.

    Writes ``metrics.csv`` (one row per epoch) and ``best.pt`` (weights of the best epoch)
    into ``run_dir``. The test loader is not touched here.
    """
    epochs = cfg["train"]["epochs"]
    patience = cfg["train"].get("early_stopping_patience", 0)
    history: list[dict[str, float]] = []
    best_val, since_best = float("inf"), 0
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_seconds"]

    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = run_epoch(model, loaders["train"], device, optimizer)
            val_loss, val_acc = run_epoch(model, loaders["val"], device)
            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "epoch_seconds": round(time.perf_counter() - t0, 3),
            }
            writer.writerow(row)
            f.flush()
            history.append(row)
            print(
                f"epoch {epoch:3d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.3f}"
                f" | val loss {val_loss:.4f} acc {val_acc:.3f} | {row['epoch_seconds']:.1f}s"
            )
            # Always checkpoint the first epoch, so a run whose validation loss is never finite
            # (e.g. a diverged run with NaN loss) still has a best.pt to evaluate and report.
            if epoch == 1 or val_loss < best_val:
                best_val, since_best = (val_loss if val_loss == val_loss else float("inf")), 0
                torch.save(model.state_dict(), run_dir / "best.pt")
            else:
                since_best += 1
                if patience and since_best >= patience:
                    print(f"early stopping: no val-loss improvement for {patience} epochs")
                    break
    return history


def load_weights(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model
