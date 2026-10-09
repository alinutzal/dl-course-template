"""Plot learning curves from a run's metrics.csv.

Usage:
    python scripts/plot_run.py runs/<name>/<run>/
Saves learning_curves.png in the run directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


def plot(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "metrics.csv")
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(df["epoch"], df["train_loss"], label="train")
    ax_loss.plot(df["epoch"], df["val_loss"], label="validation")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax_acc.plot(df["epoch"], df["train_acc"], label="train")
    ax_acc.plot(df["epoch"], df["val_acc"], label="validation")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    for ax in (ax_loss, ax_acc):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = run_dir / "learning_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot(sys.argv[1]))
