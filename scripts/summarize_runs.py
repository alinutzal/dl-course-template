"""Summarize repeated runs: mean +/- sample std of each metric per experiment name.

Usage:
    python scripts/summarize_runs.py [runs_dir]

Report results across seeds (e.g. three or more) rather than a single run when random
variation matters; Lab 2 and the capstone require this.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METRICS = ["test_acc", "test_loss", "val_acc", "best_epoch", "train_seconds"]


def summarize(runs_dir: str | Path = "runs") -> dict[str, dict[str, tuple[float, float, int]]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(Path(runs_dir).glob("*/*/results.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        groups[r["name"]].append(r)
    table: dict[str, dict[str, tuple[float, float, int]]] = {}
    for name, rows in groups.items():
        table[name] = {}
        for m in METRICS:
            vals = [r[m] for r in rows if r.get(m) is not None]
            if vals:
                sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
                table[name][m] = (statistics.mean(vals), sd, len(vals))
    return table


def main() -> None:
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "runs"
    table = summarize(runs_dir)
    if not table:
        print(f"no results.json files under {runs_dir}/")
        return
    for name, metrics in sorted(table.items()):
        n = next(iter(metrics.values()))[2]
        print(f"\n{name}  (n={n} runs)")
        for m, (mean, sd, _) in metrics.items():
            print(f"  {m:14s} {mean:10.4f} +/- {sd:.4f}")


if __name__ == "__main__":
    main()
