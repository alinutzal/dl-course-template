# CSCI 4852/6952 Deep Learning: Project Template

This is the starting repository for every lab and for your capstone. It gives you a reproducible training setup, so you can spend your time on the deep-learning questions instead of on plumbing.

## Quick start

```bash
# 1. Set up the course environment once (see SETUP_OSC.md or SETUP_LOCAL.md), then activate it
# 2. From this folder:
pip install -e .            # makes the `dlcourse` package importable
python -m pytest            # should report: 8 passed
python scripts/train.py     # trains the baseline MLP on a small synthetic dataset
```

Train on MNIST, change settings from the command line, and compare seeds:

```bash
python scripts/train.py data.dataset=mnist train.epochs=5 seed=1
python scripts/train.py data.dataset=mnist train.epochs=5 seed=2
python scripts/train.py data.dataset=mnist train.epochs=5 seed=3
python scripts/summarize_runs.py          # mean +/- std across seeds
python scripts/plot_run.py runs/baseline-mlp/<run-folder>/
```

Give every distinct experiment its own name (for example `name=mnist-adam-lr1e-3`), so that the summary groups only runs that differ by seed.

## Layout

| Path | What it is |
|---|---|
| `configs/default.yaml` | **Every** setting that affects results. Copy it for each experiment; do not hard-code values in code |
| `src/dlcourse/` | Reusable code: config loading, seeding and devices, data splits, models, training loop |
| `scripts/train.py` | Runs one experiment and writes a self-describing run folder |
| `scripts/summarize_runs.py`, `scripts/plot_run.py` | Aggregate seeds; plot learning curves |
| `tests/` | Checks for determinism, split leakage, checkpoints and learning. **Keep them passing** |
| `notebooks/00_quickstart.ipynb` | The same workflow from a notebook |
| `environment.yml`, `requirements*.txt`, `smoke_test.py` | The pinned course environment |
| `runs/`, `data/` | Created automatically; **git-ignored** (too large, and regenerable) |

Each run folder contains `config.yaml` (the resolved settings), `env.json` (versions, device, git commit), `metrics.csv` (per-epoch curves), `best.pt` (best-validation weights) and `results.json` (test metrics, parameter count, time, GPU memory).

## Reproducibility checklist (graded in labs and the capstone)

- [ ] Code is in Git, with small, descriptive commits. `env.json` records the commit, so commit before final runs.
- [ ] Environment pinned (`environment.yml` / `requirements*.txt`); any extra package is added there.
- [ ] Seeds set through the config. Report **mean ± std over at least 3 seeds** when comparing models (6952: required in the capstone).
- [ ] All settings in config files or command-line overrides; nothing important is hard-coded.
- [ ] Train, validation and test splits are fixed by the seed. **The test set is used once, at the end**, never for choosing hyperparameters or epochs.
- [ ] Logs and checkpoints are saved (`metrics.csv`, `best.pt`).
- [ ] The README explains exactly how to reproduce every reported number (commands plus configs).
- [ ] Device handling is explicit (`device=auto|cpu|cuda`), and run time and GPU memory are reported.

## Determinism

`deterministic: true` in the config makes repeated runs on the **same machine** produce identical numbers, including on GPU, at some speed cost. Different GPUs, drivers or library versions can still give slightly different results. That is one reason to report variation across seeds. Background: https://docs.pytorch.org/docs/stable/notes/randomness.html

## AI-use statement

Every submission includes a short statement naming any AI tools used, what for, and which parts of the work they influenced, or "No AI tools used". See the syllabus *AI Use in This Course* section.

## Credits

The training-loop and helper design follows the structure of the course's earlier lab helpers, which were adapted from Sebastian Raschka's STAT 453 materials (MIT License, © 2020 Sebastian Raschka). This template is a rewrite; the course repositories are released under the Apache-2.0 license (`LICENSE`).
