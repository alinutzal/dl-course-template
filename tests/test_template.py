"""Tests that the scaffold is reproducible and correct. Run with:  python -m pytest

They use the offline synthetic dataset on CPU and finish in a few seconds. Keep them
passing as you modify the template for labs and the capstone.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from dlcourse.config import apply_overrides, load_config
from dlcourse.data import load_splits, make_loaders, split_indices
from dlcourse.models import build_model, build_optimizer, count_parameters
from dlcourse.reproducibility import set_seed
from dlcourse.training import fit, load_weights, run_epoch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "default.yaml"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cfg(tmp_path):
    return load_config(
        DEFAULT,
        ["device=cpu", "train.epochs=3", f"output.runs_dir={tmp_path.as_posix()}/runs"],
    )


def _train(cfg, run_dir):
    set_seed(cfg["seed"])
    train, val, test = load_splits(cfg)
    loaders = make_loaders(train, val, test, cfg)
    model = build_model(cfg, 20, 3)
    opt = build_optimizer(cfg, model.parameters())
    run_dir.mkdir(parents=True, exist_ok=True)
    history = fit(model, loaders, opt, cfg, run_dir, torch.device("cpu"))
    return model, loaders, history


def test_config_overrides_are_typed_and_validated(cfg):
    out = apply_overrides(cfg, ["optim.lr=0.05", "model.hidden_sizes=[32]", "deterministic=true"])
    assert out["optim"]["lr"] == 0.05 and out["model"]["hidden_sizes"] == [32] and out["deterministic"] is True
    assert cfg["optim"]["lr"] == 0.001  # original untouched
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["optim.learning_rate=0.1"])  # typo must not pass silently


def test_splits_are_disjoint_and_complete(cfg):
    train, val, test = load_splits(cfg)
    a, b, c = set(split_indices(train)), set(split_indices(val)), set(split_indices(test))
    assert not (a & b) and not (a & c) and not (b & c), "data leakage between splits"
    assert len(a | b | c) == len(train) + len(val) + len(test)


def test_same_seed_same_split_different_seed_different_split(cfg):
    s1 = split_indices(load_splits(cfg)[1])
    s2 = split_indices(load_splits(cfg)[1])
    s3 = split_indices(load_splits(apply_overrides(cfg, ["seed=2"]))[1])
    assert s1 == s2 and s1 != s3


def test_training_is_reproducible(cfg, tmp_path):
    _, _, h1 = _train(cfg, tmp_path / "a")
    _, _, h2 = _train(cfg, tmp_path / "b")
    strip = lambda h: [{k: v for k, v in r.items() if k != "epoch_seconds"} for r in h]  # noqa: E731
    assert strip(h1) == strip(h2)


def test_training_learns(cfg, tmp_path):
    _, _, history = _train(cfg, tmp_path / "run")
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert history[-1]["val_acc"] > 0.6  # three-class problem; chance is about 0.33


def test_best_checkpoint_round_trip(cfg, tmp_path):
    model, loaders, history = _train(cfg, tmp_path / "run")
    best = min(history, key=lambda r: r["val_loss"])
    fresh = load_weights(build_model(cfg, 20, 3), tmp_path / "run" / "best.pt", torch.device("cpu"))
    val_loss, _ = run_epoch(fresh, loaders["val"], torch.device("cpu"))
    assert val_loss == pytest.approx(best["val_loss"], abs=1e-5)


def test_parameter_count_matches_architecture(cfg):
    model = build_model(cfg, 20, 3)  # 20 -> 128 -> 64 -> 3
    assert count_parameters(model) == (20 * 128 + 128) + (128 * 64 + 64) + (64 * 3 + 3)


def test_cli_end_to_end(cfg, tmp_path):
    runs = (tmp_path / "runs").as_posix()
    train_script = _load_script("train")
    for seed in (1, 2):
        run_dir = train_script.main(
            ["--config", str(DEFAULT), "device=cpu", "train.epochs=2", f"seed={seed}", f"output.runs_dir={runs}"]
        )
        for f in ("config.yaml", "env.json", "metrics.csv", "best.pt", "results.json"):
            assert (run_dir / f).exists(), f
    results = json.loads((run_dir / "results.json").read_text())
    assert 0.0 <= results["test_acc"] <= 1.0
    table = _load_script("summarize_runs").summarize(runs)
    assert table["baseline-mlp"]["test_acc"][2] == 2  # two seeds aggregated
    assert _load_script("plot_run").plot(run_dir).exists()


def test_diverged_run_still_checkpoints(cfg, tmp_path):
    """A run whose loss is NaN from the start must still save best.pt (regression test)."""
    set_seed(cfg["seed"])
    train, val, test = load_splits(cfg)
    loaders = make_loaders(train, val, test, cfg)
    model = build_model(cfg, 20, 3)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float("nan"))
    history = fit(model, loaders, build_optimizer(cfg, model.parameters()), cfg, tmp_path, torch.device("cpu"))
    assert (tmp_path / "best.pt").exists()
    assert all(r["val_loss"] != r["val_loss"] for r in history)  # NaN everywhere, as constructed
