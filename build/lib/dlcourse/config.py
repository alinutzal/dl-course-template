"""Load a YAML experiment config and apply ``key.sub=value`` command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return apply_overrides(cfg, overrides or [])


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Return a copy of ``cfg`` with each ``a.b.c=value`` override applied.

    Values are parsed as YAML, so ``lr=0.01`` gives a float, ``hidden_sizes=[64,32]``
    a list and ``deterministic=true`` a bool. Unknown keys raise ``KeyError`` so that
    typos do not silently create new settings.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.split(".")
        node = cfg
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                raise KeyError(f"Unknown config section {k!r} in {dotted!r}")
            node = node[k]
        if keys[-1] not in node:
            raise KeyError(f"Unknown config key {dotted!r}")
        node[keys[-1]] = yaml.safe_load(raw)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
