"""Baseline models. Add your own architectures here and register them in ``build_model``."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: list[int], n_classes: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        width = in_features
        for h in hidden_sizes:
            layers += [nn.Linear(width, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = h
        layers.append(nn.Linear(width, n_classes))  # logits; CrossEntropyLoss applies log-softmax
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(cfg: dict[str, Any], in_features: int, n_classes: int) -> nn.Module:
    m = cfg["model"]
    return MLP(in_features, list(m["hidden_sizes"]), n_classes, m.get("dropout", 0.0))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(cfg: dict[str, Any], params) -> torch.optim.Optimizer:
    o = cfg["optim"]
    if o["name"] == "sgd":
        return torch.optim.SGD(params, lr=o["lr"], momentum=o.get("momentum", 0.0), weight_decay=o["weight_decay"])
    if o["name"] == "adam":
        return torch.optim.Adam(params, lr=o["lr"], weight_decay=o["weight_decay"])
    if o["name"] == "adamw":
        return torch.optim.AdamW(params, lr=o["lr"], weight_decay=o["weight_decay"])
    raise ValueError(f"Unknown optimizer {o['name']!r}")
