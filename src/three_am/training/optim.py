from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def mark_trainable_3am_modules(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        trainable = any(token in name.lower() for token in ("memory_attention", "mask_decoder", "feature_merger"))
        parameter.requires_grad_(trainable)


def build_adamw(
    model: nn.Module,
    learning_rates: dict[str, float] | None = None,
    *,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    groups: list[dict[str, object]] = []
    learning_rates = learning_rates or {
        "memory_attention": 5e-6,
        "mask_decoder": 5e-6,
        "feature_merger": 1e-5,
    }
    for key, learning_rate in learning_rates.items():
        params = [parameter for name, parameter in model.named_parameters() if key in name.lower() and parameter.requires_grad]
        if params:
            groups.append({"params": params, "lr": learning_rate, "weight_decay": weight_decay})
    if not groups:
        trainable_params: Iterable[nn.Parameter] = (parameter for parameter in model.parameters() if parameter.requires_grad)
        groups.append({"params": list(trainable_params), "lr": learning_rates.get("default", 1e-5), "weight_decay": weight_decay})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)
