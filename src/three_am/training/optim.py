from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


TRAINABLE_3AM_MODULE_TOKENS = ("memory_attention", "memory_encoder", "mask_decoder", "feature_merger")


def mark_trainable_3am_modules(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        trainable = any(token in name.lower() for token in TRAINABLE_3AM_MODULE_TOKENS)
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
        "memory_encoder": 5e-6,
        "mask_decoder": 5e-6,
        "feature_merger": 1e-5,
    }
    grouped_parameter_ids: set[int] = set()
    for key, learning_rate in learning_rates.items():
        params = [parameter for name, parameter in model.named_parameters() if key in name.lower() and parameter.requires_grad]
        if params:
            grouped_parameter_ids.update(id(parameter) for parameter in params)
            groups.append({"params": params, "lr": learning_rate, "weight_decay": weight_decay})
    ungrouped_trainable_params: Iterable[nn.Parameter] = (
        parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in grouped_parameter_ids
    )
    default_params = list(ungrouped_trainable_params)
    if default_params or not groups:
        groups.append({"params": default_params, "lr": learning_rates.get("default", 1e-5), "weight_decay": weight_decay})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)
