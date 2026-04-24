from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def mark_trainable_3am_modules(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        trainable = any(token in name.lower() for token in ("memory_attention", "mask_decoder", "feature_merger"))
        parameter.requires_grad_(trainable)


def build_adamw(model: nn.Module) -> torch.optim.Optimizer:
    groups: list[dict[str, object]] = []
    learning_rates = {
        "memory_attention": 5e-6,
        "mask_decoder": 5e-6,
        "feature_merger": 1e-5,
    }
    for key, learning_rate in learning_rates.items():
        params = [parameter for name, parameter in model.named_parameters() if key in name.lower() and parameter.requires_grad]
        if params:
            groups.append({"params": params, "lr": learning_rate})
    if not groups:
        trainable_params: Iterable[nn.Parameter] = (parameter for parameter in model.parameters() if parameter.requires_grad)
        groups.append({"params": list(trainable_params), "lr": 1e-5})
    return torch.optim.AdamW(groups)
