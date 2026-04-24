from __future__ import annotations

import torch
from torch.nn import functional as F


def binary_mask_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(-2, -1))
    union = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = 1 - (2 * intersection + 1) / (union + 1)
    return bce + dice.mean()
