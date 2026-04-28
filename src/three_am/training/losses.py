from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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


@dataclass(frozen=True)
class Sam2LossWeights:
    focal: float = 20.0
    dice: float = 1.0
    iou: float = 1.0
    occlusion: float = 1.0


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    focal: torch.Tensor
    dice: torch.Tensor
    iou: torch.Tensor
    occlusion: torch.Tensor

    def detached_items(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach().cpu()),
            "focal": float(self.focal.detach().cpu()),
            "dice": float(self.dice.detach().cpu()),
            "iou": float(self.iou.detach().cpu()),
            "occlusion": float(self.occlusion.detach().cpu()),
        }


def sigmoid_focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    target = target.to(dtype=logits.dtype)
    probability = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = probability * target + (1 - probability) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t).pow(gamma) * cross_entropy).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(dtype=logits.dtype)
    probability = logits.sigmoid()
    probability = probability.flatten(1)
    target = target.flatten(1)
    numerator = 2 * (probability * target).sum(dim=1) + 1
    denominator = probability.sum(dim=1) + target.sum(dim=1) + 1
    return (1 - numerator / denominator).mean()


def mask_iou_targets(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = logits.detach().sigmoid() > 0.5
    target_bool = target.detach() > 0.5
    prediction = prediction.flatten(1)
    target_bool = target_bool.flatten(1)
    intersection = (prediction & target_bool).sum(dim=1).to(dtype=logits.dtype)
    union = (prediction | target_bool).sum(dim=1).to(dtype=logits.dtype)
    return torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union, dtype=logits.dtype))


def _require_output(outputs: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in outputs:
        raise KeyError(f"SAM2 training adapter output is missing required key: {key}")
    return outputs[key]


def _normalize_mask_logits(mask_logits: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    if mask_logits.ndim == 4 and mask_logits.shape[1] == 1:
        mask_logits = mask_logits[:, 0]
    if mask_logits.shape != target_masks.shape:
        raise ValueError(f"mask_logits shape {tuple(mask_logits.shape)} does not match target {tuple(target_masks.shape)}")
    return mask_logits


def sam2_training_loss(
    outputs: Mapping[str, torch.Tensor],
    target_masks: torch.Tensor,
    has_object: torch.Tensor,
    weights: Sam2LossWeights | None = None,
) -> LossBreakdown:
    weights = weights or Sam2LossWeights()
    mask_logits = _normalize_mask_logits(_require_output(outputs, "mask_logits"), target_masks)
    iou_scores = _require_output(outputs, "iou_scores").flatten()
    occlusion_logits = _require_output(outputs, "occlusion_logits")
    target_masks = target_masks.to(dtype=mask_logits.dtype)
    if iou_scores.shape[0] != target_masks.shape[0]:
        raise ValueError(f"iou_scores must have one value per frame, got shape {tuple(iou_scores.shape)}")
    if occlusion_logits.ndim != 2 or occlusion_logits.shape != (target_masks.shape[0], 2):
        raise ValueError(
            "occlusion_logits must have shape "
            f"({target_masks.shape[0]}, 2), got {tuple(occlusion_logits.shape)}"
        )
    focal = sigmoid_focal_loss(mask_logits, target_masks)
    dice = dice_loss(mask_logits, target_masks)
    iou_targets = mask_iou_targets(mask_logits, target_masks).to(device=iou_scores.device, dtype=iou_scores.dtype)
    iou = F.l1_loss(iou_scores, iou_targets)
    visibility_targets = has_object.to(device=occlusion_logits.device, dtype=torch.long)
    occlusion = F.cross_entropy(occlusion_logits, visibility_targets)
    total = weights.focal * focal + weights.dice * dice + weights.iou * iou + weights.occlusion * occlusion
    return LossBreakdown(total=total, focal=focal, dice=dice, iou=iou, occlusion=occlusion)
