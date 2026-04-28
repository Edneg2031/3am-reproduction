from __future__ import annotations

import pytest
import torch

from three_am.training.losses import sam2_training_loss


def test_sam2_training_loss_is_finite_and_backpropagates() -> None:
    mask_logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    iou_scores = torch.sigmoid(torch.randn(2, requires_grad=True))
    occlusion_logits = torch.randn(2, 2, requires_grad=True)
    target_masks = torch.zeros(2, 4, 4)
    target_masks[0, 1:3, 1:3] = 1
    has_object = torch.tensor([True, False])

    loss = sam2_training_loss(
        {
            "mask_logits": mask_logits,
            "iou_scores": iou_scores,
            "occlusion_logits": occlusion_logits,
        },
        target_masks,
        has_object,
    )
    loss.total.backward()

    assert torch.isfinite(loss.total)
    assert mask_logits.grad is not None
    assert occlusion_logits.grad is not None


def test_sam2_training_loss_requires_adapter_fields() -> None:
    with pytest.raises(KeyError, match="iou_scores"):
        sam2_training_loss(
            {
                "mask_logits": torch.zeros(1, 4, 4),
                "occlusion_logits": torch.zeros(1, 2),
            },
            torch.zeros(1, 4, 4),
            torch.tensor([False]),
        )
