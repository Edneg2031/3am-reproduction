from __future__ import annotations

import numpy as np

from three_am.evaluation.metrics import mask_iou, tracking_metrics


def test_mask_iou_empty_masks_is_one() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    assert mask_iou(mask, mask) == 1.0


def test_tracking_metrics_visible_frames() -> None:
    target = [np.array([[1, 0], [0, 0]], dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)]
    prediction = [np.array([[1, 0], [0, 0]], dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)]
    metrics = tracking_metrics(prediction, target)
    assert metrics["iou"] == 1.0
    assert metrics["tracking_recall"] == 1.0
    assert metrics["accuracy"] == 1.0
