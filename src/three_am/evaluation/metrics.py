from __future__ import annotations

import numpy as np


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction_bool = prediction.astype(bool)
    target_bool = target.astype(bool)
    union = np.logical_or(prediction_bool, target_bool).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction_bool, target_bool).sum() / union)


def tracking_metrics(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, float]:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    ious = np.array([mask_iou(prediction, target) for prediction, target in zip(predictions, targets, strict=True)])
    visible = np.array([target.astype(bool).any() for target in targets], dtype=bool)
    predicted_visible = np.array([prediction.astype(bool).any() for prediction in predictions], dtype=bool)
    empty_empty = ~visible & ~predicted_visible
    successful_visible = np.array(
        [np.logical_and(pred.astype(bool), target.astype(bool)).any() for pred, target in zip(predictions, targets, strict=True)],
        dtype=bool,
    ) & visible
    tracking_recall = float((ious[visible] > 0).mean()) if visible.any() else 0.0
    accuracy = float(ious[successful_visible].mean()) if successful_visible.any() else 0.0
    visible_iou = float(ious[visible].mean()) if visible.any() else 0.0
    absent_iou = float(ious[~visible].mean()) if (~visible).any() else 0.0
    return {
        "iou": float(ious.mean()) if len(ious) else 0.0,
        "visible_iou": visible_iou,
        "absent_iou": absent_iou,
        "tracking_recall": tracking_recall,
        "accuracy": accuracy,
        "num_frames": float(len(ious)),
        "num_visible_frames": float(visible.sum()),
        "num_absent_frames": float((~visible).sum()),
        "empty_empty_frames": float(empty_empty.sum()),
    }
