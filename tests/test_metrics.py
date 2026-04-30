from __future__ import annotations

import json

import numpy as np
from PIL import Image

from three_am.evaluation.metrics import mask_iou, tracking_metrics
from three_am.evaluation.track2d import evaluate_saved_masks


def test_mask_iou_empty_masks_is_one() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    assert mask_iou(mask, mask) == 1.0


def test_tracking_metrics_visible_frames() -> None:
    target = [np.array([[1, 0], [0, 0]], dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)]
    prediction = [np.array([[1, 0], [0, 0]], dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)]
    metrics = tracking_metrics(prediction, target)
    assert metrics["iou"] == 1.0
    assert metrics["visible_iou"] == 1.0
    assert metrics["absent_iou"] == 1.0
    assert metrics["tracking_recall"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["num_absent_frames"] == 1.0
    assert metrics["empty_empty_frames"] == 1.0


def test_tracking_metrics_reports_empty_empty_contribution_separately() -> None:
    target = [np.zeros((2, 2), dtype=np.uint8), np.array([[1, 0], [0, 0]], dtype=np.uint8)]
    prediction = [np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)]
    metrics = tracking_metrics(prediction, target)

    assert metrics["iou"] == 0.5
    assert metrics["visible_iou"] == 0.0
    assert metrics["absent_iou"] == 1.0
    assert metrics["tracking_recall"] == 0.0
    assert metrics["accuracy"] == 0.0
    assert metrics["empty_empty_frames"] == 1.0


def test_evaluate_saved_masks_records_paper_conditioning_frame(tmp_path) -> None:
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    masks = [
        np.zeros((4, 4), dtype=np.uint8),
        np.pad(np.ones((1, 1), dtype=np.uint8) * 255, ((0, 3), (0, 3))),
        np.pad(np.ones((2, 2), dtype=np.uint8) * 255, ((0, 2), (0, 2))),
    ]
    for index, mask in enumerate(masks):
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(pred_dir / f"{index:03d}.png")
        Image.fromarray(mask).save(gt_dir / f"{index:03d}.png")

    metrics = evaluate_saved_masks(
        sorted(pred_dir.glob("*.png")),
        sorted(gt_dir.glob("*.png")),
        tmp_path / "metrics.json",
    )

    assert metrics["conditioning_frame"] == 2.0
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["conditioning_frame"] == 2.0
