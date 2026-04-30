from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .metrics import tracking_metrics


def load_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path)) > 0


def choose_conditioning_frame(mask_paths: list[str | Path]) -> int:
    areas = [int(load_mask(path).sum()) for path in mask_paths]
    if not areas:
        raise ValueError("mask_paths is empty")
    return int(np.argmax(areas))


def evaluate_saved_masks(
    prediction_paths: list[str | Path],
    target_paths: list[str | Path],
    output_path: str | Path,
    *,
    conditioning_frame: int | None = None,
) -> dict[str, float]:
    predictions = [load_mask(path) for path in prediction_paths]
    targets = [load_mask(path) for path in target_paths]
    metrics = tracking_metrics(predictions, targets)
    selected_conditioning_frame = choose_conditioning_frame(target_paths) if conditioning_frame is None else int(conditioning_frame)
    metrics["conditioning_frame"] = float(selected_conditioning_frame)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics
